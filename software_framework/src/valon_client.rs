use crate::protocol::InferenceResponse;
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::mpsc::Receiver;
use std::time::Duration;

const IF_OFFSET_MHZ: f32 = 25.0;
const VALUE_SCALE_MILLI: f32 = 1000.0;
const CENTER_FREQS_MHZ: [f32; 3] = [2405.0, 2420.0, 2435.0];

#[derive(Debug, Clone, Copy)]
pub struct ValonControlValues {
    pub lo_freq_mhz: f32,
    pub rf_level_dbm: f32,
    lo_freq_milli: i32,
    rf_level_milli: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LastSentValues {
    lo_freq_milli: i32,
    rf_level_milli: i32,
}

#[derive(Debug, Default)]
pub struct ValonCommandTracker {
    last_sent: Option<LastSentValues>,
}

#[derive(Debug, Clone, Copy)]
pub enum ValonCommand {
    SetFreq { seq_id: u64, value_mhz: f32 },
    SetRfLevel { seq_id: u64, value_dbm: f32 },
}

pub fn map_valon_controls(resp: &InferenceResponse) -> io::Result<ValonControlValues> {
    let center_freq_mhz = CENTER_FREQS_MHZ
        .get(resp.center_class as usize)
        .copied()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported center_class value: {}", resp.center_class),
            )
        })?;

    let lo_freq_mhz = center_freq_mhz - IF_OFFSET_MHZ;
    let rf_level_dbm = resp.mixer_dbm;
    let lo_freq_milli = (lo_freq_mhz * VALUE_SCALE_MILLI).round() as i32;
    let rf_level_milli = (rf_level_dbm * VALUE_SCALE_MILLI).round() as i32;

    Ok(ValonControlValues {
        lo_freq_mhz,
        rf_level_dbm,
        lo_freq_milli,
        rf_level_milli,
    })
}

impl ValonCommandTracker {
    pub fn commands_for(&mut self, seq_id: u64, values: &ValonControlValues) -> Vec<ValonCommand> {
        let mut commands = Vec::new();
        let next = LastSentValues {
            lo_freq_milli: values.lo_freq_milli,
            rf_level_milli: values.rf_level_milli,
        };

        match self.last_sent {
            None => {
                commands.push(ValonCommand::SetFreq {
                    seq_id,
                    value_mhz: values.lo_freq_mhz,
                });
                commands.push(ValonCommand::SetRfLevel {
                    seq_id,
                    value_dbm: values.rf_level_dbm,
                });
            }
            Some(prev) => {
                if prev.lo_freq_milli != next.lo_freq_milli {
                    commands.push(ValonCommand::SetFreq {
                        seq_id,
                        value_mhz: values.lo_freq_mhz,
                    });
                }
                if prev.rf_level_milli != next.rf_level_milli {
                    commands.push(ValonCommand::SetRfLevel {
                        seq_id,
                        value_dbm: values.rf_level_dbm,
                    });
                }
            }
        }

        self.last_sent = Some(next);
        commands
    }
}

pub fn send_valon_commands(
    socket_path: &str,
    rx: Receiver<ValonCommand>,
    print_logs: bool,
) -> io::Result<()> {
    let mut client = ValonSocketClient::new(socket_path.to_string(), print_logs);
    for command in rx {
        if let Err(e) = client.send_command(&command) {
            eprintln!("Valon sender warning: {}", e);
            client.disconnect();
        }
    }
    Ok(())
}

struct ValonSocketClient {
    socket_path: String,
    stream: Option<UnixStream>,
    next_request_id: u64,
    read_buffer: Vec<u8>,
    print_logs: bool,
}

impl ValonSocketClient {
    fn new(socket_path: String, print_logs: bool) -> Self {
        Self {
            socket_path,
            stream: None,
            next_request_id: 1,
            read_buffer: Vec::with_capacity(1024),
            print_logs,
        }
    }

    fn disconnect(&mut self) {
        self.stream = None;
    }

    fn ensure_connected(&mut self) -> io::Result<()> {
        if self.stream.is_some() {
            return Ok(());
        }

        let stream = UnixStream::connect(&self.socket_path).map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotConnected,
                format!("Failed to connect Valon socket {}: {}", self.socket_path, e),
            )
        })?;
        stream.set_write_timeout(Some(Duration::from_millis(200)))?;
        stream.set_read_timeout(Some(Duration::from_millis(2)))?;

        if self.print_logs {
            println!("VALON socket connected: {}", self.socket_path);
        }
        self.stream = Some(stream);
        Ok(())
    }

    fn send_command(&mut self, command: &ValonCommand) -> io::Result<()> {
        self.ensure_connected()?;

        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);
        let payload = command.to_json_line(request_id);

        let write_result = if let Some(stream) = self.stream.as_mut() {
            stream.write_all(payload.as_bytes())
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "Valon socket became unavailable before write",
            ))
        };

        if let Err(e) = write_result {
            self.stream = None;
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!(
                    "Failed to send {} for seq {} to Valon: {}",
                    command.name(),
                    command.seq_id(),
                    e
                ),
            ));
        }

        if self.print_logs {
            println!(
                "VALON seq={} tx={}",
                command.seq_id(),
                payload.trim_end_matches('\n')
            );
        }

        let _ = self.drain_one_response();
        Ok(())
    }

    fn drain_one_response(&mut self) -> io::Result<()> {
        let mut chunk = [0u8; 1024];
        let read_result = if let Some(stream) = self.stream.as_mut() {
            stream.read(&mut chunk)
        } else {
            return Ok(());
        };

        match read_result {
            Ok(0) => {
                self.stream = None;
                Ok(())
            }
            Ok(n) => {
                self.read_buffer.extend_from_slice(&chunk[..n]);
                while let Some(pos) = self.read_buffer.iter().position(|&b| b == b'\n') {
                    let line = self.read_buffer.drain(..=pos).collect::<Vec<u8>>();
                    let text = String::from_utf8_lossy(&line).trim().to_string();
                    if text.is_empty() {
                        continue;
                    }
                    if text.contains("\"ok\":false") {
                        eprintln!("VALON response error: {}", text);
                    } else if self.print_logs {
                        println!("VALON rx={}", text);
                    }
                }
                Ok(())
            }
            Err(ref e)
                if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut =>
            {
                Ok(())
            }
            Err(e) => {
                self.stream = None;
                Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    format!("Valon response read failed: {}", e),
                ))
            }
        }
    }
}

impl ValonCommand {
    fn name(&self) -> &'static str {
        match self {
            Self::SetFreq { .. } => "set_freq",
            Self::SetRfLevel { .. } => "set_rflevel",
        }
    }

    fn seq_id(&self) -> u64 {
        match self {
            Self::SetFreq { seq_id, .. } | Self::SetRfLevel { seq_id, .. } => *seq_id,
        }
    }

    fn to_json_line(&self, request_id: u64) -> String {
        match self {
            Self::SetFreq { value_mhz, .. } => {
                format!(
                    "{{\"id\":\"{}\",\"op\":\"set_freq\",\"value_mhz\":{:.3}}}\n",
                    request_id, value_mhz
                )
            }
            Self::SetRfLevel { value_dbm, .. } => {
                format!(
                    "{{\"id\":\"{}\",\"op\":\"set_rflevel\",\"value_dbm\":{:.3}}}\n",
                    request_id, value_dbm
                )
            }
        }
    }
}