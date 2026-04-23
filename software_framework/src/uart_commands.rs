use crate::protocol::InferenceResponse;
use std::io;

const IF_AMP_SCALE_MILLI: f32 = 1000.0;

#[derive(Debug, Clone, Copy)]
pub struct AgentControlValues {
    pub lna_voltage: u8,
    pub filter_mhz: u8,
    pub if_amp_value: f32,
    if_amp_milli: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LastSentValues {
    lna_voltage: u8,
    filter_mhz: u8,
    if_amp_milli: i32,
}

#[derive(Debug, Default)]
pub struct CommandTracker {
    last_sent: Option<LastSentValues>,
}

pub fn map_agent_controls(resp: &InferenceResponse) -> io::Result<AgentControlValues> {
    let lna_voltage = match resp.lna_class {
        0 => 3,
        1 => 5,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported lna_class value: {}", other),
            ));
        }
    };

    let filter_mhz = match resp.filter_class {
        0 => 1,
        1 => 10,
        2 => 20,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported filter_class value: {}", other),
            ));
        }
    };

    let if_amp_milli = (resp.ifamp_db * IF_AMP_SCALE_MILLI).round() as i32;
    let if_amp_value = if_amp_milli as f32 / IF_AMP_SCALE_MILLI;

    Ok(AgentControlValues {
        lna_voltage,
        filter_mhz,
        if_amp_value,
        if_amp_milli,
    })
}

impl CommandTracker {
    pub fn commands_for(&mut self, values: &AgentControlValues) -> Vec<Vec<u8>> {
        let mut commands: Vec<Vec<u8>> = Vec::new();

        let next = LastSentValues {
            lna_voltage: values.lna_voltage,
            filter_mhz: values.filter_mhz,
            if_amp_milli: values.if_amp_milli,
        };

        match self.last_sent {
            None => {
                commands.push(build_lna_command(next.lna_voltage));
                commands.push(build_filter_command(next.filter_mhz));
                commands.push(build_ifamp_command(values.if_amp_value));
            }
            Some(prev) => {
                if prev.lna_voltage != next.lna_voltage {
                    commands.push(build_lna_command(next.lna_voltage));
                }
                if prev.filter_mhz != next.filter_mhz {
                    commands.push(build_filter_command(next.filter_mhz));
                }
                if prev.if_amp_milli != next.if_amp_milli {
                    commands.push(build_ifamp_command(values.if_amp_value));
                }
            }
        }

        self.last_sent = Some(next);
        commands
    }
}

fn build_lna_command(voltage: u8) -> Vec<u8> {
    format!("lna {}\r\n", voltage).into_bytes()
}

fn build_filter_command(filter_mhz: u8) -> Vec<u8> {
    format!("filter {}\r\n", filter_mhz).into_bytes()
}

fn build_ifamp_command(if_amp_value: f32) -> Vec<u8> {
    format!("ifamp {:.3}\r\n", if_amp_value).into_bytes()
}