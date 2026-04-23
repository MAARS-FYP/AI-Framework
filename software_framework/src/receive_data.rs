use crate::uart::Uart;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const ADC_MAX_U12: f32 = 4095.0;
const ADC_VREF_VOLTS: f32 = 3.3;
const SENSOR_MIN_VOLTS: f32 = 0.2;
const SENSOR_MAX_VOLTS: f32 = 1.7;
const SENSOR_MIN_DBM: f32 = -30.0;
const SENSOR_MAX_DBM: f32 = 15.0;
const SENSOR_DBM_BIAS: f32 = 10.0;

// Data structures for parsed data
#[derive(Clone, Debug)]
pub struct IQSample {
    pub data: Vec<u8>,
}

impl IQSample {
    /// Parse Q,I interleaved i16 format: [Q0, I0, Q1, I1, ...]
    /// Each value is a 2-byte little-endian signed integer
    pub fn parse_qi_interleaved_i16_to_iq_f32(&self) -> io::Result<Vec<(f32, f32)>> {
        if self.data.len() % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IQ payload length must be divisible by 4 bytes (Q,I pair = 4 bytes)",
            ));
        }

        let mut out = Vec::with_capacity(self.data.len() / 4);
        for chunk in self.data.chunks_exact(4) {
            // Q is first 2 bytes, I is next 2 bytes, both as little-endian i16
            let q_bytes = [chunk[0], chunk[1]];
            let i_bytes = [chunk[2], chunk[3]];
            let q = i16::from_le_bytes(q_bytes) as f32;
            let i = i16::from_le_bytes(i_bytes) as f32;
            out.push((i, q));
        }
        Ok(out)
    }

    /// Display sample payload contents for debugging (show first few samples)
    pub fn display_debug_info(&self, sample_num: usize, source_label: &str) {
        let num_samples = self.data.len() / 4;
        eprintln!("=== {} Sample #{} ===", source_label, sample_num);
        eprintln!("Size: {} bytes ({} I/Q samples)", self.data.len(), num_samples);
        
        // Show first 5 samples
        let show_count = std::cmp::min(5, num_samples);
        eprintln!("First {} samples (Q, I):", show_count);
        for i in 0..show_count {
            let offset = i * 4;
            let q_bytes = [self.data[offset], self.data[offset + 1]];
            let i_bytes = [self.data[offset + 2], self.data[offset + 3]];
            let q = i16::from_le_bytes(q_bytes);
            let i = i16::from_le_bytes(i_bytes);
            eprintln!("  Sample {}: Q={}, I={}", i, q, i);
        }
        eprintln!("");
    }
}

#[derive(Clone, Debug)]
pub struct PowerMeasurement {
    pub power_lna_raw: f32,
    pub power_pa_raw: f32,
}

// Simple circular buffer
pub struct CircularBuffer<T> {
    buffer: Vec<Option<T>>,
    write_ptr: usize,
    read_ptr: usize,
    capacity: usize,
}

impl<T: Clone> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![None; capacity],
            write_ptr: 0,
            read_ptr: 0,
            capacity,
        }
    }

    pub fn write(&mut self, item: T) {
        self.buffer[self.write_ptr] = Some(item);
        self.write_ptr = (self.write_ptr + 1) % self.capacity;
    }

    pub fn read(&mut self) -> Option<T> {
        if self.read_ptr == self.write_ptr {
            return None;
        }
        let item = self.buffer[self.read_ptr].clone();
        self.read_ptr = (self.read_ptr + 1) % self.capacity;
        item
    }

    pub fn available(&self) -> usize {
        if self.write_ptr >= self.read_ptr {
            self.write_ptr - self.read_ptr
        } else {
            self.capacity - self.read_ptr + self.write_ptr
        }
    }
}

pub fn receive_ila_csv_probe0_data(
    csv_path: &str,
    request_flag_path: &str,
    buffer: Arc<Mutex<CircularBuffer<IQSample>>>,
    print_ila_input: bool,
    poll_interval_ms: u64,
    request_timeout_ms: u64,
    batch_samples: usize,
) -> io::Result<()> {
    let csv_path = Path::new(csv_path);
    let request_flag_path = Path::new(request_flag_path);
    let poll_sleep = Duration::from_millis(poll_interval_ms.max(1));
    let mut sample_count: u64 = 0;

    loop {
        request_capture_if_needed(request_flag_path)?;

        wait_for_capture_ready(request_flag_path, poll_sleep, request_timeout_ms)?;

        let Some(data) = collect_probe0_iq_bytes(csv_path, batch_samples)? else {
            std::thread::sleep(poll_sleep);
            continue;
        };

        sample_count += 1;
        let sample = IQSample { data };

        if print_ila_input {
            eprintln!(
                "\n[ILA Capture {}] parsed {} I/Q samples from {}",
                sample_count,
                sample.data.len() / 4,
                csv_path.display()
            );
            sample.display_debug_info(sample_count as usize, "ILA CSV");
        }

        buffer.lock().unwrap().write(sample);
        truncate_file(csv_path)?;
    }
}

pub fn receive_uart_adc_measurements(
    mut uart: Uart,
    buffer: Arc<Mutex<CircularBuffer<PowerMeasurement>>>,
    print_uart_input: bool,
) -> io::Result<()> {
    const ADC_CMD: &[u8] = b"adc read\r\n";
    const MAX_FRAMES_PER_REQUEST: usize = 8;

    loop {
        uart.write_all(ADC_CMD)?;
        uart.flush()?;

        let mut parsed_sample: Option<(f32, f32)> = None;
        for _ in 0..MAX_FRAMES_PER_REQUEST {
            let response = match read_uart_response_frame(&mut uart) {
                Ok(v) => v,
                Err(ref e) if e.kind() == io::ErrorKind::TimedOut => break,
                Err(e) => return Err(e),
            };

            if let Some(v) = parse_ascii_power_csv(&response) {
                parsed_sample = Some(v);
                break;
            }

            if print_uart_input {
                let raw = String::from_utf8_lossy(&response);
                eprintln!("UART ignored frame: {:?}", raw.trim_end_matches(['\r', '\n']));
            }
        }

        let (power_lna_raw, power_pa_raw) = match parsed_sample {
            Some(v) => v,
            None => {
                if print_uart_input {
                    eprintln!("UART: no valid CSV sample in response to adc read");
                }
                continue;
            }
        };

        buffer.lock().unwrap().write(PowerMeasurement {
            power_lna_raw,
            power_pa_raw,
        });

        if print_uart_input {
            let lna_dbm = calibrate_power_raw_to_dbm(power_lna_raw);
            let pa_dbm = calibrate_power_raw_to_dbm(power_pa_raw);
            eprintln!(
                "UART power: lna_raw={:.0} lna_dbm={:.3} pa_raw={:.0} pa_dbm={:.3}",
                power_lna_raw,
                lna_dbm,
                power_pa_raw,
                pa_dbm,
            );
        }

        std::thread::sleep(Duration::from_millis(20));
    }
}

fn read_uart_response_frame(uart: &mut Uart) -> io::Result<Vec<u8>> {
    let mut response = Vec::with_capacity(64);
    let mut byte = [0u8; 1];

    loop {
        match uart.read(&mut byte) {
            Ok(0) => break,
            Ok(1) => {
                response.push(byte[0]);
                if byte[0] == b'\n' || response.len() >= 64 {
                    break;
                }
            }
            Ok(_) => unreachable!(),
            Err(ref e) if e.kind() == io::ErrorKind::TimedOut => {
                if !response.is_empty() {
                    break;
                }
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "No UART response frame",
                ));
            }
            Err(e) => return Err(e),
        }
    }

    if response.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "No UART response frame",
        ));
    }

    Ok(response)
}

fn request_capture_if_needed(request_flag_path: &Path) -> io::Result<()> {
    if request_flag_path.exists() {
        return Ok(());
    }

    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(request_flag_path)?;
    f.write_all(b"capture\n")?;
    f.flush()?;
    Ok(())
}

fn wait_for_capture_ready(
    request_flag_path: &Path,
    poll_sleep: Duration,
    request_timeout_ms: u64,
) -> io::Result<()> {
    let timeout = Duration::from_millis(request_timeout_ms.max(1));
    let start = Instant::now();

    while request_flag_path.exists() {
        if start.elapsed() > timeout {
            eprintln!(
                "ILA capture timeout waiting for request flag to clear: {}",
                request_flag_path.display()
            );
            // Clear stale request so a new capture can be requested next cycle.
            let _ = fs::remove_file(request_flag_path);
            return Ok(());
        }
        std::thread::sleep(poll_sleep);
    }

    Ok(())
}

fn collect_probe0_iq_bytes(csv_path: &Path, batch_samples: usize) -> io::Result<Option<Vec<u8>>> {
    if !csv_path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(csv_path)?;
    if content.trim().is_empty() {
        return Ok(None);
    }

    let mut out = Vec::with_capacity(batch_samples * 4);
    let mut parsed_rows = 0usize;

    for line in content.lines() {
        let Some(word) = parse_probe0_word(line) else {
            continue;
        };

        let bytes = decode_probe0_qi_word(word);
        out.extend_from_slice(&bytes);
        parsed_rows += 1;

        if parsed_rows >= batch_samples {
            break;
        }
    }

    if parsed_rows < batch_samples {
        return Ok(None);
    }

    Ok(Some(out))
}

fn parse_probe0_word(line: &str) -> Option<u32> {
    let first_field = line.split(',').next()?.trim();
    if first_field.is_empty() {
        return None;
    }

    if let Some(hex) = first_field
        .strip_prefix("0x")
        .or_else(|| first_field.strip_prefix("0X"))
    {
        return u32::from_str_radix(hex, 16).ok();
    }

    first_field.parse::<u32>().ok()
}

fn decode_probe0_qi_word(word: u32) -> [u8; 4] {
    let q_u16 = ((word >> 16) & 0xFFFF) as u16;
    let i_u16 = (word & 0xFFFF) as u16;
    let q_bytes = q_u16.to_le_bytes();
    let i_bytes = i_u16.to_le_bytes();
    [q_bytes[0], q_bytes[1], i_bytes[0], i_bytes[1]]
}

fn truncate_file(path: &Path) -> io::Result<()> {
    if !path.exists() {
        return Ok(());
    }

    OpenOptions::new().write(true).truncate(true).open(path)?;
    Ok(())
}

fn parse_ascii_power_csv(response: &[u8]) -> Option<(f32, f32)> {
    let text = String::from_utf8_lossy(response);
    let trimmed = text.trim();

    if trimmed.is_empty() || trimmed.chars().any(char::is_whitespace) {
        return None;
    }

    let mut parts = trimmed.split(',');
    let lna = parts.next()?.parse::<f32>().ok()?;
    let pa = parts.next()?.parse::<f32>().ok()?;
    if parts.next().is_some() {
        return None;
    }

    Some((lna, pa))
}

fn calibrate_power_raw_to_dbm(raw_u12: f32) -> f32 {
    let raw = raw_u12.clamp(0.0, ADC_MAX_U12);
    let voltage = (raw / ADC_MAX_U12) * ADC_VREF_VOLTS;
    let clamped_voltage = voltage.clamp(SENSOR_MIN_VOLTS, SENSOR_MAX_VOLTS);
    let span = SENSOR_MAX_VOLTS - SENSOR_MIN_VOLTS;
    let ratio = if span > 0.0 {
        (clamped_voltage - SENSOR_MIN_VOLTS) / span
    } else {
        0.0
    };
    let dbm = SENSOR_MIN_DBM + ratio * (SENSOR_MAX_DBM - SENSOR_MIN_DBM);
    dbm + SENSOR_DBM_BIAS
}