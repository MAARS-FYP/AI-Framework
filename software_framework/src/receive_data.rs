use crate::uart::Uart;
use std::io::{self, Read, Write};
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

    /// Display packet contents for debugging (show first few samples)
    pub fn display_debug_info(&self, packet_num: usize) {
        let num_samples = self.data.len() / 4;
        eprintln!("=== UDP Packet #{} ===", packet_num);
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

pub fn receive_udp_data(buffer: Arc<Mutex<CircularBuffer<IQSample>>>) -> io::Result<()> {
    // Try binding to all interfaces; can be overridden with --udp-bind
    receive_udp_data_with_bind("172.25.122.155:62510", buffer, false)
}

pub fn receive_udp_data_with_bind(
    bind_addr: &str,
    buffer: Arc<Mutex<CircularBuffer<IQSample>>>,
    print_udp_input: bool,
) -> io::Result<()> {
    let socket = UdpSocket::bind(bind_addr)?;
    eprintln!("UDP socket bound to: {}", bind_addr);
    let mut buf = [0; 1024];
    let mut packet_count = 0u64;

    loop {
        let (amt, src) = socket.recv_from(&mut buf)?;
        packet_count += 1;

        // All 1024 bytes are raw Q,I interleaved i16 data - no header
        let data = buf[0..amt].to_vec();

        let sample = IQSample { data: data.clone() };
        
        if print_udp_input {
            eprintln!("\n[Packet {}] Received {} bytes from {}", packet_count, amt, src);
            sample.display_debug_info(packet_count as usize);
        }
        
        buffer.lock().unwrap().write(sample);
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