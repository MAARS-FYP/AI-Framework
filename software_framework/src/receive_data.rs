use crate::uart::Uart;
use std::io::{self, Read, Write};
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
    receive_udp_data_with_bind("172.25.122.155:62510", buffer)
}

pub fn receive_udp_data_with_bind(
    bind_addr: &str,
    buffer: Arc<Mutex<CircularBuffer<IQSample>>>,
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
        
        // Display debug info
        eprintln!("\n[Packet {}] Received {} bytes from {}", packet_count, amt, src);
        sample.display_debug_info(packet_count as usize);
        
        buffer.lock().unwrap().write(sample);
    }
}

/// Reads UART packets: power_lna_raw(4) + power_pa_raw(4) = 8 bytes each
pub fn receive_uart_data(
    mut uart: Uart,
    buffer: Arc<Mutex<CircularBuffer<PowerMeasurement>>>,
) -> io::Result<()> {
    let mut buf = [0u8; 8];

    loop {
        match uart.read_exact(&mut buf) {
            Ok(()) => {
                let power_lna_raw = f32::from_be_bytes(buf[0..4].try_into().unwrap());
                let power_pa_raw = f32::from_be_bytes(buf[4..8].try_into().unwrap());
                buffer.lock().unwrap().write(PowerMeasurement {
                    power_lna_raw,
                    power_pa_raw,
                });
            }
            Err(ref e) if e.kind() == io::ErrorKind::TimedOut => continue,
            Err(e) => return Err(e),
        }
    }
}

pub fn receive_uart_adc_measurements(
    mut uart: Uart,
    buffer: Arc<Mutex<CircularBuffer<PowerMeasurement>>>,
) -> io::Result<()> {
    const ADC_CMD: &[u8] = b"adc read\r\n";

    loop {
        uart.write_all(ADC_CMD)?;
        uart.flush()?;

        let raw_24 = match read_adc_response_24bit(&mut uart) {
            Ok(v) => v,
            Err(ref e)
                if e.kind() == io::ErrorKind::TimedOut
                    || e.kind() == io::ErrorKind::InvalidData =>
            {
                continue;
            }
            Err(e) => return Err(e),
        };
        let power_lna_raw = ((raw_24 >> 12) & 0x0FFF) as u16;
        let power_pa_raw = (raw_24 & 0x0FFF) as u16;

        buffer.lock().unwrap().write(PowerMeasurement {
            power_lna_raw: power_lna_raw as f32,
            power_pa_raw: power_pa_raw as f32,
        });

        std::thread::sleep(Duration::from_millis(20));
    }
}

fn read_adc_response_24bit(uart: &mut Uart) -> io::Result<u32> {
    const MAX_FRAMES_PER_REQUEST: usize = 8;

    for _ in 0..MAX_FRAMES_PER_REQUEST {
        let response = read_uart_response_frame(uart)?;

        if is_ignorable_uart_frame(&response) {
            continue;
        }

        if let Some(v) = parse_adc_payload(&response) {
            return Ok(v);
        }

        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported ADC UART response: {:02X?}", response),
        ));
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "No valid ADC payload in UART response frames",
    ))
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
                    "No UART response for adc read b",
                ));
            }
            Err(e) => return Err(e),
        }
    }

    if response.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "No UART response for adc read b",
        ));
    }

    Ok(response)
}

fn is_ignorable_uart_frame(response: &[u8]) -> bool {
    let text = String::from_utf8_lossy(response);
    let trimmed = text.trim();

    trimmed.is_empty()
        || trimmed == ">"
        || trimmed.eq_ignore_ascii_case("ok")
    || trimmed.eq_ignore_ascii_case("adc read")
        || trimmed.eq_ignore_ascii_case("adc read b")
}

fn parse_adc_payload(response: &[u8]) -> Option<u32> {
    let text = String::from_utf8_lossy(response);
    let trimmed = text.trim();

    let hex_only: String = response
        .iter()
        .filter_map(|b| {
            let c = *b as char;
            if c.is_ascii_hexdigit() {
                Some(c)
            } else {
                None
            }
        })
        .collect();

    let has_numeric_digit = trimmed.chars().any(|c| c.is_ascii_digit());
    if has_numeric_digit && hex_only.len() >= 6 {
        let s = &hex_only[hex_only.len() - 6..];
        if let Ok(v) = u32::from_str_radix(s, 16) {
            return Some(v & 0x00FF_FFFF);
        }
    }

    let binary: Vec<u8> = response
        .iter()
        .copied()
        .filter(|b| *b != b'\r' && *b != b'\n')
        .collect();

    // Prefer strict 3-byte binary payloads; avoid interpreting echoed ASCII as ADC bytes.
    if binary.len() == 3 {
        let v = ((binary[0] as u32) << 16) | ((binary[1] as u32) << 8) | (binary[2] as u32);
        return Some(v & 0x00FF_FFFF);
    }

    // Common shell/prompt shape: [B0 B1 B2 '>' ' '].
    if binary.len() >= 5 && binary.ends_with(b"> ") {
        let n = binary.len();
        let v = ((binary[n - 5] as u32) << 16)
            | ((binary[n - 4] as u32) << 8)
            | (binary[n - 3] as u32);
        return Some(v & 0x00FF_FFFF);
    }

    // If the frame includes non-ASCII bytes, treat the first three bytes as payload.
    let has_non_ascii = binary.iter().any(|b| !b.is_ascii());
    if has_non_ascii && binary.len() >= 3 {
        let v = ((binary[0] as u32) << 16) | ((binary[1] as u32) << 8) | (binary[2] as u32);
        return Some(v & 0x00FF_FFFF);
    }

    None
}