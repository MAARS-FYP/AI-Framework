use crate::uart::Uart;
use std::io::{self, Read};
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};

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
    pub timestamp: u64,
    pub sensor1: f32,
    pub sensor2: f32,
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

/// Reads UART packets: timestamp(8) + sensor1(4) + sensor2(4) = 16 bytes each
pub fn receive_uart_data(
    mut uart: Uart,
    buffer: Arc<Mutex<CircularBuffer<PowerMeasurement>>>,
) -> io::Result<()> {
    let mut buf = [0u8; 16];

    loop {
        match uart.read_exact(&mut buf) {
            Ok(()) => {
                let timestamp = u64::from_be_bytes(buf[0..8].try_into().unwrap());
                let sensor1 = f32::from_be_bytes(buf[8..12].try_into().unwrap());
                let sensor2 = f32::from_be_bytes(buf[12..16].try_into().unwrap());
                buffer.lock().unwrap().write(PowerMeasurement {
                    timestamp,
                    sensor1,
                    sensor2,
                });
            }
            Err(ref e) if e.kind() == io::ErrorKind::TimedOut => continue,
            Err(e) => return Err(e),
        }
    }
}