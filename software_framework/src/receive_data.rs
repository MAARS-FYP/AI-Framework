use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::io::{self, Read};
use crate::uart::Uart;

// Data structures for parsed data
#[derive(Clone, Debug)]
pub struct IQSample {
    pub timestamp: u64,
    pub data: Vec<u8>,
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
    let socket = UdpSocket::bind("127.0.0.1:5000")?;
    let mut buf = [0; 2048];

    loop {
        let (amt, _src) = socket.recv_from(&mut buf)?;
        
        // Parse: assuming first 8 bytes are timestamp (u64), rest is I/Q data
        // Adjust timestamp size as needed
        if amt >= 8 {
            let timestamp = u64::from_be_bytes(buf[0..8].try_into().unwrap());
            let data = buf[8..amt].to_vec();
            
            let sample = IQSample { timestamp, data };
            buffer.lock().unwrap().write(sample);
        }
    }
}

/// Reads UART packets: timestamp(8) + sensor1(4) + sensor2(4) = 16 bytes each
pub fn receive_uart_data(mut uart: Uart, buffer: Arc<Mutex<CircularBuffer<PowerMeasurement>>>) -> io::Result<()> {
    let mut buf = [0u8; 16];

    loop {
        match uart.read_exact(&mut buf) {
            Ok(()) => {
                let timestamp = u64::from_be_bytes(buf[0..8].try_into().unwrap());
                let sensor1 = f32::from_be_bytes(buf[8..12].try_into().unwrap());
                let sensor2 = f32::from_be_bytes(buf[12..16].try_into().unwrap());
                buffer.lock().unwrap().write(PowerMeasurement { timestamp, sensor1, sensor2 });
            }
            Err(ref e) if e.kind() == io::ErrorKind::TimedOut => continue,
            Err(e) => return Err(e),
        }
    }
}

