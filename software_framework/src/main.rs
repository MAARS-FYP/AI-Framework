mod receive_data;
mod send_data;
mod uart;

use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use receive_data::{CircularBuffer, IQSample, PowerMeasurement};
use uart::UartConfig;

fn main() {
    let udp_buffer = Arc::new(Mutex::new(CircularBuffer::<IQSample>::new(1024)));
    let uart_buffer = Arc::new(Mutex::new(CircularBuffer::<PowerMeasurement>::new(1024)));

    // UART setup — one handle for reading, one for writing
    let uart_config = UartConfig::new("/dev/ttyUSB0", 115200); // adjust port as needed
    let uart = uart::Uart::open(&uart_config).expect("Failed to open UART");
    let uart_writer = uart.try_clone().expect("Failed to clone UART handle");

    // Channel for sending data to UART from any thread
    let (uart_tx, uart_rx) = mpsc::channel::<Vec<u8>>();

    // UDP receiver thread
    let udp_buf_clone = Arc::clone(&udp_buffer);
    let _udp_thread = thread::spawn(move || {
        if let Err(e) = receive_data::receive_udp_data(udp_buf_clone) {
            eprintln!("UDP receiver error: {}", e);
        }
    });

    // UART receiver thread
    let uart_buf_clone = Arc::clone(&uart_buffer);
    let _uart_read_thread = thread::spawn(move || {
        if let Err(e) = receive_data::receive_uart_data(uart, uart_buf_clone) {
            eprintln!("UART receiver error: {}", e);
        }
    });

    // UART sender thread
    let _uart_write_thread = thread::spawn(move || {
        if let Err(e) = send_data::send_uart_data(uart_writer, uart_rx) {
            eprintln!("UART sender error: {}", e);
        }
    });

    // Example: send data via the channel from anywhere
    // uart_tx.send(vec![0x01, 0x02, 0x03]).unwrap();

    // Main processing loop
    loop {
        if let Some(iq_sample) = udp_buffer.lock().unwrap().read() {
            println!("IQ Sample - Timestamp: {}, Data len: {}", 
                     iq_sample.timestamp, iq_sample.data.len());
        }

        if let Some(power) = uart_buffer.lock().unwrap().read() {
            println!("Power - Timestamp: {}, Sensor1: {}, Sensor2: {}", 
                     power.timestamp, power.sensor1, power.sensor2);
        }

        thread::sleep(std::time::Duration::from_millis(10));
    }
}

