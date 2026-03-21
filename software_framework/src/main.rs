mod inference_client;
mod protocol;
mod receive_data;
mod send_data;
mod uart;

use inference_client::InferenceSocketClient;
use protocol::InferenceRequest;
use receive_data::{CircularBuffer, IQSample, PowerMeasurement};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
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

    let mut inference_client = InferenceSocketClient::new("/tmp/maars_infer.sock");
    if let Err(e) = inference_client.connect() {
        eprintln!("Inference worker not reachable at startup: {}", e);
    }

    let mut latest_power: Option<PowerMeasurement> = None;
    let mut seq_id: u64 = 1;

    // Main processing loop
    loop {
        if let Some(power) = uart_buffer.lock().unwrap().read() {
            println!(
                "Power - Timestamp: {}, Sensor1: {}, Sensor2: {}",
                power.timestamp, power.sensor1, power.sensor2
            );
            latest_power = Some(power);
        }

        if let Some(iq_sample) = udp_buffer.lock().unwrap().read() {
            println!(
                "IQ Sample - Timestamp: {}, Data len: {}",
                iq_sample.timestamp,
                iq_sample.data.len()
            );

            let Some(power) = latest_power.clone() else {
                eprintln!("Skipping inference: no power sample received yet");
                thread::sleep(std::time::Duration::from_millis(10));
                continue;
            };

            let iq_iq_pairs = match iq_sample.parse_qi_i16_be_to_iq_f32() {
                Ok(v) if !v.is_empty() => v,
                Ok(_) => {
                    eprintln!("Skipping inference: empty IQ payload");
                    thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
                Err(e) => {
                    eprintln!("Skipping inference: bad IQ payload: {}", e);
                    thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
            };

            let request = InferenceRequest {
                seq_id,
                sample_rate_hz: 25_000_000.0,
                power_lna_dbm: power.sensor1,
                power_pa_dbm: power.sensor2,
                iq_iq_pairs: &iq_iq_pairs,
            };

            match inference_client.infer(&request) {
                Ok(resp) => {
                    let uart_msg = format!(
                        "SEQ={} STATUS={} LNA={} FILTER={} CENTER={} MIXER_DBM={:.3} IFAMP_DB={:.3} EVM={:.3} PT_MS={:.3}\n",
                        resp.seq_id,
                        resp.status_code,
                        resp.lna_class,
                        resp.filter_class,
                        resp.center_class,
                        resp.mixer_dbm,
                        resp.ifamp_db,
                        resp.evm_value,
                        resp.processing_time_ms
                    );

                    if let Err(e) = uart_tx.send(uart_msg.into_bytes()) {
                        eprintln!("UART command send failed: {}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Inference request failed: {}", e);
                    if let Err(connect_err) = inference_client.connect() {
                        eprintln!("Inference reconnect failed: {}", connect_err);
                    }
                }
            }

            seq_id += 1;
        }

        thread::sleep(std::time::Duration::from_millis(10));
    }
}
