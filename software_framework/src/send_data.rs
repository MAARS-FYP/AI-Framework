use crate::uart::Uart;
use std::io::{self, Write};
use std::sync::mpsc::Receiver;

/// Sends data received from the channel over UART.
/// Feed data from any thread/function via the corresponding Sender<Vec<u8>>.
pub fn send_uart_data(mut uart: Uart, rx: Receiver<Vec<u8>>) -> io::Result<()> {
    for data in rx {
        uart.write_all(&data)?;
        uart.flush()?;
    }
    Ok(())
}
