use serialport::SerialPort;
use std::io::{self, Read, Write};
use std::time::Duration;

pub struct UartConfig {
    pub port: String,
    pub baud_rate: u32,
    pub timeout_ms: u64,
}

impl UartConfig {
    pub fn new(port: &str, baud_rate: u32) -> Self {
        let port = normalize_uart_port(port);
        Self {
            port: port.into(),
            baud_rate,
            timeout_ms: 100,
        }
    }
}

pub struct Uart {
    inner: Box<dyn SerialPort>,
}

impl Uart {
    pub fn open(config: &UartConfig) -> io::Result<Self> {
        let port = serialport::new(&config.port, config.baud_rate)
            .timeout(Duration::from_millis(config.timeout_ms))
            .open()
            .map_err(|e| {
                let available_ports = match serialport::available_ports() {
                    Ok(ports) if ports.is_empty() => "none detected".to_string(),
                    Ok(ports) => ports
                        .into_iter()
                        .map(|port| port.port_name)
                        .collect::<Vec<_>>()
                        .join(", "),
                    Err(list_err) => format!("unavailable ({})", list_err),
                };
                io::Error::new(
                    io::ErrorKind::Other,
                    format!(
                        "Failed to open UART port '{}' at {} baud (timeout {} ms): {}. On macOS/Linux, pass the full device path such as '/dev/cu.usbmodem11203' or '/dev/ttyUSB0' rather than a bare device name. Available serial ports: {}.",
                        config.port,
                        config.baud_rate,
                        config.timeout_ms,
                        e,
                        available_ports,
                    ),
                )
            })?;
        Ok(Self { inner: port })
    }

    /// Clone the port handle for use in a separate thread (e.g. one for read, one for write)
    pub fn try_clone(&self) -> io::Result<Self> {
        let cloned = self
            .inner
            .try_clone()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(Self { inner: cloned })
    }
}

impl Read for Uart {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl Write for Uart {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn normalize_uart_port(port: &str) -> String {
    if port.starts_with('/') {
        return port.to_string();
    }

    if port.starts_with("cu.") || port.starts_with("tty.") {
        return format!("/dev/{}", port);
    }

    port.to_string()
}
