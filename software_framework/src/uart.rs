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
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
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
