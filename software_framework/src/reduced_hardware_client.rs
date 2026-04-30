use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;

use serde_json::json;

#[derive(Debug, Clone)]
pub struct ReducedHardwareResponse {
    pub seq_id: u64,
    pub center_class: u8,
    pub bandwidth_class: u8,
    pub center_frequency_mhz: f32,
    pub bandwidth_mhz: f32,
    pub center_confidence: f32,
    pub bandwidth_confidence: f32,
}

pub struct ReducedHardwareInferenceClient {
    socket_path: String,
    stream: Option<UnixStream>,
}

impl ReducedHardwareInferenceClient {
    pub fn new(socket_path: &str) -> Self {
        Self {
            socket_path: socket_path.to_string(),
            stream: None,
        }
    }

    pub fn connect(&mut self) -> io::Result<()> {
        let stream = UnixStream::connect(&self.socket_path)?;
        self.stream = Some(stream);
        Ok(())
    }

    fn ensure_connected(&mut self) -> io::Result<()> {
        if self.stream.is_none() {
            self.connect()?;
        }
        Ok(())
    }

    fn send_request(&mut self, payload: serde_json::Value) -> io::Result<ReducedHardwareResponse> {
        self.ensure_connected()?;

        let stream = self.stream.as_mut().unwrap();
        let mut bytes = serde_json::to_vec(&payload).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Failed to serialize request: {}", e))
        })?;
        bytes.push(b'\n');

        if let Err(err) = stream.write_all(&bytes) {
            self.stream = None;
            return Err(err);
        }
        stream.flush()?;

        let mut response = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let n = stream.read(&mut chunk)?;
            if n == 0 {
                break;
            }
            response.extend_from_slice(&chunk[..n]);
            if chunk[..n].contains(&b'\n') {
                break;
            }
        }

        let line = String::from_utf8(response)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Invalid UTF-8 response: {}", e)))?;
        let line = line.lines().next().unwrap_or("").trim();
        if line.is_empty() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Empty response from reduced-hardware worker"));
        }

        let value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid JSON response: {}", e))
        })?;

        if value.get("status").and_then(|v| v.as_str()) == Some("error") {
            let message = value
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown reduced-hardware worker error");
            return Err(io::Error::new(io::ErrorKind::Other, message.to_string()));
        }

        let seq_id = value.get("seq_id").and_then(|v| v.as_u64()).unwrap_or(0);
        let center_class = value.get("center_class").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
        let bandwidth_class = value.get("bandwidth_class").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
        let center_frequency_mhz = value
            .get("center_frequency_mhz")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let bandwidth_mhz = value
            .get("bandwidth_mhz")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let center_confidence = value
            .get("center_confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let bandwidth_confidence = value
            .get("bandwidth_confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        Ok(ReducedHardwareResponse {
            seq_id,
            center_class,
            bandwidth_class,
            center_frequency_mhz,
            bandwidth_mhz,
            center_confidence,
            bandwidth_confidence,
        })
    }

    pub fn infer_capture_path(&mut self, seq_id: u64, capture_csv_path: &str, sample_rate_hz: f64) -> io::Result<ReducedHardwareResponse> {
        self.send_request(json!({
            "type": "infer",
            "seq_id": seq_id,
            "capture_csv_path": capture_csv_path,
            "sample_rate_hz": sample_rate_hz,
        }))
    }

    pub fn infer_samples(&mut self, seq_id: u64, adc_samples: &[f32], sample_rate_hz: f64) -> io::Result<ReducedHardwareResponse> {
        self.send_request(json!({
            "type": "infer",
            "seq_id": seq_id,
            "adc_samples": adc_samples,
            "sample_rate_hz": sample_rate_hz,
        }))
    }

    pub fn ping(&mut self, seq_id: u64) -> io::Result<()> {
        let response = self.send_request(json!({
            "type": "ping",
            "seq_id": seq_id,
        }))?;
        if response.seq_id != seq_id {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Ping sequence id mismatch"));
        }
        Ok(())
    }
}