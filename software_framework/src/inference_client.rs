use std::io;
use std::os::unix::net::UnixStream;

use crate::protocol::{
    InferenceRequest, InferenceResponse, MSG_ERROR_RESP, MSG_INFER_REQ, MSG_INFER_RESP,
    MSG_PING_REQ, MSG_PING_RESP, pack_infer_request, pack_ping, recv_message, send_message,
    unpack_error, unpack_infer_response, unpack_ping,
};

pub struct InferenceSocketClient {
    socket_path: String,
    stream: Option<UnixStream>,
}

impl InferenceSocketClient {
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

    pub fn ensure_connected(&mut self) -> io::Result<()> {
        if self.stream.is_none() {
            self.connect()?;
        }
        Ok(())
    }

    pub fn ping(&mut self, seq_id: u64) -> io::Result<()> {
        self.ensure_connected()?;
        let stream = self.stream.as_mut().unwrap();

        send_message(stream, MSG_PING_REQ, &pack_ping(seq_id))?;
        let (msg_type, payload) = recv_message(stream)?;
        if msg_type != MSG_PING_RESP {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unexpected ping response type: {}", msg_type),
            ));
        }
        let echoed = unpack_ping(&payload)?;
        if echoed != seq_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Ping sequence id mismatch",
            ));
        }
        Ok(())
    }

    pub fn infer(&mut self, req: &InferenceRequest<'_>) -> io::Result<InferenceResponse> {
        self.ensure_connected()?;

        let stream = self.stream.as_mut().unwrap();
        let payload = pack_infer_request(req);

        if let Err(err) = send_message(stream, MSG_INFER_REQ, &payload) {
            self.stream = None;
            return Err(err);
        }

        let (msg_type, resp_payload) = match recv_message(stream) {
            Ok(v) => v,
            Err(err) => {
                self.stream = None;
                return Err(err);
            }
        };

        match msg_type {
            MSG_INFER_RESP => unpack_infer_response(&resp_payload),
            MSG_ERROR_RESP => {
                let msg = unpack_error(&resp_payload)?;
                Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Worker error: {}", msg),
                ))
            }
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unexpected message type: {}", other),
            )),
        }
    }
}
