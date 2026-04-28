use std::io;
use std::os::unix::net::UnixStream;

use crate::protocol::{
    MSG_RFCHAIN_REQ, MSG_RFCHAIN_RESP, MSG_ERROR_RESP, RFChainRequest, RFChainResponse,
    pack_rfchain_request, unpack_rfchain_response, unpack_error, send_message, recv_message,
};

pub struct RFChainSocketClient {
    stream: UnixStream,
    seq_id: u64,
}

impl RFChainSocketClient {
    pub fn connect(socket_path: &str) -> io::Result<Self> {
        let stream = UnixStream::connect(socket_path)?;
        Ok(RFChainSocketClient {
            stream,
            seq_id: 0,
        })
    }

    fn next_seq_id(&mut self) -> u64 {
        self.seq_id += 1;
        self.seq_id
    }

    pub fn process_signal(
        &mut self,
        power_pre_lna_dbm: f32,
        bandwidth_hz: f32,
        center_freq_hz: f32,
        lna_voltage: f32,
        lo_power_dbm: f32,
        pa_gain_db: f32,
    ) -> io::Result<RFChainResponse> {
        let seq_id = self.next_seq_id();

        let req = RFChainRequest {
            seq_id,
            power_pre_lna_dbm,
            bandwidth_hz,
            center_freq_hz,
            lna_voltage,
            lo_power_dbm,
            pa_gain_db,
            num_symbols: 30,
        };

        let payload = pack_rfchain_request(&req);
        send_message(&mut self.stream, MSG_RFCHAIN_REQ, &payload)?;

        let (msg_type, resp_payload) = recv_message(&mut self.stream)?;

        match msg_type {
            MSG_RFCHAIN_RESP => {
                unpack_rfchain_response(&resp_payload)
            }
            MSG_ERROR_RESP => {
                let error_msg = unpack_error(&resp_payload)?;
                Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("RF chain error: {}", error_msg),
                ))
            }
            _ => {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unexpected response type: {}", msg_type),
                ))
            }
        }
    }
}
