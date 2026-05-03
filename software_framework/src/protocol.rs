use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;

pub const MAGIC: [u8; 4] = *b"MAAR";
pub const VERSION: u8 = 1;

pub const MSG_INFER_REQ: u8 = 1;
pub const MSG_INFER_RESP: u8 = 2;
pub const MSG_PING_REQ: u8 = 3;
pub const MSG_PING_RESP: u8 = 4;
pub const MSG_ERROR_RESP: u8 = 7;
pub const MSG_INFER_SHM_REQ: u8 = 8;
pub const MSG_RFCHAIN_REQ: u8 = 10;
pub const MSG_RFCHAIN_RESP: u8 = 11;

const HEADER_SIZE: usize = 12;
const INFER_RESP_SIZE: usize = 32;

#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub seq_id: u64,
    pub status_code: i32,
    pub lna_class: u8,
    pub filter_class: u8,
    pub center_class: u8,
    pub mixer_dbm: f32,
    pub ifamp_db: f32,
    pub evm_value: f32,
    pub processing_time_ms: f32,
}

#[derive(Debug, Clone)]
pub struct InferenceRequest<'a> {
    pub seq_id: u64,
    pub sample_rate_hz: f64,
    pub power_lna_dbm: f32,
    pub power_pa_dbm: f32,
    pub iq_iq_pairs: &'a [(f32, f32)],
}

#[derive(Debug, Clone)]
pub struct InferenceShmRequest {
    pub seq_id: u64,
    pub sample_rate_hz: f64,
    pub power_lna_dbm: f32,
    pub power_pa_dbm: f32,
    pub slot_index: u32,
    pub n_samples: u32,
}

#[derive(Debug, Clone)]
pub struct RFChainRequest {
    pub seq_id: u64,
    pub input_power_dbm: f32,
    pub bandwidth_hz: f32,
    pub center_freq_hz: f32,
    pub lo_freq_hz: f32,
    pub lna_voltage: f32,
    pub lo_power_dbm: f32,
    pub pa_gain_db: f32,
    pub num_symbols: u32,
}

#[derive(Debug, Clone)]
pub struct RFChainResponse {
    pub seq_id: u64,
    pub status: String,
    pub i_samples: Vec<f32>,
    pub q_samples: Vec<f32>,
    pub evm_percent: f32,
    pub power_post_lna_dbm: f32,
    pub power_post_pa_dbm: f32,
    pub processing_time_ms: f32,
}

fn read_exact_or_eof(stream: &mut UnixStream, size: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; size];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

pub fn send_message(stream: &mut UnixStream, msg_type: u8, payload: &[u8]) -> io::Result<()> {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC);
    header[4] = VERSION;
    header[5] = msg_type;
    header[6..8].copy_from_slice(&0u16.to_le_bytes());
    header[8..12].copy_from_slice(&(payload.len() as u32).to_le_bytes());

    stream.write_all(&header)?;
    if !payload.is_empty() {
        stream.write_all(payload)?;
    }
    Ok(())
}

pub fn recv_message(stream: &mut UnixStream) -> io::Result<(u8, Vec<u8>)> {
    let header = read_exact_or_eof(stream, HEADER_SIZE)?;
    if header[0..4] != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid protocol magic",
        ));
    }
    if header[4] != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported protocol version: {}", header[4]),
        ));
    }

    let msg_type = header[5];
    let payload_len = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let payload = if payload_len > 0 {
        read_exact_or_eof(stream, payload_len)?
    } else {
        Vec::new()
    };
    Ok((msg_type, payload))
}

pub fn pack_infer_request(req: &InferenceRequest<'_>) -> Vec<u8> {
    let mut payload = Vec::with_capacity(28 + req.iq_iq_pairs.len() * 8);

    payload.extend_from_slice(&req.seq_id.to_le_bytes());
    payload.extend_from_slice(&req.sample_rate_hz.to_le_bytes());
    payload.extend_from_slice(&req.power_lna_dbm.to_le_bytes());
    payload.extend_from_slice(&req.power_pa_dbm.to_le_bytes());
    payload.extend_from_slice(&(req.iq_iq_pairs.len() as u32).to_le_bytes());

    for (i, q) in req.iq_iq_pairs.iter() {
        payload.extend_from_slice(&i.to_le_bytes());
        payload.extend_from_slice(&q.to_le_bytes());
    }

    payload
}

pub fn unpack_infer_response(payload: &[u8]) -> io::Result<InferenceResponse> {
    if payload.len() != INFER_RESP_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Infer response payload size mismatch: got {}, expected {}",
                payload.len(),
                INFER_RESP_SIZE
            ),
        ));
    }

    let seq_id = u64::from_le_bytes(payload[0..8].try_into().unwrap());
    let status_code = i32::from_le_bytes(payload[8..12].try_into().unwrap());
    let lna_class = payload[12];
    let filter_class = payload[13];
    let center_class = payload[14];
    let mixer_dbm = f32::from_le_bytes(payload[16..20].try_into().unwrap());
    let ifamp_db = f32::from_le_bytes(payload[20..24].try_into().unwrap());
    let evm_value = f32::from_le_bytes(payload[24..28].try_into().unwrap());
    let processing_time_ms = f32::from_le_bytes(payload[28..32].try_into().unwrap());

    Ok(InferenceResponse {
        seq_id,
        status_code,
        lna_class,
        filter_class,
        center_class,
        mixer_dbm,
        ifamp_db,
        evm_value,
        processing_time_ms,
    })
}

pub fn pack_infer_shm_request(req: &InferenceShmRequest) -> Vec<u8> {
    let mut payload = Vec::with_capacity(32);
    payload.extend_from_slice(&req.seq_id.to_le_bytes());
    payload.extend_from_slice(&req.sample_rate_hz.to_le_bytes());
    payload.extend_from_slice(&req.power_lna_dbm.to_le_bytes());
    payload.extend_from_slice(&req.power_pa_dbm.to_le_bytes());
    payload.extend_from_slice(&req.slot_index.to_le_bytes());
    payload.extend_from_slice(&req.n_samples.to_le_bytes());
    payload
}

pub fn unpack_error(payload: &[u8]) -> io::Result<String> {
    if payload.len() < 4 {
        return Ok("Unknown error".to_string());
    }
    let n = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    if payload.len() < 4 + n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Malformed error payload",
        ));
    }
    let text = String::from_utf8_lossy(&payload[4..4 + n]).to_string();
    Ok(text)
}

pub fn pack_ping(seq_id: u64) -> Vec<u8> {
    seq_id.to_le_bytes().to_vec()
}

pub fn unpack_ping(payload: &[u8]) -> io::Result<u64> {
    if payload.len() != 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Ping payload size mismatch",
        ));
    }
    Ok(u64::from_le_bytes(payload.try_into().unwrap()))
}

pub fn pack_rfchain_request(req: &RFChainRequest) -> Vec<u8> {
    let mut payload = Vec::with_capacity(44);
    payload.extend_from_slice(&req.seq_id.to_le_bytes());
    payload.extend_from_slice(&req.input_power_dbm.to_le_bytes());
    payload.extend_from_slice(&req.bandwidth_hz.to_le_bytes());
    payload.extend_from_slice(&req.center_freq_hz.to_le_bytes());
    payload.extend_from_slice(&req.lo_freq_hz.to_le_bytes());
    payload.extend_from_slice(&req.lna_voltage.to_le_bytes());
    payload.extend_from_slice(&req.lo_power_dbm.to_le_bytes());
    payload.extend_from_slice(&req.pa_gain_db.to_le_bytes());
    payload.extend_from_slice(&req.num_symbols.to_le_bytes());
    payload
}

pub fn unpack_rfchain_response(payload: &[u8]) -> io::Result<RFChainResponse> {
    if payload.len() < 28 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "RF chain response payload too small",
        ));
    }

    let seq_id = u64::from_le_bytes(payload[0..8].try_into().unwrap());
    let status_len = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
    let evm_percent = f32::from_le_bytes(payload[12..16].try_into().unwrap());
    let power_post_lna_dbm = f32::from_le_bytes(payload[16..20].try_into().unwrap());
    let power_post_pa_dbm = f32::from_le_bytes(payload[20..24].try_into().unwrap());
    let processing_time_ms = f32::from_le_bytes(payload[24..28].try_into().unwrap());

    let mut offset = 28;

    // Extract status string
    if offset + status_len > payload.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Status string exceeds payload",
        ));
    }
    let status = String::from_utf8_lossy(&payload[offset..offset + status_len]).to_string();
    offset += status_len;

    // Extract sample counts
    if offset + 8 > payload.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Missing sample counts",
        ));
    }
    let n_i = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
    let n_q = u32::from_le_bytes(payload[offset + 4..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    // Extract I samples
    let i_bytes = n_i * 4;
    if offset + i_bytes > payload.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Missing I sample data",
        ));
    }
    let mut i_samples = vec![0.0f32; n_i];
    for i in 0..n_i {
        i_samples[i] = f32::from_le_bytes(
            payload[offset + i * 4..offset + (i + 1) * 4]
                .try_into()
                .unwrap(),
        );
    }
    offset += i_bytes;

    // Extract Q samples
    let q_bytes = n_q * 4;
    if offset + q_bytes > payload.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Missing Q sample data",
        ));
    }
    let mut q_samples = vec![0.0f32; n_q];
    for i in 0..n_q {
        q_samples[i] = f32::from_le_bytes(
            payload[offset + i * 4..offset + (i + 1) * 4]
                .try_into()
                .unwrap(),
        );
    }

    Ok(RFChainResponse {
        seq_id,
        status,
        i_samples,
        q_samples,
        evm_percent,
        power_post_lna_dbm,
        power_post_pa_dbm,
        processing_time_ms,
    })
}
