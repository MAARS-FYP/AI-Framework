from __future__ import annotations

import socket
import struct
from typing import Dict, Tuple

import numpy as np

MAGIC = b"MAAR"
VERSION = 1

MSG_INFER_REQ = 1
MSG_INFER_RESP = 2
MSG_PING_REQ = 3
MSG_PING_RESP = 4
MSG_SHUTDOWN_REQ = 5
MSG_SHUTDOWN_RESP = 6
MSG_ERROR_RESP = 7
MSG_INFER_SHM_REQ = 8
MSG_RFCHAIN_REQ = 10
MSG_RFCHAIN_RESP = 11

STATUS_OK = 0
STATUS_INVALID_NO_SIGNAL = 1
STATUS_BAD_REQUEST = 2
STATUS_INTERNAL_ERROR = 3

HEADER_FMT = "<4sBBHI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

INFER_REQ_META_FMT = "<QdffI"
INFER_REQ_META_SIZE = struct.calcsize(INFER_REQ_META_FMT)

INFER_RESP_FMT = "<QiBBBBffff"
INFER_RESP_SIZE = struct.calcsize(INFER_RESP_FMT)

PING_FMT = "<Q"
PING_SIZE = struct.calcsize(PING_FMT)

INFER_SHM_REQ_FMT = "<QdffII"
INFER_SHM_REQ_SIZE = struct.calcsize(INFER_SHM_REQ_FMT)

RFCHAIN_REQ_META_FMT = "<Qfffffff I"
RFCHAIN_REQ_META_SIZE = struct.calcsize(RFCHAIN_REQ_META_FMT)

RFCHAIN_RESP_META_FMT = "<QIffff"
RFCHAIN_RESP_META_SIZE = struct.calcsize(RFCHAIN_RESP_META_FMT)


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while reading message")
        data.extend(chunk)
    return bytes(data)


def recv_message(conn: socket.socket) -> Tuple[int, bytes]:
    header = _recv_exact(conn, HEADER_SIZE)
    magic, version, msg_type, _flags, payload_len = struct.unpack(HEADER_FMT, header)
    if magic != MAGIC:
        raise ValueError("Invalid protocol magic")
    if version != VERSION:
        raise ValueError(f"Unsupported protocol version: {version}")
    payload = _recv_exact(conn, payload_len) if payload_len > 0 else b""
    return msg_type, payload


def send_message(conn: socket.socket, msg_type: int, payload: bytes = b"") -> None:
    header = struct.pack(HEADER_FMT, MAGIC, VERSION, int(msg_type), 0, len(payload))
    conn.sendall(header)
    if payload:
        conn.sendall(payload)


def pack_infer_request(
    seq_id: int,
    sample_rate_hz: float,
    power_lna_dbm: float,
    power_pa_dbm: float,
    iq_complex: np.ndarray,
) -> bytes:
    iq = np.asarray(iq_complex, dtype=np.complex64).reshape(-1)
    iq_iq = np.empty(iq.size * 2, dtype=np.float32)
    iq_iq[0::2] = iq.real
    iq_iq[1::2] = iq.imag
    meta = struct.pack(
        INFER_REQ_META_FMT,
        int(seq_id),
        float(sample_rate_hz),
        float(power_lna_dbm),
        float(power_pa_dbm),
        int(iq.size),
    )
    return meta + iq_iq.tobytes(order="C")


def unpack_infer_request(payload: bytes) -> Dict[str, object]:
    if len(payload) < INFER_REQ_META_SIZE:
        raise ValueError("Infer request payload too small")

    seq_id, sample_rate_hz, power_lna_dbm, power_pa_dbm, n_samples = struct.unpack(
        INFER_REQ_META_FMT, payload[:INFER_REQ_META_SIZE]
    )
    expected_bytes = n_samples * 2 * 4
    raw_iq = payload[INFER_REQ_META_SIZE:]
    if len(raw_iq) != expected_bytes:
        raise ValueError(
            f"Infer request IQ byte size mismatch: got {len(raw_iq)}, expected {expected_bytes}"
        )

    iq_pairs = np.frombuffer(raw_iq, dtype=np.float32)
    iq_complex = (iq_pairs[0::2] + 1j * iq_pairs[1::2]).astype(np.complex64)

    return {
        "seq_id": int(seq_id),
        "sample_rate_hz": float(sample_rate_hz),
        "power_lna_dbm": float(power_lna_dbm),
        "power_pa_dbm": float(power_pa_dbm),
        "iq_complex": iq_complex,
    }


def pack_infer_shm_request(
    seq_id: int,
    sample_rate_hz: float,
    power_lna_dbm: float,
    power_pa_dbm: float,
    slot_index: int,
    n_samples: int,
) -> bytes:
    return struct.pack(
        INFER_SHM_REQ_FMT,
        int(seq_id),
        float(sample_rate_hz),
        float(power_lna_dbm),
        float(power_pa_dbm),
        int(slot_index),
        int(n_samples),
    )


def unpack_infer_shm_request(payload: bytes) -> Dict[str, object]:
    if len(payload) != INFER_SHM_REQ_SIZE:
        raise ValueError(
            f"Infer SHM request payload size mismatch: got {len(payload)}, expected {INFER_SHM_REQ_SIZE}"
        )
    seq_id, sample_rate_hz, power_lna_dbm, power_pa_dbm, slot_index, n_samples = struct.unpack(
        INFER_SHM_REQ_FMT, payload
    )
    return {
        "seq_id": int(seq_id),
        "sample_rate_hz": float(sample_rate_hz),
        "power_lna_dbm": float(power_lna_dbm),
        "power_pa_dbm": float(power_pa_dbm),
        "slot_index": int(slot_index),
        "n_samples": int(n_samples),
    }


def pack_infer_response(
    seq_id: int,
    status_code: int,
    lna_class: int,
    filter_class: int,
    center_class: int,
    mixer_dbm: float,
    ifamp_db: float,
    evm_value: float,
    processing_time_ms: float,
) -> bytes:
    return struct.pack(
        INFER_RESP_FMT,
        int(seq_id),
        int(status_code),
        int(lna_class),
        int(filter_class),
        int(center_class),
        0,
        float(mixer_dbm),
        float(ifamp_db),
        float(evm_value),
        float(processing_time_ms),
    )


def unpack_infer_response(payload: bytes) -> Dict[str, object]:
    if len(payload) != INFER_RESP_SIZE:
        raise ValueError(f"Infer response payload size mismatch: got {len(payload)}, expected {INFER_RESP_SIZE}")

    seq_id, status_code, lna_class, filter_class, center_class, _reserved, mixer_dbm, ifamp_db, evm_value, processing_time_ms = struct.unpack(
        INFER_RESP_FMT, payload
    )
    return {
        "seq_id": int(seq_id),
        "status_code": int(status_code),
        "lna_class": int(lna_class),
        "filter_class": int(filter_class),
        "center_class": int(center_class),
        "mixer_dbm": float(mixer_dbm),
        "ifamp_db": float(ifamp_db),
        "evm_value": float(evm_value),
        "processing_time_ms": float(processing_time_ms),
    }


def pack_ping(seq_id: int) -> bytes:
    return struct.pack(PING_FMT, int(seq_id))


def unpack_ping(payload: bytes) -> int:
    if len(payload) != PING_SIZE:
        raise ValueError("Ping payload size mismatch")
    (seq_id,) = struct.unpack(PING_FMT, payload)
    return int(seq_id)


def pack_error(message: str) -> bytes:
    data = message.encode("utf-8", errors="replace")
    size = struct.pack("<I", len(data))
    return size + data


def unpack_error(payload: bytes) -> str:
    if len(payload) < 4:
        return "Unknown error"
    (n,) = struct.unpack("<I", payload[:4])
    data = payload[4:4 + n]
    return data.decode("utf-8", errors="replace")


def pack_rfchain_request(
    seq_id: int,
    input_power_dbm: float,
    bandwidth_hz: float,
    center_freq_hz: float,
    lo_freq_hz: float,
    lna_voltage: float,
    lo_power_dbm: float,
    pa_gain_db: float,
    num_symbols: int = 30,
) -> bytes:
    """Pack RF chain request into binary format."""
    meta = struct.pack(
        RFCHAIN_REQ_META_FMT,
        int(seq_id),
        float(input_power_dbm),
        float(bandwidth_hz),
        float(center_freq_hz),
        float(lo_freq_hz),
        float(lna_voltage),
        float(lo_power_dbm),
        float(pa_gain_db),
        int(num_symbols),
    )
    return meta


def unpack_rfchain_request(payload: bytes) -> Dict[str, object]:
    """Unpack RF chain request from binary format."""
    if len(payload) != RFCHAIN_REQ_META_SIZE:
        raise ValueError(
            f"RF chain request payload size mismatch: got {len(payload)}, expected {RFCHAIN_REQ_META_SIZE}"
        )
    seq_id, input_power_dbm, bandwidth_hz, center_freq_hz, lo_freq_hz, lna_voltage, lo_power_dbm, pa_gain_db, num_symbols = struct.unpack(
        RFCHAIN_REQ_META_FMT, payload
    )
    return {
        "seq_id": int(seq_id),
        "input_power_dbm": float(input_power_dbm),
        "bandwidth_hz": float(bandwidth_hz),
        "center_freq_hz": float(center_freq_hz),
        "lo_freq_hz": float(lo_freq_hz),
        "lna_voltage": float(lna_voltage),
        "lo_power_dbm": float(lo_power_dbm),
        "pa_gain_db": float(pa_gain_db),
        "num_symbols": int(num_symbols),
    }


def pack_rfchain_response(
    seq_id: int,
    status: str,
    i_samples: np.ndarray,
    q_samples: np.ndarray,
    evm_percent: float,
    power_post_lna_dbm: float,
    power_post_pa_dbm: float,
    processing_time_ms: float,
) -> bytes:
    """Pack RF chain response into binary format."""
    # Encode status string
    status_bytes = status.encode("utf-8", errors="replace")
    
    # Convert samples to float32
    i_data = np.asarray(i_samples, dtype=np.float32).tobytes(order="C")
    q_data = np.asarray(q_samples, dtype=np.float32).tobytes(order="C")
    
    # Pack header with lengths
    meta = struct.pack(
        RFCHAIN_RESP_META_FMT,
        int(seq_id),
        len(status_bytes),
        float(evm_percent),
        float(power_post_lna_dbm),
        float(power_post_pa_dbm),
        float(processing_time_ms),
    )
    
    # Pack I/Q sample counts (number of samples, not bytes)
    n_samples = len(i_samples)
    sample_counts = struct.pack("<II", int(n_samples), int(n_samples))
    
    # Concatenate: meta + status_len + status + i_len + i_data + q_len + q_data
    return meta + status_bytes + sample_counts + i_data + q_data


def unpack_rfchain_response(payload: bytes) -> Dict[str, object]:
    """Unpack RF chain response from binary format."""
    if len(payload) < RFCHAIN_RESP_META_SIZE:
        raise ValueError("RF chain response payload too small for header")
    
    seq_id, status_len, evm_percent, power_post_lna_dbm, power_post_pa_dbm, processing_time_ms = struct.unpack(
        RFCHAIN_RESP_META_FMT, payload[:RFCHAIN_RESP_META_SIZE]
    )
    
    offset = RFCHAIN_RESP_META_SIZE
    
    # Extract status string
    status_bytes = payload[offset:offset + status_len]
    status = status_bytes.decode("utf-8", errors="replace")
    offset += status_len
    
    # Extract sample counts
    if len(payload) < offset + 8:
        raise ValueError("RF chain response missing sample counts")
    n_i, n_q = struct.unpack("<II", payload[offset:offset + 8])
    offset += 8
    
    # Extract I/Q samples
    i_bytes = n_i * 4
    q_bytes = n_q * 4
    
    if len(payload) < offset + i_bytes + q_bytes:
        raise ValueError("RF chain response missing I/Q sample data")
    
    i_data = np.frombuffer(payload[offset:offset + i_bytes], dtype=np.float32)
    offset += i_bytes
    q_data = np.frombuffer(payload[offset:offset + q_bytes], dtype=np.float32)
    
    return {
        "seq_id": int(seq_id),
        "status": str(status),
        "i_samples": i_data,
        "q_samples": q_data,
        "evm_percent": float(evm_percent),
        "power_post_lna_dbm": float(power_post_lna_dbm),
        "power_post_pa_dbm": float(power_post_pa_dbm),
        "processing_time_ms": float(processing_time_ms),
    }
