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
