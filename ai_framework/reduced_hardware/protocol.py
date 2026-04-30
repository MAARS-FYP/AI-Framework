from __future__ import annotations

import json
from typing import Any, Dict, Optional


def dumps_message(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def loads_message(data: bytes) -> Dict[str, Any]:
    message = json.loads(data.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("Message must decode to a JSON object")
    return message


def make_response(
    seq_id: int,
    center_class: int,
    bandwidth_class: int,
    center_confidence: float,
    bandwidth_confidence: float,
    status: str = "ok",
    capture_source: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": "infer_response",
        "status": status,
        "seq_id": int(seq_id),
        "center_class": int(center_class),
        "bandwidth_class": int(bandwidth_class),
        "center_frequency_mhz": int(2405 + 15 * int(center_class)),
        "bandwidth_mhz": int((1, 10, 20)[int(bandwidth_class)]),
        "center_confidence": float(center_confidence),
        "bandwidth_confidence": float(bandwidth_confidence),
    }
    if capture_source is not None:
        payload["capture_source"] = capture_source
    return payload


def make_error(seq_id: Optional[int], message: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"type": "error", "status": "error", "message": message}
    if seq_id is not None:
        payload["seq_id"] = int(seq_id)
    return payload
