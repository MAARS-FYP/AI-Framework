"""Protocol + validation helpers for Valon worker/client IPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


# Safety ranges copied from Defaults.py
FREQ_MIN_MHZ = 10.0
FREQ_MAX_MHZ = 19000.0
RFLEVEL_MIN_DBM = -50.0
RFLEVEL_MAX_DBM = 20.0


@dataclass(frozen=True)
class RangeSpec:
    minimum: float
    maximum: float
    unit: str


FREQ_RANGE = RangeSpec(FREQ_MIN_MHZ, FREQ_MAX_MHZ, "MHz")
RFLEVEL_RANGE = RangeSpec(RFLEVEL_MIN_DBM, RFLEVEL_MAX_DBM, "dBm")


def _require_number(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric")


def _check_range(name: str, value: float, spec: RangeSpec) -> float:
    if value < spec.minimum or value > spec.maximum:
        raise ValueError(
            f"{name} out of range [{spec.minimum}, {spec.maximum}] {spec.unit}"
        )
    return value


def validate_set_freq_mhz(value: Any) -> float:
    return _check_range("frequency", _require_number("frequency", value), FREQ_RANGE)


def validate_set_rflevel_dbm(value: Any) -> float:
    return _check_range("rf level", _require_number("rf level", value), RFLEVEL_RANGE)


def build_set_freq_command(mhz: float) -> str:
    # Matches existing GUI command shape: "Freq <value> MHz"
    return f"Freq {mhz} MHz"


def build_set_rflevel_command(dbm: float) -> str:
    # Existing GUI sets "PWR <value> " (trailing units field is blank).
    # Command parser on device accepts "PWR <value>".
    return f"PWR {dbm}"


def ok_response(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": req_id, "ok": True, "result": result}


def error_response(req_id: Any, code: str, message: str, retryable: bool) -> Dict[str, Any]:
    return {
        "id": req_id,
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }
