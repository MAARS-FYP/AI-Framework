#!/usr/bin/env python3
"""Send AI/RF telemetry as UDP JSON packets.

This reads the latest inference snapshot written by the Rust digital-twin
or hardware inference loop and forwards it over UDP so the WebSocket bridge
and browser dashboard can update in real time.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional


LNA_VOLTAGE_BY_CLASS = {0: 3.0, 1: 5.0}
FILTER_MHZ_BY_CLASS = {0: 1.0, 1: 10.0, 2: 20.0}
CENTER_MHZ_BY_CLASS = {0: 2405.0, 1: 2420.0, 2: 2435.0}


def parse_numeric(value: str) -> Optional[float]:
    try:
        return float(value.strip())
    except Exception:
        return None


def parse_snapshot_file(path: Path) -> Dict[str, Any]:
    """Parse a Rust inference snapshot file into a dictionary."""
    if not path.exists():
        return {}

    result: Dict[str, Any] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return result

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue

        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if key in {
            "seq_id",
            "status_code",
            "lna_class",
            "filter_class",
            "center_class",
        }:
            numeric = parse_numeric(value)
            if numeric is not None:
                result[key] = int(numeric)
            else:
                result[key] = value
            continue

        numeric = parse_numeric(value)
        result[key] = numeric if numeric is not None else value

    return result


def derive_payload(snapshot: Dict[str, Any], seq_fallback: int) -> Dict[str, Any]:
    seq_id = int(snapshot.get("seq_id", seq_fallback) or seq_fallback)

    status_value = snapshot.get("status", snapshot.get("status_text", "ok"))
    status_code = int(snapshot.get("status_code", 0) or 0)
    if isinstance(status_value, str) and status_value.lower() not in {"ok", "0", "live"}:
        status_code = 1 if status_code == 0 else status_code

    lna_class = int(snapshot.get("lna_class", 0) or 0)
    filter_class = int(snapshot.get("filter_class", 1) or 1)
    center_class = int(snapshot.get("center_class", 1) or 1)

    lna_voltage_v = float(
        snapshot.get("lna_voltage_v", LNA_VOLTAGE_BY_CLASS.get(lna_class, 3.0))
    )
    selected_filter_mhz = float(
        snapshot.get("selected_filter_mhz", FILTER_MHZ_BY_CLASS.get(filter_class, 10.0))
    )
    lo_center_mhz = float(
        snapshot.get("lo_center_mhz", snapshot.get("valon_frequency_mhz", CENTER_MHZ_BY_CLASS.get(center_class, 2420.0)))
    )
    lo_power_dbm = float(
        snapshot.get("lo_power_dbm", snapshot.get("valon_power_dbm", snapshot.get("mixer_dbm", -15.0)))
    )
    if_amp_gain_db = float(
        snapshot.get("if_amp_gain_db", snapshot.get("ifamp_db", snapshot.get("ifamp_value", 0.0)))
    )

    power_lna_dbm = float(
        snapshot.get("power_lna_dbm", snapshot.get("power_pre_lna", snapshot.get("power_lna_raw", -35.0)))
    )
    power_pa_dbm = float(
        snapshot.get("power_pa_dbm", snapshot.get("power_post_pa", snapshot.get("power_pa_raw", -24.0)))
    )
    evm_value = float(snapshot.get("evm_value", snapshot.get("evm", 0.0)) or 0.0)
    processing_time_ms = float(snapshot.get("processing_time_ms", 0.0) or 0.0)

    return {
        "seq_id": seq_id,
        "status_code": status_code,
        "status_text": str(status_value),
        "lna_class": lna_class,
        "lna_voltage_v": round(lna_voltage_v, 2),
        "filter_class": filter_class,
        "selected_filter_mhz": round(selected_filter_mhz, 2),
        "selected_filter_label": f"{selected_filter_mhz:.0f} MHz" if selected_filter_mhz >= 1 else f"{selected_filter_mhz:.2f} MHz",
        "center_class": center_class,
        "lo_center_mhz": round(lo_center_mhz, 3),
        "lo_power_dbm": round(lo_power_dbm, 2),
        "mixer_dbm": round(lo_power_dbm, 2),
        "ifamp_db": round(if_amp_gain_db, 2),
        "power_lna_raw": round(power_lna_dbm, 2),
        "power_pa_raw": round(power_pa_dbm, 2),
        "power_lna_dbm": round(power_lna_dbm, 2),
        "power_pa_dbm": round(power_pa_dbm, 2),
        "evm_value": round(evm_value, 2),
        "processing_time_ms": round(processing_time_ms, 3),
        "source": snapshot.get("source", "inference_snapshot"),
    }


def run_sender(
    host: str,
    port: int,
    hz: float,
    start_seq: int,
    count: int | None,
    verbose: bool,
    source_file: Path,
) -> None:
    interval = 1.0 / hz
    seq = start_seq
    sent = 0
    last_snapshot: Dict[str, Any] = {}

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Sending UDP telemetry to {host}:{port} at {hz:.2f} Hz")
    print(f"Telemetry source: {source_file}")
    if count is None:
        print("Packet count: infinite (Ctrl+C to stop)")
    else:
        print(f"Packet count: {count}")

    next_send = time.perf_counter()

    try:
        while count is None or sent < count:
            now = time.perf_counter()
            if now < next_send:
                time.sleep(next_send - now)

            last_snapshot = parse_snapshot_file(source_file)

            if not last_snapshot:
                time.sleep(0.05)
                continue

            payload = derive_payload(last_snapshot, seq)
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            sock.sendto(encoded, (host, port))

            if verbose:
                print(encoded.decode("utf-8"))

            seq += 1
            sent += 1
            next_send += interval
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UDP telemetry packet generator")
    parser.add_argument("--host", default="127.0.0.1", help="UDP destination host")
    parser.add_argument("--port", type=int, default=9000, help="UDP destination port")
    parser.add_argument("--hz", type=float, default=15.0, help="Packet rate in Hz")
    parser.add_argument("--start-seq", type=int, default=1, help="Starting sequence ID")
    parser.add_argument(
        "--source-file",
        default="../inference_results.txt",
        help="Inference snapshot text file to stream",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of packets to send (default: unlimited)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each packet JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source_path = Path(args.source_file)
    if not source_path.is_absolute():
        source_path = (Path(__file__).parent / source_path).resolve()
    run_sender(
        host=args.host,
        port=args.port,
        hz=args.hz,
        start_seq=args.start_seq,
        count=args.count,
        verbose=args.verbose,
        source_file=source_path,
    )
