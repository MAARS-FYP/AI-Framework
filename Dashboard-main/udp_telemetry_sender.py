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


class SenderState:
    def __init__(self, seq_start: int):
        self.seq_id = seq_start
        self.status_code = 0
        self.status_text = "ok"
        self.lna_class = 0
        self.filter_class = 1
        self.center_class = 1
        self.lna_voltage_v = 3.0
        self.selected_filter_mhz = 10.0
        self.lo_center_mhz = 2420.0
        self.lo_power_dbm = -15.0
        self.ifamp_db = 0.0
        self.power_lna_raw = -35.0
        self.power_pa_raw = -24.0
        self.evm_value = 0.0
        self.processing_time_ms = 0.0
        self.source = "init"

    def update(self, snapshot: Dict[str, Any]):
        if not snapshot:
            return

        if "seq_id" in snapshot:
            self.seq_id = int(snapshot["seq_id"])
        
        if "status" in snapshot or "status_text" in snapshot:
            self.status_text = str(snapshot.get("status", snapshot.get("status_text")))
        
        if "status_code" in snapshot:
            self.status_code = int(snapshot["status_code"])
        
        if "lna_class" in snapshot:
            self.lna_class = int(snapshot["lna_class"])
        if "filter_class" in snapshot:
            self.filter_class = int(snapshot["filter_class"])
        if "center_class" in snapshot:
            self.center_class = int(snapshot["center_class"])

        self.lna_voltage_v = float(snapshot.get("lna_voltage_v", LNA_VOLTAGE_BY_CLASS.get(self.lna_class, self.lna_voltage_v)))
        self.selected_filter_mhz = float(snapshot.get("selected_filter_mhz", FILTER_MHZ_BY_CLASS.get(self.filter_class, self.selected_filter_mhz)))
        
        # Priority: explicit lo_center_mhz -> valon_frequency_mhz -> class-based
        if "lo_center_mhz" in snapshot:
            self.lo_center_mhz = float(snapshot["lo_center_mhz"])
        elif "valon_frequency_mhz" in snapshot:
            self.lo_center_mhz = float(snapshot["valon_frequency_mhz"])
        else:
            self.lo_center_mhz = float(CENTER_MHZ_BY_CLASS.get(self.center_class, self.lo_center_mhz))

        self.lo_power_dbm = float(snapshot.get("lo_power_dbm", snapshot.get("valon_power_dbm", snapshot.get("mixer_dbm", self.lo_power_dbm))))
        self.ifamp_db = float(snapshot.get("ifamp_db", snapshot.get("ifamp_value", self.ifamp_db)))
        
        self.power_lna_raw = float(
            snapshot.get(
                "power_lna_raw",
                snapshot.get("power_lna_dbm", snapshot.get("power_pre_lna_dbm", snapshot.get("power_pre_lna", self.power_lna_raw))),
            )
        )
        self.power_pa_raw = float(
            snapshot.get(
                "power_pa_raw",
                snapshot.get("power_pa_dbm", snapshot.get("power_post_pa_dbm", snapshot.get("power_post_pa", self.power_pa_raw))),
            )
        )
        
        self.evm_value = float(snapshot.get("evm_value", snapshot.get("evm_percent", snapshot.get("evm", self.evm_value))))
        self.processing_time_ms = float(snapshot.get("processing_time_ms", self.processing_time_ms))
        self.source = str(snapshot.get("source", self.source))

    def to_payload(self) -> Dict[str, Any]:
        return {
            "seq_id": self.seq_id,
            "status_code": self.status_code,
            "status_text": self.status_text,
            "lna_class": self.lna_class,
            "lna_voltage_v": round(self.lna_voltage_v, 2),
            "filter_class": self.filter_class,
            "selected_filter_mhz": round(self.selected_filter_mhz, 2),
            "selected_filter_label": f"{self.selected_filter_mhz:.0f} MHz" if self.selected_filter_mhz >= 1 else f"{self.selected_filter_mhz:.2f} MHz",
            "center_class": self.center_class,
            "lo_center_mhz": round(self.lo_center_mhz, 3),
            "lo_power_dbm": round(self.lo_power_dbm, 2),
            "mixer_dbm": round(self.lo_power_dbm, 2),
            "ifamp_db": round(self.ifamp_db, 2),
            "power_lna_raw": round(self.power_lna_raw, 2),
            "power_pa_raw": round(self.power_pa_raw, 2),
            "power_lna_dbm": round(self.power_lna_raw, 2),
            "power_pre_lna_dbm": round(self.power_lna_raw, 2),
            "power_pa_dbm": round(self.power_pa_raw, 2),
            "power_post_pa_dbm": round(self.power_pa_raw, 2),
            "evm_value": round(self.evm_value, 2),
            "evm_percent": round(self.evm_value, 2),
            "processing_time_ms": round(self.processing_time_ms, 3),
            "source": self.source,
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
    sent = 0
    state = SenderState(start_seq)
    last_sent_payload = None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Sending UDP telemetry to {host}:{port} at {hz:.2f} Hz")
    print(f"Telemetry source: {source_file}")
    
    next_send = time.perf_counter()

    try:
        while count is None or sent < count:
            now = time.perf_counter()
            if now < next_send:
                time.sleep(next_send - now)

            snapshot = parse_snapshot_file(source_file)
            if snapshot:
                state.update(snapshot)

            payload = state.to_payload()
            
            # Send every cycle for high-resolution updates
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            sock.sendto(encoded, (host, port))
            
            if verbose or (sent % 30 == 0):
                # Print abbreviated summary for standard logging
                print(f"[telemetry seq={payload['seq_id']}] LNA={payload['power_lna_raw']} PA={payload['power_pa_raw']} EVM={payload['evm_value']} BW={payload['selected_filter_mhz']} RF={payload['lo_center_mhz']}")
                # Detailed payload log for deep debugging
                if verbose:
                    print(f"FULL PAYLOAD: {encoded.decode('utf-8')}")

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
