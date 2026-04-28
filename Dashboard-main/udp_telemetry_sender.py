#!/usr/bin/env python3
"""Send synthetic RF telemetry as UDP JSON packets.

This is useful for testing an upstream bridge that converts UDP to WebSocket.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import time
from typing import Dict, Any


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_payload(seq_id: int, t_seconds: float) -> Dict[str, Any]:
    mixer_dbm = clamp(-15.0 + 8.0 * math.sin(t_seconds * 1.2) + random.uniform(-0.6, 0.6), -30.0, 15.0)
    ifamp_db = clamp(14.0 + 9.0 * math.cos(t_seconds * 0.85) + random.uniform(-0.7, 0.7), -10.0, 30.0)
    evm_value = clamp(4.0 + 1.2 * math.sin(t_seconds * 2.0) + random.uniform(-0.25, 0.25), 0.5, 12.0)

    return {
        "seq_id": seq_id,
        "status_code": 0,
        "lna_class": 1 if math.sin(t_seconds * 0.25) > 0 else 0,
        "filter_class": int(((math.sin(t_seconds * 0.2) + 1.0) * 1.5)) % 3,
        "center_class": int(((math.cos(t_seconds * 0.3) + 1.0) * 1.5)) % 3,
        "mixer_dbm": round(mixer_dbm, 2),
        "ifamp_db": round(ifamp_db, 2),
        "power_lna_raw": round(clamp(-35.0 + 2.0 * math.sin(t_seconds * 1.6), -60.0, -10.0), 2),
        "power_pa_raw": round(clamp(-24.0 + 2.4 * math.cos(t_seconds * 1.4), -45.0, 5.0), 2),
        "evm_value": round(evm_value, 2),
        "processing_time_ms": round(clamp(0.55 + random.random() * 0.5, 0.3, 2.0), 3),
    }


def run_sender(host: str, port: int, hz: float, start_seq: int, count: int | None, verbose: bool) -> None:
    interval = 1.0 / hz
    seq = start_seq
    sent = 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Sending UDP telemetry to {host}:{port} at {hz:.2f} Hz")
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

            payload = build_payload(seq, time.time())
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
        "--count",
        type=int,
        default=None,
        help="Number of packets to send (default: unlimited)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each packet JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sender(
        host=args.host,
        port=args.port,
        hz=args.hz,
        start_seq=args.start_seq,
        count=args.count,
        verbose=args.verbose,
    )
