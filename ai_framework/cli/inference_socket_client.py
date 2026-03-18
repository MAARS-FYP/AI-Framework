from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np

from ai_framework.inference.protocol import (
    MSG_ERROR_RESP,
    MSG_INFER_REQ,
    MSG_INFER_RESP,
    MSG_PING_REQ,
    MSG_PING_RESP,
    MSG_SHUTDOWN_REQ,
    MSG_SHUTDOWN_RESP,
    pack_infer_request,
    pack_ping,
    recv_message,
    send_message,
    unpack_error,
    unpack_infer_response,
)


def _load_iq(path: str) -> np.ndarray:
    arr = np.load(path)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64).reshape(-1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
    raise ValueError("IQ file must be complex1D or Nx2 real")


def main():
    parser = argparse.ArgumentParser(description="Client for MAARS inference socket worker")
    parser.add_argument("--socket-path", default="/tmp/maars_infer.sock")
    parser.add_argument("--ping", action="store_true")
    parser.add_argument("--shutdown", action="store_true")
    parser.add_argument("--iq-npy")
    parser.add_argument("--power-lna-dbm", type=float)
    parser.add_argument("--power-pa-dbm", type=float)
    parser.add_argument("--sample-rate-hz", type=float, default=0.0)
    parser.add_argument("--seq-id", type=int, default=1)
    args = parser.parse_args()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(args.socket_path)

        if args.ping:
            send_message(conn, MSG_PING_REQ, pack_ping(args.seq_id))
            msg_type, payload = recv_message(conn)
            if msg_type != MSG_PING_RESP:
                raise RuntimeError("Unexpected ping response")
            print(json.dumps({"status": "ok", "seq_id": args.seq_id, "pong": True}))
            return

        if args.shutdown:
            send_message(conn, MSG_SHUTDOWN_REQ, b"")
            msg_type, _ = recv_message(conn)
            if msg_type != MSG_SHUTDOWN_RESP:
                raise RuntimeError("Unexpected shutdown response")
            print(json.dumps({"status": "ok", "shutdown": True}))
            return

        if not args.iq_npy:
            raise ValueError("--iq-npy is required for inference")
        if args.power_lna_dbm is None or args.power_pa_dbm is None:
            raise ValueError("--power-lna-dbm and --power-pa-dbm are required for inference")

        iq = _load_iq(args.iq_npy)
        req = pack_infer_request(
            seq_id=args.seq_id,
            sample_rate_hz=args.sample_rate_hz,
            power_lna_dbm=args.power_lna_dbm,
            power_pa_dbm=args.power_pa_dbm,
            iq_complex=iq,
        )
        send_message(conn, MSG_INFER_REQ, req)

        msg_type, payload = recv_message(conn)
        if msg_type == MSG_ERROR_RESP:
            print(json.dumps({"status": "error", "message": unpack_error(payload)}), file=sys.stderr)
            sys.exit(1)
        if msg_type != MSG_INFER_RESP:
            print(json.dumps({"status": "error", "message": f"Unexpected message type: {msg_type}"}), file=sys.stderr)
            sys.exit(1)

        resp = unpack_infer_response(payload)
        print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()
