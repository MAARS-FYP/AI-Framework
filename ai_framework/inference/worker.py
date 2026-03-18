from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
from typing import Optional

from ai_framework.inference.config import InferenceConfig
from ai_framework.inference.engine import RFInferenceEngine
from ai_framework.inference.protocol import (
    MSG_ERROR_RESP,
    MSG_INFER_REQ,
    MSG_INFER_RESP,
    MSG_PING_REQ,
    MSG_PING_RESP,
    MSG_SHUTDOWN_REQ,
    MSG_SHUTDOWN_RESP,
    STATUS_BAD_REQUEST,
    STATUS_INTERNAL_ERROR,
    STATUS_INVALID_NO_SIGNAL,
    STATUS_OK,
    pack_error,
    pack_infer_response,
    recv_message,
    send_message,
    unpack_infer_request,
    unpack_ping,
)


def _status_code(status: str) -> int:
    if status == "ok":
        return STATUS_OK
    if status == "invalid_no_signal":
        return STATUS_INVALID_NO_SIGNAL
    return STATUS_INTERNAL_ERROR


class InferenceSocketWorker:
    def __init__(
        self,
        socket_path: str,
        checkpoint_path: str,
        scalers_path: str,
        device: str = "auto",
        config: Optional[InferenceConfig] = None,
    ):
        self.socket_path = socket_path
        self.engine = RFInferenceEngine(
            checkpoint_path=checkpoint_path,
            scalers_path=scalers_path,
            device=device,
            config=config or InferenceConfig(),
        )
        self._running = True

    def run_forever(self):
        path = Path(self.socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(self.socket_path)
            server.listen(1)
            print(f"[worker] listening on {self.socket_path}")

            while self._running:
                conn, _ = server.accept()
                with conn:
                    self._handle_client(conn)
        finally:
            server.close()
            if path.exists():
                path.unlink()

    def _handle_client(self, conn: socket.socket):
        while self._running:
            try:
                msg_type, payload = recv_message(conn)
            except ConnectionError:
                return
            except Exception as exc:
                send_message(conn, MSG_ERROR_RESP, pack_error(f"protocol_error: {exc}"))
                return

            if msg_type == MSG_PING_REQ:
                try:
                    seq_id = unpack_ping(payload)
                    send_message(conn, MSG_PING_RESP, payload)
                except Exception as exc:
                    send_message(conn, MSG_ERROR_RESP, pack_error(f"bad_ping: {exc}"))
                continue

            if msg_type == MSG_SHUTDOWN_REQ:
                self._running = False
                send_message(conn, MSG_SHUTDOWN_RESP, b"")
                return

            if msg_type != MSG_INFER_REQ:
                send_message(conn, MSG_ERROR_RESP, pack_error(f"unknown_msg_type: {msg_type}"))
                continue

            try:
                req = unpack_infer_request(payload)
                sample_rate_hz = req["sample_rate_hz"] if req["sample_rate_hz"] > 0 else None
                out = self.engine.infer_compact(
                    iq_samples=req["iq_complex"],
                    power_lna_dbm=req["power_lna_dbm"],
                    power_pa_dbm=req["power_pa_dbm"],
                    sample_rate_hz=sample_rate_hz,
                )
                resp_payload = pack_infer_response(
                    seq_id=req["seq_id"],
                    status_code=_status_code(out["status"]),
                    lna_class=out["lna_class"],
                    filter_class=out["filter_class"],
                    center_class=out["center_class"],
                    mixer_dbm=out["mixer_dbm"],
                    ifamp_db=out["ifamp_db"],
                    evm_value=out["evm_value"],
                    processing_time_ms=out["processing_time_ms"],
                )
                send_message(conn, MSG_INFER_RESP, resp_payload)
            except ValueError as exc:
                send_message(conn, MSG_ERROR_RESP, pack_error(f"bad_request: {exc}"))
            except Exception as exc:
                send_message(conn, MSG_ERROR_RESP, pack_error(f"internal_error: {exc}"))


def main():
    parser = argparse.ArgumentParser(description="MAARS persistent inference socket worker")
    parser.add_argument("--socket-path", default="/tmp/maars_infer.sock")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--scalers", default="checkpoints/scalers.joblib")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-rate-hz", type=float, default=25e6)
    parser.add_argument("--allow-center-shift", action="store_true")
    args = parser.parse_args()

    cfg = InferenceConfig(sample_rate_hz=args.sample_rate_hz, allow_center_shift=args.allow_center_shift)
    worker = InferenceSocketWorker(
        socket_path=args.socket_path,
        checkpoint_path=args.checkpoint,
        scalers_path=args.scalers,
        device=args.device,
        config=cfg,
    )
    worker.run_forever()


if __name__ == "__main__":
    main()
