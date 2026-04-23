#!/usr/bin/env python3
"""Valon headless worker process (Unix socket JSON-line API)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socketserver
import threading
from typing import Any, Dict

from valon_core import ValonController
from valon_protocol import error_response, ok_response


DEFAULT_SOCKET_PATH = "/tmp/valon5019.sock"


class _ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class WorkerService:
    def __init__(self, port: str | None, timeout: float, baud: int):
        self.controller = ValonController(port=port, timeout=timeout, baud=baud)
        self._shutdown_requested = threading.Event()

    def stop(self) -> None:
        self._shutdown_requested.set()
        self.controller.close()

    def dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id")
        op = str(request.get("op", "")).strip().lower()
        try:
            if op == "set_freq":
                result = self.controller.set_frequency_mhz(request.get("value_mhz"))
                lock = self.controller.get_lock()
                result["lock"] = lock
                return ok_response(req_id, result)
            if op == "set_rflevel":
                result = self.controller.set_rf_level_dbm(request.get("value_dbm"))
                lock = self.controller.get_lock()
                result["lock"] = lock
                return ok_response(req_id, result)
            if op == "get":
                result = self.controller.get_current_state()
                result["lock"] = self.controller.get_lock()
                return ok_response(req_id, result)
            if op == "status":
                return ok_response(req_id, self.controller.get_status())
            if op == "shutdown":
                self._shutdown_requested.set()
                return ok_response(req_id, {"message": "worker shutdown requested"})
            return error_response(req_id, "BAD_COMMAND", f"unknown op '{op}'", False)
        except ValueError as exc:
            return error_response(req_id, "OUT_OF_RANGE", str(exc), False)
        except PermissionError as exc:
            return error_response(req_id, "PERMISSION_DENIED", str(exc), False)
        except Exception as exc:
            return error_response(req_id, "DEVICE_ERROR", str(exc), True)


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        service: WorkerService = self.server.service  # type: ignore[attr-defined]
        while True:
            line = self.rfile.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                req = json.loads(text)
                if not isinstance(req, dict):
                    raise ValueError("request must be a JSON object")
            except Exception as exc:
                resp = error_response(None, "BAD_REQUEST", str(exc), False)
                self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
                self.wfile.flush()
                continue

            resp = service.dispatch(req)
            self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
            self.wfile.flush()


def _remove_stale_socket(path: str) -> None:
    if os.path.exists(path):
        if os.path.isfile(path):
            raise RuntimeError(f"Refusing to remove non-socket file at {path}")
        os.unlink(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Valon synthesizer worker")
    parser.add_argument("--socket", default=DEFAULT_SOCKET_PATH, help="Unix socket path")
    parser.add_argument("--port", default=None, help="Optional explicit serial port")
    parser.add_argument("--timeout", default=1.0, type=float, help="Serial timeout seconds")
    parser.add_argument("--baud", default=115200, type=int, help="Target baud rate")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    _remove_stale_socket(args.socket)
    service = WorkerService(port=args.port, timeout=args.timeout, baud=args.baud)
    server = _ThreadedUnixServer(args.socket, RequestHandler)
    server.service = service  # type: ignore[attr-defined]

    # Restrict socket to current user/group.
    os.chmod(args.socket, 0o660)
    logging.info("Worker listening on %s", args.socket)

    stop_event = threading.Event()

    def _signal_handler(signum: int, _frame: Any) -> None:
        logging.info("Signal %s received, shutting down", signum)
        stop_event.set()
        service.stop()
        server.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        while not stop_event.is_set() and not service._shutdown_requested.is_set():
            server.handle_request()
    finally:
        service.stop()
        server.server_close()
        try:
            if os.path.exists(args.socket):
                os.unlink(args.socket)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
