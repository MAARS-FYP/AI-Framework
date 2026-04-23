#!/usr/bin/env python3
"""Example external CLI client for valon_worker.py."""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import sys
import uuid
from typing import Any, Dict


DEFAULT_SOCKET_PATH = "/tmp/valon5019.sock"


class ValonClient:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path

    def request(self, op: str, **kwargs: Any) -> Dict[str, Any]:
        req = {"id": str(uuid.uuid4()), "op": op}
        req.update(kwargs)
        payload = (json.dumps(req) + "\n").encode("utf-8")

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(self.socket_path)
            s.sendall(payload)
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
        if not data:
            raise RuntimeError("No response from worker")
        return json.loads(data.decode("utf-8", errors="replace").strip())


def _print_response(resp: Dict[str, Any]) -> None:
    if resp.get("ok"):
        print(json.dumps(resp.get("result", {}), indent=2, sort_keys=True))
        return
    err = resp.get("error", {})
    code = err.get("code", "ERROR")
    msg = err.get("message", "unknown error")
    print(f"{code}: {msg}")


def _help_text() -> str:
    return (
        "Commands:\n"
        "  freq <mhz>         set frequency in MHz\n"
        "  rflevel <dbm>      set RF level in dBm\n"
        "  get                read current frequency/RF level\n"
        "  status             worker/device status\n"
        "  quit               exit client\n"
        "  help               show this text"
    )


def interactive(client: ValonClient) -> int:
    print("Valon CLI example")
    print(_help_text())
    while True:
        try:
            line = input("valon> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        parts = shlex.split(line)
        cmd = parts[0].lower()

        try:
            if cmd in ("quit", "exit"):
                return 0
            if cmd == "help":
                print(_help_text())
                continue
            if cmd == "freq":
                if len(parts) != 2:
                    print("Usage: freq <mhz>")
                    continue
                resp = client.request("set_freq", value_mhz=float(parts[1]))
                _print_response(resp)
                continue
            if cmd == "rflevel":
                if len(parts) != 2:
                    print("Usage: rflevel <dbm>")
                    continue
                resp = client.request("set_rflevel", value_dbm=float(parts[1]))
                _print_response(resp)
                continue
            if cmd == "get":
                _print_response(client.request("get"))
                continue
            if cmd == "status":
                _print_response(client.request("status"))
                continue

            print(f"Unknown command: {cmd}")
            print(_help_text())
        except ValueError:
            print("Invalid numeric value")
        except FileNotFoundError:
            print(f"Worker socket not found: {client.socket_path}")
            print("Start worker first.")
        except Exception as exc:
            print(f"Request failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Example CLI for valon_worker")
    parser.add_argument("--socket", default=DEFAULT_SOCKET_PATH, help="Unix socket path")
    parser.add_argument("--command", default=None, help="Optional one-shot command")
    args = parser.parse_args()

    client = ValonClient(args.socket)

    if args.command:
        parts = shlex.split(args.command)
        if not parts:
            return 0
        cmd = parts[0].lower()
        if cmd == "freq" and len(parts) == 2:
            _print_response(client.request("set_freq", value_mhz=float(parts[1])))
            return 0
        if cmd == "rflevel" and len(parts) == 2:
            _print_response(client.request("set_rflevel", value_dbm=float(parts[1])))
            return 0
        if cmd == "get":
            _print_response(client.request("get"))
            return 0
        if cmd == "status":
            _print_response(client.request("status"))
            return 0
        print("Unsupported one-shot command")
        return 2

    return interactive(client)


if __name__ == "__main__":
    sys.exit(main())
