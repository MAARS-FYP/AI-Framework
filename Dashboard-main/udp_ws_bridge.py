#!/usr/bin/env python3
"""Bridge UDP telemetry packets to WebSocket clients.

Typical flow:
- UDP sender pushes JSON datagrams to this process.
- This process fans out the same JSON to all connected WebSocket clients.
- Dashboard subscribes over WebSocket and updates in real time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import websockets


class BridgeState:
    def __init__(self, ws_path: str, verbose: bool) -> None:
        self.ws_path = ws_path
        self.verbose = verbose
        self.clients: set[Any] = set()
        self.udp_packets = 0
        self.forwarded_messages = 0

    async def handle_ws(self, websocket: Any) -> None:
        client_path = getattr(websocket, "path", None)
        if client_path is None:
            request = getattr(websocket, "request", None)
            client_path = getattr(request, "path", None)

        if self.ws_path and client_path and client_path != self.ws_path:
            if self.verbose:
                print(
                    f"Rejecting WebSocket client on unexpected path {client_path!r}."
                    f" Expected {self.ws_path!r}."
                )
            await websocket.close(code=1008, reason="Unexpected path")
            return

        self.clients.add(websocket)
        if self.verbose:
            print(f"WebSocket client connected. Active clients: {len(self.clients)}")

        try:
            async for _ in websocket:
                # Bridge is one-way: upstream UDP -> websocket clients.
                # Incoming websocket messages are ignored.
                continue
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            if self.verbose:
                print(f"WebSocket client disconnected. Active clients: {len(self.clients)}")

    async def broadcast(self, message: str) -> None:
        if not self.clients:
            return

        targets = list(self.clients)
        results = await asyncio.gather(
            *(client.send(message) for client in targets),
            return_exceptions=True,
        )

        self.forwarded_messages += 1
        stale_clients: list[Any] = []
        for client, result in zip(targets, results):
            if isinstance(result, Exception):
                stale_clients.append(client)

        for client in stale_clients:
            self.clients.discard(client)


class TelemetryUdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, state: BridgeState, loop: asyncio.AbstractEventLoop) -> None:
        self.state = state
        self.loop = loop

    def datagram_received(self, data: bytes, _addr: tuple[str, int]) -> None:
        try:
            decoded = data.decode("utf-8")
            payload = json.loads(decoded)
            if not isinstance(payload, dict):
                return

            message = json.dumps(payload, separators=(",", ":"))
            self.state.udp_packets += 1
            self.loop.create_task(self.state.broadcast(message))

            if self.state.verbose and self.state.udp_packets % 50 == 0:
                print(
                    "UDP packets:"
                    f" {self.state.udp_packets},"
                    f" broadcasts: {self.state.forwarded_messages},"
                    f" ws clients: {len(self.state.clients)}"
                )
        except (UnicodeDecodeError, json.JSONDecodeError):
            return


async def run_bridge(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    state = BridgeState(ws_path=args.ws_path, verbose=args.verbose)

    transport, _protocol = await loop.create_datagram_endpoint(
        lambda: TelemetryUdpProtocol(state, loop),
        local_addr=(args.udp_host, args.udp_port),
    )

    print(
        f"UDP listener ready on {args.udp_host}:{args.udp_port} | "
        f"WebSocket server on {args.ws_host}:{args.ws_port}{args.ws_path}"
    )

    try:
        async with websockets.serve(
            state.handle_ws,
            args.ws_host,
            args.ws_port,
            ping_interval=20,
            ping_timeout=20,
            max_size=1_000_000,
        ):
            await asyncio.Event().wait()
    finally:
        transport.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UDP to WebSocket telemetry bridge")
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP bind host")
    parser.add_argument("--udp-port", type=int, default=9000, help="UDP bind port")
    parser.add_argument("--ws-host", default="127.0.0.1", help="WebSocket bind host")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket bind port")
    parser.add_argument(
        "--ws-path",
        default="/telemetry",
        help="Accepted WebSocket path (set empty string to allow any path)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable bridge logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run_bridge(args))
    except KeyboardInterrupt:
        print("\nBridge stopped by user.")
