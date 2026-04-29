from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import struct
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Set

# Set up path to find ai_framework module
_script_dir = Path(__file__).parent.absolute()
_project_root = _script_dir.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import websockets
from websockets.server import serve, WebSocketServerProtocol

from ai_framework.inference.protocol import (
    MSG_PING_REQ,
    MSG_PING_RESP,
    MSG_RFCHAIN_REQ,
    MSG_RFCHAIN_RESP,
    MSG_ERROR_RESP,
    pack_rfchain_request,
    unpack_rfchain_response,
    pack_ping,
    unpack_ping,
    send_message,
    recv_message,
)

logger = logging.getLogger(__name__)


class RFChainDashboardBackend:
    """
    WebSocket server for RF chain dashboard.
    """

    def __init__(
        self,
        rfchain_socket: str = "/tmp/maars_rfchain.sock",
        inference_socket: Optional[str] = None,
        ws_host: str = "127.0.0.1",
        ws_port: int = 8877,
        params_path: str = "/tmp/maars_digital_twin_params.txt",
    ):
        self.rfchain_socket = rfchain_socket
        self.inference_socket = inference_socket
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.params_path = params_path
        
        self.rfchain_conn: Optional[socket.socket] = None
        self.inference_conn: Optional[socket.socket] = None
        self.rfchain_response_timeout_sec = 2.0
        
        self.clients: Set[WebSocketServerProtocol] = set()
        
        self.current_params = {
            "power_pre_lna_dbm": -40.0,
            "bandwidth_hz": 10e6,
            "center_freq_hz": 2420e6,
            "lna_voltage": 3.0,
            "lo_power_dbm": 0.0,
            "pa_gain_db": 10.0,
            "manual_mode": 0,
        }
        
        self.seq_id = 0
        logger.info(f"Dashboard backend initialized: WS {ws_host}:{ws_port}")

    async def connect_to_rfchain_worker(self) -> bool:
        try:
            self.rfchain_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.rfchain_conn.connect(self.rfchain_socket)
            self.rfchain_conn.setblocking(False)
            logger.info(f"Connected to RF chain worker at {self.rfchain_socket}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to RF chain worker: {e}")
            return False

    async def connect_to_inference_worker(self) -> bool:
        if self.inference_socket is None:
            return False
        try:
            self.inference_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.inference_conn.connect(self.inference_socket)
            self.inference_conn.setblocking(False)
            logger.info(f"Connected to inference worker at {self.inference_socket}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to inference worker: {e}")
            return False

    def _send_message_to_socket(self, conn: socket.socket, msg_type: int, payload: bytes) -> bool:
        try:
            send_message(conn, msg_type, payload)
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    def _recv_message_from_socket(self, conn: socket.socket) -> Optional[tuple]:
        try:
            conn.settimeout(self.rfchain_response_timeout_sec)
            msg_type, payload = recv_message(conn)
            return msg_type, payload
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    async def process_rfchain(self) -> Optional[Dict[str, Any]]:
        if self.rfchain_conn is None:
            return None
        try:
            self.seq_id += 1
            payload = pack_rfchain_request(
                seq_id=self.seq_id,
                power_pre_lna_dbm=self.current_params["power_pre_lna_dbm"],
                bandwidth_hz=self.current_params["bandwidth_hz"],
                center_freq_hz=self.current_params["center_freq_hz"],
                lna_voltage=self.current_params["lna_voltage"],
                lo_power_dbm=self.current_params["lo_power_dbm"],
                pa_gain_db=self.current_params["pa_gain_db"],
                num_symbols=30,
            )
            if not self._send_message_to_socket(self.rfchain_conn, MSG_RFCHAIN_REQ, payload):
                return None
            result = self._recv_message_from_socket(self.rfchain_conn)
            if result is None:
                return None
            msg_type, resp_payload = result
            if msg_type != MSG_RFCHAIN_RESP:
                return None
            resp = unpack_rfchain_response(resp_payload)
            return {
                "seq_id": resp["seq_id"],
                "status": resp["status"],
                "i_samples": resp["i_samples"].tolist(),
                "q_samples": resp["q_samples"].tolist(),
                "evm_percent": float(resp["evm_percent"]),
                "evm_value": float(resp["evm_percent"]),
                "power_pre_lna_dbm": float(resp["power_pre_lna_dbm"]),
                "power_lna_dbm": float(resp["power_pre_lna_dbm"]),
                "power_lna_raw": float(resp["power_pre_lna_dbm"]),
                "power_post_pa_dbm": float(resp["power_post_pa_dbm"]),
                "power_pa_dbm": float(resp["power_post_pa_dbm"]),
                "power_pa_raw": float(resp["power_post_pa_dbm"]),
                "processing_time_ms": float(resp["processing_time_ms"]),
            }
        except Exception as e:
            logger.error(f"Error processing RF chain: {e}")
            return None

    async def get_agent_recommendations(self, i_samples: np.ndarray, q_samples: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.inference_conn is None:
            return None
        try:
            from ai_framework.inference.protocol import pack_infer_request, unpack_infer_response, MSG_INFER_REQ, MSG_INFER_RESP
            iq_complex = i_samples + 1j * q_samples
            payload = pack_infer_request(
                seq_id=self.seq_id,
                sample_rate_hz=125e6,
                power_lna_dbm=self.current_params["power_pre_lna_dbm"],
                power_pa_dbm=self._last_power_post_pa if hasattr(self, "_last_power_post_pa") else -20.0,
                iq_complex=iq_complex,
            )
            if not self._send_message_to_socket(self.inference_conn, MSG_INFER_REQ, payload):
                return None
            result = self._recv_message_from_socket(self.inference_conn)
            if result is None:
                return None
            msg_type, resp_payload = result
            if msg_type != MSG_INFER_RESP:
                return None
            return unpack_infer_response(resp_payload)
        except Exception as e:
            logger.warning(f"Error getting agent recommendations: {e}")
            return None

    async def broadcast_to_clients(self, message: Dict[str, Any]):
        if not self.clients:
            return
        message_json = json.dumps(message)
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message_json)
            except Exception:
                disconnected.add(client)
        self.clients -= disconnected

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                    if cmd.get("action") == "update_params":
                        params = cmd.get("params", {})
                        self.current_params.update(params)
                        self.current_params["manual_mode"] = 1
                        logger.info(f"Manual update: {params}")
                        try:
                            with open(self.params_path, "w") as f:
                                for k, v in self.current_params.items():
                                    f.write(f"{k}={v}\n")
                        except Exception as e:
                            logger.error(f"Failed write {self.params_path}: {e}")
                        
                        rfchain_result = await self.process_rfchain()
                        if rfchain_result:
                            self._last_power_post_pa = rfchain_result["power_post_pa_dbm"]
                            agent_output = await self.get_agent_recommendations(
                                np.array(rfchain_result["i_samples"], dtype=np.float32),
                                np.array(rfchain_result["q_samples"], dtype=np.float32)
                            )
                            response = {
                                "type": "rfchain_update",
                                "timestamp": datetime.now().isoformat(),
                                "params": self.current_params,
                                "rfchain": rfchain_result,
                            }
                            if agent_output: response["agent"] = agent_output
                            await self.broadcast_to_clients(response)
                    elif cmd.get("action") == "ping":
                        await websocket.send(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
                except Exception as e:
                    logger.error(f"Error handle message: {e}")
        finally:
            self.clients.discard(websocket)

    async def broadcast_updates_continuously(self):
        while True:
            try:
                if self.rfchain_conn is None:
                    if not await self.connect_to_rfchain_worker():
                        await asyncio.sleep(1)
                        continue
                rfchain_result = await self.process_rfchain()
                if rfchain_result and self.clients:
                    self._last_power_post_pa = rfchain_result["power_post_pa_dbm"]
                    agent_output = await self.get_agent_recommendations(
                        np.array(rfchain_result["i_samples"], dtype=np.float32),
                        np.array(rfchain_result["q_samples"], dtype=np.float32)
                    )
                    response = {"type": "rfchain_update", "params": self.current_params, "rfchain": rfchain_result}
                    if agent_output: response["agent"] = agent_output
                    await self.broadcast_to_clients(response)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error loop: {e}")
                await asyncio.sleep(1)

    async def run(self):
        await self.connect_to_rfchain_worker()
        await self.connect_to_inference_worker()
        async with serve(self.handle_client, self.ws_host, self.ws_port):
            await self.broadcast_updates_continuously()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfchain-socket", default="/tmp/maars_rfchain.sock")
    parser.add_argument("--inference-socket", default=None)
    parser.add_argument("--params-path", default="/tmp/maars_digital_twin_params.txt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[rf_dashboard] %(asctime)s - %(levelname)s - %(message)s")
    backend = RFChainDashboardBackend(args.rfchain_socket, args.inference_socket, args.host, args.port, args.params_path)
    await backend.run()

if __name__ == "__main__":
    asyncio.run(main())
