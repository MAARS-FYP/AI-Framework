"""
RF Chain Dashboard Backend

Provides WebSocket server for real-time RF chain parameter control and visualization.
Connects to RF chain worker via Unix socket IPC and optional inference worker for agent recommendations.
"""

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
# This script is in /path/to/AI-Framework/rf_chain_dashboard/
# We need to add /path/to/AI-Framework/ to the path
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
    
    Provides:
    - Real-time parameter control
    - RF chain signal processing
    - EVM and power measurements
    - Optional AI agent recommendations
    """

    def __init__(
        self,
        rfchain_socket: str = "/tmp/maars_rfchain.sock",
        inference_socket: Optional[str] = None,
        ws_host: str = "127.0.0.1",
        ws_port: int = 8877,
        params_path: str = "/tmp/maars_digital_twin_params.txt",
    ):
        """
        Initialize the dashboard backend.
        
        Args:
            rfchain_socket: Path to RF chain worker socket
            inference_socket: Path to inference worker socket (optional)
            ws_host: WebSocket server host
            ws_port: WebSocket server port
            params_path: Path to shared parameter file
        """
        self.rfchain_socket = rfchain_socket
        self.inference_socket = inference_socket
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.params_path = params_path
        
        self.rfchain_conn: Optional[socket.socket] = None
        self.inference_conn: Optional[socket.socket] = None
        self.rfchain_response_timeout_sec = 2.0
        
        # Connected WebSocket clients
        self.clients: Set[WebSocketServerProtocol] = set()
        
        # Current parameters
        self.current_params = {
            "power_pre_lna_dbm": -40.0,
            "bandwidth_hz": 10e6,
            "center_freq_hz": 2420e6,
            "lna_voltage": 3.0,
            "lo_power_dbm": 0.0,
            "pa_gain_db": 10.0,
        }
        
        # Sequence counter
        self.seq_id = 0
        
        logger.info(f"Dashboard backend initialized: WS {ws_host}:{ws_port}")

    async def connect_to_rfchain_worker(self) -> bool:
        """Connect to RF chain worker socket."""
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
        """Connect to inference worker socket (if configured)."""
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

    def _send_message_to_socket(
        self, conn: socket.socket, msg_type: int, payload: bytes
    ) -> bool:
        """Send message to socket (non-blocking)."""
        try:
            send_message(conn, msg_type, payload)
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    def _recv_message_from_socket(self, conn: socket.socket) -> Optional[tuple]:
        """Receive message from socket (non-blocking, with timeout)."""
        try:
            # RF chain processing can take longer than a typical UI polling interval,
            # so use a more forgiving timeout before treating the worker as unresponsive.
            conn.settimeout(self.rfchain_response_timeout_sec)
            msg_type, payload = recv_message(conn)
            return msg_type, payload
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    async def process_rfchain(self) -> Optional[Dict[str, Any]]:
        """
        Process RF chain request with current parameters.
        
        Returns:
            Dictionary with I/Q samples, EVM, and power measurements.
        """
        if self.rfchain_conn is None:
            logger.error("Not connected to RF chain worker")
            return None

        try:
            self.seq_id += 1
            
            # Pack and send Request
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
                logger.error("Failed to send RF chain request")
                return None
            
            # Receive response
            result = self._recv_message_from_socket(self.rfchain_conn)
            if result is None:
                logger.warning("No response from RF chain worker")
                return None
            
            msg_type, resp_payload = result
            if msg_type != MSG_RFCHAIN_RESP:
                logger.error(f"Unexpected response type: {msg_type}")
                return None
            
            # Unpack response
            resp = unpack_rfchain_response(resp_payload)
            
            return {
                "seq_id": resp["seq_id"],
                "status": resp["status"],
                "i_samples": resp["i_samples"].tolist(),  # Convert to list for JSON
                "q_samples": resp["q_samples"].tolist(),
                "evm_percent": float(resp["evm_percent"]),
                "power_pre_lna_dbm": float(resp["power_pre_lna_dbm"]),
                "power_post_pa_dbm": float(resp["power_post_pa_dbm"]),
                "processing_time_ms": float(resp["processing_time_ms"]),
            }
            
        except Exception as e:
            logger.error(f"Error processing RF chain: {e}", exc_info=True)
            return None

    async def get_agent_recommendations(
        self, i_samples: np.ndarray, q_samples: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Get AI agent recommendations from inference worker.
        
        Args:
            i_samples: I channel samples
            q_samples: Q channel samples
        
        Returns:
            Dictionary with agent outputs (LNA class, mixer dBm, etc.)
        """
        if self.inference_conn is None:
            return None
        
        try:
            from ai_framework.inference.protocol import (
                pack_infer_request,
                unpack_infer_response,
                MSG_INFER_REQ,
                MSG_INFER_RESP,
            )
            
            iq_complex = i_samples + 1j * q_samples
            
            payload = pack_infer_request(
                seq_id=self.seq_id,
                sample_rate_hz=25e6,
                power_lna_dbm=self.current_params["power_pre_lna_dbm"],
                power_pa_dbm=self.current_params["power_post_pa_dbm"] 
                if hasattr(self, "_last_power_post_pa")
                else -20.0,
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
        """Broadcast message to all connected clients."""
        if not self.clients:
            return
        
        message_json = json.dumps(message)
        # Remove disconnected clients and send to active ones
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message_json)
            except Exception as e:
                logger.warning(f"Error sending to client: {e}")
                disconnected.add(client)
        
        self.clients -= disconnected

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a WebSocket client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                    
                    # Handle parameter update
                    if cmd.get("action") == "update_params":
                        params = cmd.get("params", {})
                        self.current_params.update(params)
                        logger.info(f"Manual parameter update received: {params}")
                        
                        # Write to shared file for software-framework coupling
                        try:
                            with open(self.params_path, "w") as f:
                                for key, value in self.current_params.items():
                                    f.write(f"{key}={value}\n")
                            logger.info(f"Wrote parameters to {self.params_path}")
                        except Exception as e:
                            logger.error(f"Failed to write parameters to {self.params_path}: {e}")
                        
                        # Process RF chain with new parameters
                        rfchain_result = await self.process_rfchain()
                        
                        if rfchain_result:
                            # Store power for agent inference
                            self._last_power_post_pa = rfchain_result["power_post_pa_dbm"]
                            
                            # Convert samples to numpy for agent inference
                            i_array = np.array(rfchain_result["i_samples"], dtype=np.float32)
                            q_array = np.array(rfchain_result["q_samples"], dtype=np.float32)
                            
                            # Get agent recommendations (optional)
                            agent_output = await self.get_agent_recommendations(i_array, q_array)
                            
                            # Build response message
                            response = {
                                "type": "rfchain_update",
                                "timestamp": datetime.now().isoformat(),
                                "params": self.current_params,
                                "rfchain": rfchain_result,
                            }
                            
                            if agent_output:
                                response["agent"] = agent_output
                            
                            # Broadcast to all clients
                            await self.broadcast_to_clients(response)
                        else:
                            error_msg = {
                                "type": "error",
                                "message": "Failed to process RF chain",
                                "timestamp": datetime.now().isoformat(),
                            }
                            await websocket.send(json.dumps(error_msg))
                    
                    # Handle ping
                    elif cmd.get("action") == "ping":
                        pong = {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat(),
                        }
                        await websocket.send(json.dumps(pong))
                    
                    else:
                        logger.warning(f"Unknown action: {cmd.get('action')}")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from client: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)
        
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected: {websocket.remote_address}")

    async def broadcast_updates_continuously(self):
        """Continuously process RF chain and broadcast updates to clients."""
        logger.info("Starting continuous update loop")
        while True:
            try:
                if self.rfchain_conn is None:
                    # Try to reconnect
                    if not await self.connect_to_rfchain_worker():
                        await asyncio.sleep(1)
                        continue
                
                # Process RF chain with current parameters
                rfchain_result = await self.process_rfchain()
                
                if rfchain_result and self.clients:
                    # Store power for agent inference
                    self._last_power_post_pa = rfchain_result["power_post_pa_dbm"]
                    
                    # Convert samples to numpy for agent inference
                    i_array = np.array(rfchain_result["i_samples"], dtype=np.float32)
                    q_array = np.array(rfchain_result["q_samples"], dtype=np.float32)
                    
                    # Get agent recommendations (optional)
                    agent_output = await self.get_agent_recommendations(i_array, q_array)
                    
                    # Build response message
                    response = {
                        "type": "rfchain_update",
                        "timestamp": datetime.now().isoformat(),
                        "params": self.current_params,
                        "rfchain": rfchain_result,
                    }
                    
                    if agent_output:
                        response["agent"] = agent_output
                    
                    # Broadcast to all clients
                    await self.broadcast_to_clients(response)
                
                # Update every 100ms
                await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error in update loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def run(self):
        """Run the WebSocket server."""
        try:
            # Connect to workers (optional, dashboard will wait for them)
            rfchain_connected = await self.connect_to_rfchain_worker()
            if rfchain_connected:
                logger.info("RF chain worker connected")
            else:
                logger.warning("RF chain worker not available yet, dashboard will reconnect automatically")
            
            await self.connect_to_inference_worker()  # Optional
            
            # Start WebSocket server and update loop concurrently
            async with serve(
                self.handle_client,
                self.ws_host,
                self.ws_port,
                ping_interval=20,
                ping_timeout=10,
            ):
                logger.info(f"WebSocket server started on ws://{self.ws_host}:{self.ws_port}")
                
                # Run update loop and server forever
                await self.broadcast_updates_continuously()
        
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            if self.rfchain_conn:
                self.rfchain_conn.close()
            if self.inference_conn:
                self.inference_conn.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RF Chain Dashboard Backend")
    parser.add_argument(
        "--rfchain-socket",
        default="/tmp/maars_rfchain.sock",
        help="RF chain worker socket path",
    )
    parser.add_argument(
        "--inference-socket",
        default=None,
        help="Inference worker socket path (optional)",
    )
    parser.add_argument(
        "--params-path",
        default="/tmp/maars_digital_twin_params.txt",
        help="Shared parameter file path",
    )
    parser.add_argument("--host", default="127.0.0.1", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8877, help="WebSocket server port")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="[rf_dashboard] %(asctime)s - %(levelname)s - %(message)s"
    )
    
    backend = RFChainDashboardBackend(
        rfchain_socket=args.rfchain_socket,
        inference_socket=args.inference_socket,
        ws_host=args.host,
        ws_port=args.port,
        params_path=args.params_path,
    )
    
    try:
        await backend.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    asyncio.run(main())
