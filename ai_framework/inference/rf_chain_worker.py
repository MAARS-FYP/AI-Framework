"""
RF Chain Socket Worker

Exposes the RF chain digital twin via Unix socket IPC.
Uses the MAAR protocol to handle RF chain simulation requests.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
from pathlib import Path
from typing import Optional

from ai_framework.inference.rf_engine import RFChainEngine
from ai_framework.inference.protocol import (
    MSG_ERROR_RESP,
    MSG_PING_REQ,
    MSG_PING_RESP,
    MSG_SHUTDOWN_REQ,
    MSG_SHUTDOWN_RESP,
    MSG_RFCHAIN_REQ,
    MSG_RFCHAIN_RESP,
    pack_error,
    pack_rfchain_response,
    recv_message,
    send_message,
    unpack_ping,
    unpack_rfchain_request,
)

logger = logging.getLogger(__name__)


class RFChainSocketWorker:
    """
    Worker that exposes the RF chain digital twin via Unix socket.
    
    Accepts RF chain simulation requests and returns I/Q samples,
    EVM measurements, and power readings.
    """

    def __init__(self, socket_path: str, seed: Optional[int] = None):
        """
        Initialize RF chain socket worker.
        
        Args:
            socket_path: Path to Unix socket
            seed: Random seed for reproducible simulation
        """
        self.socket_path = socket_path
        self.engine = RFChainEngine(seed=seed)
        self._running = True
        self._seq_counter = 0
        logger.info(f"RF Chain Socket Worker initialized, socket: {socket_path}")

    def run_forever(self):
        """Run the worker in an infinite loop, accepting connections."""
        path = Path(self.socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(self.socket_path)
            server.listen(5)
            print(f"[rf_chain_worker] listening on {self.socket_path}")

            while self._running:
                try:
                    server.settimeout(1.0)
                    conn, _ = server.accept()
                    thread = threading.Thread(target=self._handle_client, args=(conn,), daemon=True)
                    thread.start()
                except socket.timeout:
                    continue
                except Exception as exc:
                    if self._running:
                        logger.error(f"Error accepting connection: {exc}")
        finally:
            server.close()
            if path.exists():
                path.unlink()
            print("[rf_chain_worker] shutdown complete")

    def _run_rfchain_request(self, req: dict) -> bytes:
        """
        Process an RF chain simulation request.
        
        Args:
            req: Unpacked RF chain request dictionary
        
        Returns:
            Packed response bytes
        """
        try:
            output = self.engine.process(
                input_power_dbm=req["input_power_dbm"],
                bandwidth_hz=req["bandwidth_hz"],
                center_freq_hz=req["center_freq_hz"],
                lo_freq_hz=req.get("lo_freq_hz", 0.0),
                lna_voltage=req["lna_voltage"],
                lo_power_dbm=req["lo_power_dbm"],
                pa_gain_db=req["pa_gain_db"],
                num_symbols=req.get("num_symbols", 30),
                seq_id=req["seq_id"],
            )
            
            logger.debug(
                f"RF chain processed: EVM={output.evm_percent:.2f}%, "
                f"power_post_lna={output.power_post_lna_dbm:.2f}dBm, "
                f"power_post_pa={output.power_post_pa_dbm:.2f}dBm"
            )
            
            return pack_rfchain_response(
                seq_id=output.seq_id,
                status=output.status,
                i_samples=output.i_samples,
                q_samples=output.q_samples,
                evm_percent=output.evm_percent,
                power_post_lna_dbm=output.power_post_lna_dbm,
                power_post_pa_dbm=output.power_post_pa_dbm,
                processing_time_ms=output.processing_time_ms,
            )
        except Exception as exc:
            logger.error(f"Error processing RF chain request: {exc}", exc_info=True)
            raise

    def _handle_client(self, conn: socket.socket):
        """
        Handle a single client connection.
        
        Processes messages until client disconnects or shutdown is requested.
        
        Args:
            conn: Connected socket
        """
        with conn:
            def _safe_send(msg_type: int, payload: bytes = b"") -> bool:
                try:
                    send_message(conn, msg_type, payload)
                    return True
                except (BrokenPipeError, ConnectionError, OSError) as exc:
                    logger.info(f"Client disconnected during send: {exc}")
                    return False

            while self._running:
                try:
                    msg_type, payload = recv_message(conn)
                except ConnectionError:
                    return
                except Exception as exc:
                    logger.warning(f"Protocol error: {exc}")
                    if not _safe_send(MSG_ERROR_RESP, pack_error(f"protocol_error: {exc}")):
                        return
                    return

                # Handle ping
                if msg_type == MSG_PING_REQ:
                    try:
                        seq_id = unpack_ping(payload)
                        if not _safe_send(MSG_PING_RESP, payload):
                            return
                    except Exception as exc:
                        if not _safe_send(MSG_ERROR_RESP, pack_error(f"bad_ping: {exc}")):
                            return
                    continue

                # Handle shutdown
                if msg_type == MSG_SHUTDOWN_REQ:
                    logger.info("Shutdown requested")
                    self._running = False
                    if not _safe_send(MSG_SHUTDOWN_RESP, b""):
                        return
                    return

                # Handle RF chain request
                if msg_type == MSG_RFCHAIN_REQ:
                    try:
                        req = unpack_rfchain_request(payload)
                        resp_payload = self._run_rfchain_request(req)
                        if not _safe_send(MSG_RFCHAIN_RESP, resp_payload):
                            return
                    except ValueError as exc:
                        logger.warning(f"Bad RF chain request: {exc}")
                        if not _safe_send(MSG_ERROR_RESP, pack_error(f"bad_request: {exc}")):
                            return
                    except Exception as exc:
                        logger.error(f"Internal error processing RF chain request: {exc}")
                        if not _safe_send(MSG_ERROR_RESP, pack_error(f"internal_error: {exc}")):
                            return
                    continue

                # Unknown message type
                logger.warning(f"Unknown message type: {msg_type}")
                if not _safe_send(MSG_ERROR_RESP, pack_error(f"unknown_msg_type: {msg_type}")):
                    return


def main():
    """Main entry point for RF chain worker CLI."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='[rf_chain_worker] %(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="RF Chain Digital Twin Socket Worker")
    parser.add_argument("--socket-path", default="/tmp/maars_rfchain.sock",
                        help="Unix socket path for IPC")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible OFDM generation")
    args = parser.parse_args()

    worker = RFChainSocketWorker(socket_path=args.socket_path, seed=args.seed)
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        print("\n[rf_chain_worker] interrupted by user")


if __name__ == "__main__":
    main()
