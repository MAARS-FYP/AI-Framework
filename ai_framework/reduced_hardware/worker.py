from __future__ import annotations

import argparse
import socket
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ai_framework.reduced_hardware.config import ReducedHardwareConfig
from ai_framework.reduced_hardware.features import compute_fft_feature, load_ila_adc_samples
from ai_framework.reduced_hardware.model import ReducedHardwareFFTNet
from ai_framework.reduced_hardware.protocol import dumps_message, loads_message, make_error, make_response


class ReducedHardwareInferenceWorker:
    def __init__(
        self,
        socket_path: str,
        checkpoint_path: str,
        device: str = "auto",
        config: Optional[ReducedHardwareConfig] = None,
    ):
        self.socket_path = socket_path
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config or ReducedHardwareConfig()
        self.device = self._resolve_device(device)
        self.model = ReducedHardwareFFTNet(input_length=self.config.n_fft)
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()
        self._running = True

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict)

    @staticmethod
    def _extract_samples(request: Dict[str, Any]) -> np.ndarray:
        if "capture_csv_path" in request:
            return load_ila_adc_samples(request["capture_csv_path"])
        if "adc_samples" in request:
            return np.asarray(request["adc_samples"], dtype=np.float32).reshape(-1)
        raise ValueError("Request must include either capture_csv_path or adc_samples")

    def _infer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        seq_id = int(request.get("seq_id", 0))
        samples = self._extract_samples(request)
        feature, _ = compute_fft_feature(samples, config=self.config)

        inputs = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            center_logits = outputs["center_logits"]
            bandwidth_logits = outputs["bandwidth_logits"]

            center_probs = torch.softmax(center_logits, dim=1)
            bandwidth_probs = torch.softmax(bandwidth_logits, dim=1)

            center_class = int(center_probs.argmax(dim=1).item())
            bandwidth_class = int(bandwidth_probs.argmax(dim=1).item())

        return make_response(
            seq_id=seq_id,
            center_class=center_class,
            bandwidth_class=bandwidth_class,
            center_confidence=float(center_probs.max().item()),
            bandwidth_confidence=float(bandwidth_probs.max().item()),
            capture_source=request.get("capture_csv_path"),
        )

    def _handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        message_type = message.get("type")
        if message_type == "ping":
            return {"type": "pong", "status": "ok", "seq_id": int(message.get("seq_id", 0))}
        if message_type == "shutdown":
            self._running = False
            return {"type": "shutdown_response", "status": "ok"}
        if message_type != "infer":
            raise ValueError(f"Unsupported message type: {message_type}")
        return self._infer(message)

    @staticmethod
    def _read_message(conn: socket.socket) -> Optional[bytes]:
        buffer = bytearray()
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                return bytes(buffer) if buffer else None
            buffer.extend(chunk)
            if b"\n" in chunk:
                break
        return bytes(buffer)

    def run_forever(self) -> None:
        path = Path(self.socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(self.socket_path)
            server.listen(1)
            print(f"[reduced-worker] listening on {self.socket_path}")

            while self._running:
                conn, _ = server.accept()
                with conn:
                    while self._running:
                        raw_message = self._read_message(conn)
                        if raw_message is None:
                            break
                        request: Dict[str, Any]
                        try:
                            request = loads_message(raw_message.strip())
                            response = self._handle_message(request)
                        except Exception as exc:
                            response = make_error(request.get("seq_id") if isinstance(request, dict) else None, str(exc))
                        conn.sendall(dumps_message(response))
        finally:
            server.close()
            if path.exists():
                path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduced-hardware FFT inference worker")
    parser.add_argument("--socket-path", default="/tmp/maars_reduced_hw.sock")
    parser.add_argument("--checkpoint", default="checkpoints/reduced_hardware/reduced_hardware_fftnet.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-rate-hz", type=float, default=125e6)
    parser.add_argument("--n-fft", type=int, default=16384)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = ReducedHardwareConfig(sample_rate_hz=args.sample_rate_hz, n_fft=args.n_fft)
    worker = ReducedHardwareInferenceWorker(
        socket_path=args.socket_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        config=cfg,
    )
    worker.run_forever()


if __name__ == "__main__":
    main()
