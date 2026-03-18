from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ai_framework.inference import InferenceConfig, RFInferenceEngine


def _load_iq_from_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if np.iscomplexobj(arr):
        if arr.ndim != 1:
            raise ValueError("Complex IQ .npy must be 1D.")
        return arr.astype(np.complex64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
    raise ValueError("IQ .npy must be complex 1D or real Nx2 [I,Q].")


def _load_payload(args) -> Dict[str, Any]:
    if args.input_json:
        return json.loads(Path(args.input_json).read_text())
    if args.stdin_json:
        return json.loads(sys.stdin.read())

    if not args.iq_npy:
        raise ValueError("Provide one of --iq-npy, --input-json, or --stdin-json")
    if args.power_lna_dbm is None or args.power_pa_dbm is None:
        raise ValueError("--power-lna-dbm and --power-pa-dbm are required with --iq-npy")

    return {
        "iq_source": "npy",
        "iq_npy": args.iq_npy,
        "power_lna_dbm": float(args.power_lna_dbm),
        "power_pa_dbm": float(args.power_pa_dbm),
        "sample_rate_hz": args.sample_rate_hz,
    }


def _extract_iq(payload: Dict[str, Any]) -> np.ndarray:
    if payload.get("iq_source") == "npy" and payload.get("iq_npy"):
        return _load_iq_from_npy(payload["iq_npy"])

    if "iq_real" in payload and "iq_imag" in payload:
        i = np.asarray(payload["iq_real"], dtype=np.float32)
        q = np.asarray(payload["iq_imag"], dtype=np.float32)
        if i.shape != q.shape:
            raise ValueError("iq_real and iq_imag must have the same length")
        return (i + 1j * q).astype(np.complex64)

    if "iq" in payload:
        return np.asarray(payload["iq"])

    raise ValueError("Payload must include iq_real/iq_imag or iq")


def main():
    parser = argparse.ArgumentParser(description="MAARS end-to-end inference CLI")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--scalers", default="checkpoints/scalers.joblib")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--iq-npy", help="Path to .npy IQ input (complex1D or Nx2 real IQ)")
    parser.add_argument("--power-lna-dbm", type=float)
    parser.add_argument("--power-pa-dbm", type=float)
    parser.add_argument("--sample-rate-hz", type=float, default=None)

    parser.add_argument("--input-json", help="Path to input JSON payload")
    parser.add_argument("--stdin-json", action="store_true", help="Read JSON payload from stdin")
    parser.add_argument("--output-json", help="Path to write output JSON; defaults to stdout")

    args = parser.parse_args()

    try:
        payload = _load_payload(args)
        iq_samples = _extract_iq(payload)
        power_lna_dbm = float(payload["power_lna_dbm"])
        power_pa_dbm = float(payload["power_pa_dbm"])
        sample_rate_hz = payload.get("sample_rate_hz", args.sample_rate_hz)

        cfg = InferenceConfig()
        engine = RFInferenceEngine(
            checkpoint_path=args.checkpoint,
            scalers_path=args.scalers,
            device=args.device,
            config=cfg,
        )

        result = engine.infer_from_iq_and_power(
            iq_samples=iq_samples,
            power_lna_dbm=power_lna_dbm,
            power_pa_dbm=power_pa_dbm,
            sample_rate_hz=sample_rate_hz,
        ).to_dict()

        out = json.dumps(result, indent=2)
        if args.output_json:
            Path(args.output_json).write_text(out)
        else:
            print(out)

    except Exception as exc:
        err = {"status": "error", "message": str(exc)}
        print(json.dumps(err), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
