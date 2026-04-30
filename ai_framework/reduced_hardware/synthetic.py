from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ai_framework.reduced_hardware.features import class_to_bandwidth_mhz, class_to_center_frequency_mhz


@dataclass(frozen=True)
class SyntheticCaptureSpec:
    center_class: int
    bandwidth_class: int
    seed: int


def _synthesize_samples(spec: SyntheticCaptureSpec, sample_rate_hz: float, num_samples: int) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate_hz)

    center_hz = float(class_to_center_frequency_mhz(spec.center_class)) * 1e6
    bandwidth_hz = float(class_to_bandwidth_mhz(spec.bandwidth_class)) * 1e6

    tone_offsets = np.linspace(-0.35, 0.35, 5, dtype=np.float32) * bandwidth_hz
    if spec.bandwidth_class == 0:
        tone_offsets = np.array([0.0], dtype=np.float32)

    signal = np.zeros(num_samples, dtype=np.float32)
    for offset in tone_offsets:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += np.sin(2.0 * np.pi * (center_hz + float(offset)) * t + phase).astype(np.float32)

    signal /= max(1, len(tone_offsets))
    envelope = 0.65 + 0.35 * np.hanning(num_samples).astype(np.float32)
    noise = rng.normal(scale=0.08 + 0.02 * spec.bandwidth_class, size=num_samples).astype(np.float32)

    samples = (signal * envelope + noise) * 72.0
    return np.clip(np.round(samples), -128, 127).astype(np.int16)


def write_ila_capture_csv(path: str | Path, samples: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "Sample in Buffer,Sample in Window,TRIGGER,design_1_i/system_ila_0/inst/probe0[7:0],"
            "design_1_i/system_ila_0/inst/probe1[0:0],design_1_i/system_ila_0/inst/probe2[0:0]\n"
        )
        handle.write("Radix - UNSIGNED,UNSIGNED,UNSIGNED,SIGNED,BINARY,HEX\n")
        for idx, sample in enumerate(np.asarray(samples).reshape(-1)):
            handle.write(f"{idx},{idx},0,{int(sample)},{idx % 2},1\n")

    return output_path


def generate_synthetic_dataset(
    output_dir: str | Path,
    samples_per_class: int = 4,
    sample_rate_hz: float = 125e6,
    num_samples: int = 16384,
) -> Path:
    output_dir = Path(output_dir)
    capture_dir = output_dir / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seed = 1000
    for center_class in range(3):
        for bandwidth_class in range(3):
            for example_idx in range(samples_per_class):
                spec = SyntheticCaptureSpec(center_class=center_class, bandwidth_class=bandwidth_class, seed=seed)
                samples = _synthesize_samples(spec, sample_rate_hz=sample_rate_hz, num_samples=num_samples)
                capture_name = f"capture_c{center_class}_b{bandwidth_class}_{example_idx:03d}.csv"
                capture_path = write_ila_capture_csv(capture_dir / capture_name, samples)
                rows.append(
                    {
                        "capture_csv_path": str(capture_path.relative_to(output_dir)),
                        "center_frequency_mhz": class_to_center_frequency_mhz(center_class),
                        "bandwidth_mhz": class_to_bandwidth_mhz(bandwidth_class),
                    }
                )
                seed += 1

    manifest_path = output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path
