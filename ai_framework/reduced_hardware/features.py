from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ai_framework.reduced_hardware.config import ReducedHardwareConfig

CENTER_FREQUENCY_CLASS_TO_MHZ = {0: 2405, 1: 2420, 2: 2435}
CENTER_FREQUENCY_MHZ_TO_CLASS = {v: k for k, v in CENTER_FREQUENCY_CLASS_TO_MHZ.items()}
BANDWIDTH_CLASS_TO_MHZ = {0: 1, 1: 10, 2: 20}
BANDWIDTH_MHZ_TO_CLASS = {v: k for k, v in BANDWIDTH_CLASS_TO_MHZ.items()}


def _resolve_sample_column(columns: Iterable[str], sample_column: Optional[str] = None) -> int:
    columns = list(columns)
    if sample_column is not None:
        if sample_column.isdigit():
            return int(sample_column)
        try:
            return columns.index(sample_column)
        except ValueError as exc:
            raise KeyError(f"Sample column '{sample_column}' was not found") from exc

    for idx, name in enumerate(columns):
        lowered = str(name).lower()
        if "probe0" in lowered or "adc" in lowered or lowered == "sample":
            return idx

    for idx, name in enumerate(columns):
        lowered = str(name).lower()
        if "trigger" not in lowered and "window" not in lowered and "buffer" not in lowered:
            return idx

    return 0


def load_ila_adc_samples(
    csv_path: str | Path,
    sample_column: Optional[str] = None,
    max_samples: Optional[int] = 16384,
) -> np.ndarray:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"ADC capture CSV not found: {path}")

    header = pd.read_csv(path, nrows=1)
    data = pd.read_csv(path, skiprows=2, header=None)
    if data.empty:
        raise ValueError(f"ADC capture CSV contains no sample rows: {path}")

    sample_idx = _resolve_sample_column(header.columns, sample_column=sample_column)
    if sample_idx >= data.shape[1]:
        raise IndexError(f"Sample column index {sample_idx} is out of range for {path}")

    samples = pd.to_numeric(data.iloc[:, sample_idx], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if max_samples is not None:
        samples = samples[: int(max_samples)]

    if samples.size == 0:
        raise ValueError(f"No numeric ADC samples could be loaded from {path}")

    return samples


def _pad_or_trim(samples: np.ndarray, n_fft: int) -> np.ndarray:
    if samples.size >= n_fft:
        return samples[:n_fft].astype(np.float32, copy=False)
    padded = np.zeros(int(n_fft), dtype=np.float32)
    padded[: samples.size] = samples.astype(np.float32, copy=False)
    return padded


def compute_fft_feature(
    adc_samples: np.ndarray,
    config: Optional[ReducedHardwareConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = config or ReducedHardwareConfig()
    samples = np.asarray(adc_samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise ValueError("adc_samples must contain at least one value")

    centered = samples - float(samples.mean())
    centered = _pad_or_trim(centered, cfg.n_fft)

    window = np.hanning(cfg.n_fft).astype(np.float32)
    fft_values = np.fft.fftshift(np.fft.fft(centered * window, n=cfg.n_fft))
    magnitude = np.abs(fft_values).astype(np.float32)

    if cfg.use_log_magnitude:
        magnitude = np.log1p(magnitude)

    if cfg.normalize_feature:
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)

    freq_axis_hz = np.fft.fftshift(np.fft.fftfreq(cfg.n_fft, d=1.0 / cfg.sample_rate_hz)).astype(np.float32)
    return magnitude, freq_axis_hz


def center_frequency_to_class(center_frequency_mhz: float) -> int:
    key = int(round(float(center_frequency_mhz)))
    if key not in CENTER_FREQUENCY_MHZ_TO_CLASS:
        raise ValueError(f"Unsupported center frequency: {center_frequency_mhz}")
    return CENTER_FREQUENCY_MHZ_TO_CLASS[key]


def bandwidth_to_class(bandwidth_mhz: float) -> int:
    key = int(round(float(bandwidth_mhz)))
    if key not in BANDWIDTH_MHZ_TO_CLASS:
        raise ValueError(f"Unsupported bandwidth: {bandwidth_mhz}")
    return BANDWIDTH_MHZ_TO_CLASS[key]


def class_to_center_frequency_mhz(center_class: int) -> int:
    if center_class not in CENTER_FREQUENCY_CLASS_TO_MHZ:
        raise ValueError(f"Unsupported center_class: {center_class}")
    return CENTER_FREQUENCY_CLASS_TO_MHZ[int(center_class)]


def class_to_bandwidth_mhz(bandwidth_class: int) -> int:
    if bandwidth_class not in BANDWIDTH_CLASS_TO_MHZ:
        raise ValueError(f"Unsupported bandwidth_class: {bandwidth_class}")
    return BANDWIDTH_CLASS_TO_MHZ[int(bandwidth_class)]
