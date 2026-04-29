"""
Digital Signal Processing utilities for RF signal analysis.

This module provides standalone functions for converting raw I/Q data
into structured features suitable for neural network processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from scipy import signal

from ai_framework.config import DSPConfig, get_logger

logger = get_logger(__name__)

# IMPORTANT: NEED TO CONFIRM THAT SPECTROGRAM CALCULATION FOR DATASET CREATION MATCHES EXACTLY TO THIS.
def compute_spectrogram(
    iq_data: Tensor,
    config: Optional[DSPConfig] = None,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    center: Optional[bool] = None,
) -> Tensor:
    """
    Convert raw I/Q data to a 2-channel spectrogram (Real/Imaginary).
    
        Performs Short-Time Fourier Transform (STFT) using the same settings as
        dataset generation (scipy.signal.stft):
            - nperseg = min(n_fft, signal_length)
            - noverlap = nperseg // 2
            - return_onesided = False

        Then stacks the real and imaginary components as separate channels.
    
    Args:
        iq_data: Complex-valued I/Q samples.
            Shape: [batch, time_steps] or [time_steps]
        config: DSP configuration object. If provided, other parameters
            override config values.
        n_fft: Maximum STFT segment size (FFT_SIZE_STFT). Default: 1024
        hop_length: Unused directly for SciPy path; kept for API compatibility.
        win_length: Window length. Default: n_fft
        center: Whether to center frames (maps to SciPy boundary/padding behavior).
    
    Returns:
        Spectrogram tensor with real and imaginary channels.
        Shape: [batch, 2, freq_bins, time_frames]
        - Channel 0: Real component
        - Channel 1: Imaginary component
    
    Raises:
        ValueError: If iq_data has invalid dimensions.
        TypeError: If iq_data is not a torch.Tensor.
    """
    # Input validation
    if not isinstance(iq_data, Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(iq_data).__name__}")
    
    if iq_data.dim() not in (1, 2):
        raise ValueError(
            f"Expected 1D or 2D tensor, got {iq_data.dim()}D. "
            f"Shape should be [time_steps] or [batch, time_steps]"
        )
    
    # Handle unbatched input
    squeeze_output = False
    if iq_data.dim() == 1:
        iq_data = iq_data.unsqueeze(0)
        squeeze_output = True
    
    # Resolve configuration
    if config is None:
        config = DSPConfig()
    
    _n_fft = n_fft if n_fft is not None else config.n_fft
    _hop_length = hop_length if hop_length is not None else config.hop_length
    _win_length = win_length if win_length is not None else config.win_length
    _center = center if center is not None else config.center
    
    logger.debug(
        f"Computing spectrogram: n_fft={_n_fft}, hop_length={_hop_length}, "
        f"win_length={_win_length}, center={_center}"
    )
    
    # Compute STFT (SciPy path to mirror dataset generation algorithm)
    spec_batch = []
    for b in range(iq_data.shape[0]):
        rx_signal = iq_data[b].detach().cpu().numpy()
        nperseg = min(int(_n_fft), int(rx_signal.shape[-1]))
        if nperseg < 1:
            raise ValueError("Input signal length must be >= 1")
        noverlap = nperseg // 2 if nperseg > 1 else 0

        _, _, zxx = signal.stft(
            rx_signal,
            fs=config.sample_rate_hz,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False,
            boundary="zeros" if _center else None,
            padded=bool(_center),
            detrend=False,
        )
        spec_batch.append(torch.from_numpy(zxx.astype(np.complex64)))

    spec = torch.stack(spec_batch, dim=0).to(iq_data.device)
    
    # Stack Real and Imag parts to make it "Image-like" for the CNN
    # Shape: [batch, 2, freq_bins, time_frames]
    output = torch.stack([spec.real, spec.imag], dim=1)
    
    if squeeze_output:
        output = output.squeeze(0)
    
    return output


def calculate_evm(
    iq_data: Tensor,
    reference_data: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tensor:
    """Calculate Error Vector Magnitude (EVM) for I/Q data."""
    # Input validation
    if not isinstance(iq_data, Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(iq_data).__name__}")
    
    if reference_data is not None:
        if not isinstance(reference_data, Tensor):
            raise TypeError(
                f"reference_data must be torch.Tensor, got {type(reference_data).__name__}"
            )
        if iq_data.shape != reference_data.shape:
            raise ValueError(
                f"Shape mismatch: iq_data {iq_data.shape} vs reference_data {reference_data.shape}"
            )
    
    # Handle unbatched input
    squeeze_output = False
    if iq_data.dim() == 1:
        iq_data = iq_data.unsqueeze(0)
        if reference_data is not None:
            reference_data = reference_data.unsqueeze(0)
        squeeze_output = True
    
    if reference_data is not None:
        # Standard EVM: RMS(|Rx - Tx|) / RMS(|Tx|)
        error_vector = iq_data - reference_data
        evm_rms = torch.sqrt(torch.mean(torch.abs(error_vector) ** 2, dim=-1))
        ref_power = torch.sqrt(torch.mean(torch.abs(reference_data) ** 2, dim=-1))
        
        # Avoid division by zero
        evm = evm_rms / (ref_power + 1e-10)
    else:
        # Blind EVM: Dispersion from unit circle (heuristic for QPSK/QAM)
        magnitude = torch.abs(iq_data)
        error = torch.abs(magnitude - 1.0)
        evm = torch.mean(error, dim=-1)
    
    if normalize:
        evm = evm * 100.0  # Convert to percentage
    
    if squeeze_output:
        evm = evm.squeeze(0)
    
    return evm


def calculate_power(iq_data: Tensor, db: bool = True) -> Tensor:
    """Calculate signal power from I/Q data."""
    if not isinstance(iq_data, Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(iq_data).__name__}")
    
    # Handle unbatched input
    squeeze_output = False
    if iq_data.dim() == 1:
        iq_data = iq_data.unsqueeze(0)
        squeeze_output = True
    
    # Calculate mean power
    power = torch.mean(torch.abs(iq_data) ** 2, dim=-1)
    
    if db:
        power = 10 * torch.log10(power + 1e-10)
    
    if squeeze_output:
        power = power.squeeze(0)
    
    return power


@dataclass
class BandwidthResult:
    """Result container for bandwidth extraction from STFT."""
    low_cutoff_hz: float
    high_cutoff_hz: float
    bandwidth_hz: float
    bandwidth_class: str
    psd_db: Optional[Tensor] = None
    freq_axis_hz: Optional[Tensor] = None


@dataclass
class BandwidthConfig:
    """Configuration for bandwidth extraction algorithm."""
    sample_rate_hz: float = 125e6  # 125 MSPS
    n_fft: int = 2048  # FFT size
    threshold_db: float = 3.0  # -3dB cutoff
    bandwidth_classes: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        '1MHz': (0.0, 5e6),
        '10MHz': (5e6, 15e6),
        '20MHz': (15e6, 125e6),
    })


def extract_bandwidth_from_stft(
    stft_complex: Union[Tensor, "np.ndarray"],
    config: Optional[BandwidthConfig] = None,
    sample_rate_hz: Optional[float] = None,
    n_fft: Optional[int] = None,
    threshold_db: Optional[float] = None,
    return_debug_info: bool = False,
) -> BandwidthResult:
    """Extract bandwidth information from complex STFT data."""
    import numpy as np
    
    if isinstance(stft_complex, np.ndarray):
        stft_complex = torch.from_numpy(stft_complex)
    
    if not stft_complex.is_complex():
        raise ValueError("Expected complex tensor.")
    
    squeeze_output = False
    if stft_complex.dim() == 2:
        stft_complex = stft_complex.unsqueeze(0)
        squeeze_output = True
    
    if config is None:
        config = BandwidthConfig()
    
    _sample_rate = sample_rate_hz if sample_rate_hz is not None else config.sample_rate_hz
    _n_fft = n_fft if n_fft is not None else config.n_fft
    _threshold_db = threshold_db if threshold_db is not None else config.threshold_db
    _bw_classes = config.bandwidth_classes
    
    batch_size, freq_bins, time_frames = stft_complex.shape
    
    magnitude = torch.abs(stft_complex)
    mean_spectrum = torch.mean(magnitude, dim=-1)
    psd_db = 10 * torch.log10(mean_spectrum ** 2 + 1e-12)
    
    freq_resolution = _sample_rate / _n_fft
    freq_axis = torch.arange(freq_bins, dtype=torch.float32) * freq_resolution
    
    results = []
    for b in range(batch_size):
        psd_sample = psd_db[b]
        peak_power_db = torch.max(psd_sample)
        threshold_level = peak_power_db - _threshold_db
        above_threshold = psd_sample >= threshold_level
        indices_above = torch.where(above_threshold)[0]
        
        if len(indices_above) == 0:
            low_idx = 0
            high_idx = freq_bins - 1
        else:
            low_idx = indices_above[0].item()
            high_idx = indices_above[-1].item()
        
        low_cutoff_hz = freq_axis[low_idx].item()
        high_cutoff_hz = freq_axis[high_idx].item()
        bandwidth_hz = high_cutoff_hz - low_cutoff_hz
        
        bandwidth_class = classify_bandwidth(bandwidth_hz, _bw_classes)
        
        results.append(BandwidthResult(
            low_cutoff_hz=low_cutoff_hz,
            high_cutoff_hz=high_cutoff_hz,
            bandwidth_hz=bandwidth_hz,
            bandwidth_class=bandwidth_class,
            psd_db=psd_sample if return_debug_info else None,
            freq_axis_hz=freq_axis if return_debug_info else None,
        ))
    
    return results[0] if squeeze_output else results


def classify_bandwidth(
    bandwidth_hz: float,
    bandwidth_classes: Optional[Dict[str, Tuple[float, float]]] = None,
) -> str:
    """Classify a bandwidth value into predefined classes."""
    if bandwidth_classes is None:
        bandwidth_classes = {
            '1MHz': (0.0, 5e6),
            '10MHz': (5e6, 15e6),
            '20MHz': (15e6, 125e6),
        }
    
    for class_name, (min_hz, max_hz) in bandwidth_classes.items():
        if min_hz <= bandwidth_hz < max_hz:
            return class_name
    
    return 'Unknown'


def compute_psd_from_stft(
    stft_complex: Union[Tensor, "np.ndarray"],
    sample_rate_hz: float = 125e6,
    n_fft: int = 2048,
) -> Tuple[Tensor, Tensor]:
    """Compute Power Spectral Density from complex STFT data."""
    if isinstance(stft_complex, np.ndarray):
        stft_complex = torch.from_numpy(stft_complex)
    
    magnitude = torch.abs(stft_complex)
    if magnitude.dim() == 2:
        mean_spectrum = torch.mean(magnitude, dim=-1)
    else:
        mean_spectrum = magnitude
    
    psd_db = 10 * torch.log10(mean_spectrum ** 2 + 1e-12)
    freq_bins = len(mean_spectrum)
    freq_resolution = sample_rate_hz / n_fft
    freq_axis_hz = torch.arange(freq_bins, dtype=torch.float32) * freq_resolution
    
    return psd_db, freq_axis_hz


FILTER_CLASS_MAP_SYM = {0: "1MHz", 1: "10MHz", 2: "20MHz"}
FILTER_CLASS_INV_SYM = {"1MHz": 0, "10MHz": 1, "20MHz": 2}


def _extract_symbolic_observation(
    stft_complex: Union[Tensor, "np.ndarray"],
    sample_rate_hz: float,
    n_fft: int,
    threshold_db: float,
    center_tolerance_bins: float,
    edge_margin_bins: int,
    energy_floor_db: float,
    min_span_bins: int,
) -> dict:
    """Extract shared PSD/cutoff observation used by coupled symbolic logic."""
    import numpy as np

    if isinstance(stft_complex, Tensor):
        stft_np = stft_complex.detach().cpu().numpy()
    else:
        stft_np = stft_complex

    magnitude = np.abs(stft_np)
    if magnitude.ndim == 3:
        magnitude = magnitude[0]
    psd = np.mean(magnitude ** 2, axis=-1)
    psd_db = 10.0 * np.log10(psd + 1e-12)

    peak_db = float(psd_db.max()) if psd_db.size > 0 else -np.inf
    # Use 6dB threshold for more robust detection of digital twin peaks
    effective_threshold = max(threshold_db, 6.0)
    threshold_level = peak_db - effective_threshold
    above = np.where(psd_db >= threshold_level)[0]

    n_bins = int(stft_np.shape[-2])
    
    # In RF Chain, signal is centered at FC_IF = 25 MHz
    fc_if = 25e6
    # Frequency mapping depends on return_onesided=False
    # [0, FS/2, FS] -> bins [0, n_bins/2, n_bins]
    band_center_bin = (fc_if / sample_rate_hz) * n_bins
    
    peak_bin = int(np.argmax(psd_db)) if psd_db.size > 0 else int(band_center_bin)

    has_cutoffs = len(above) > 0
    if has_cutoffs:
        lower_bin = int(above[0])
        upper_bin = int(above[-1])
        span_bins = int(upper_bin - lower_bin)
        signal_center_bin = (lower_bin + upper_bin) / 2.0
    else:
        lower_bin = peak_bin
        upper_bin = peak_bin
        span_bins = 0
        signal_center_bin = float(peak_bin)

    # Resolution is FS / n_bins because return_onesided=False
    real_freq_res_mhz = sample_rate_hz / n_bins / 1e6
    bw_mhz = span_bins * real_freq_res_mhz

    low_energy = peak_db < energy_floor_db
    degenerate_span = (not has_cutoffs) or (span_bins < min_span_bins)
    no_spectrum = low_energy and degenerate_span

    offset_bins = signal_center_bin - band_center_bin
    offset_mhz = offset_bins * real_freq_res_mhz

    return {
        "n_bins": n_bins,
        "peak_db": peak_db,
        "peak_bin": peak_bin,
        "lower_bin": lower_bin,
        "upper_bin": upper_bin,
        "span_bins": span_bins,
        "bw_mhz": float(bw_mhz),
        "offset_mhz": float(offset_mhz),
        "signal_center_bin": signal_center_bin,
        "band_center_bin": band_center_bin,
        "no_spectrum": bool(no_spectrum),
    }


def symbolic_coupled_filter_center_select(
    stft_complex: Union[Tensor, "np.ndarray"],
    sample_rate_hz: float = 125e6,
    n_fft: int = 2048,
    threshold_db: float = 6.0,
    boundary_low_mhz: float = 5.0,
    boundary_high_mhz: float = 15.0,
    center_freqs_mhz: tuple = (2405, 2420, 2435),
    edge_margin_bins: int = 5,
    center_tolerance_bins: float = 8.0,
    energy_floor_db: float = -90.0,
    min_span_bins: int = 1,
    allow_center_shift: bool = True,
) -> tuple:
    """Coupled symbolic selection of (filter bandwidth class, mixer center frequency)."""
    obs = _extract_symbolic_observation(
        stft_complex=stft_complex,
        sample_rate_hz=sample_rate_hz,
        n_fft=n_fft,
        threshold_db=threshold_db,
        center_tolerance_bins=center_tolerance_bins,
        edge_margin_bins=edge_margin_bins,
        energy_floor_db=energy_floor_db,
        min_span_bins=min_span_bins,
    )

    # DEBUG LOGGING TO /tmp
    try:
        with open("/tmp/maars_dsp_debug.log", "a") as f:
            f.write(f"OFF={obs['offset_mhz']:.2f}MHz SIG={obs['signal_center_bin']:.1f} BND={obs['band_center_bin']:.1f} BW={obs['bw_mhz']:.2f}MHz PEAK={obs['peak_db']:.1f} SHIFT={allow_center_shift}\n")
    except:
        pass

    def classify_bw(bw_mhz: float) -> int:
        if bw_mhz < boundary_low_mhz:
            return 0
        if bw_mhz < boundary_high_mhz:
            return 1
        return 2

    measured_filter = classify_bw(obs["bw_mhz"])
    default_center_class = 1

    if obs["no_spectrum"]:
        return 1, default_center_class, "invalid_no_signal"

    if not allow_center_shift:
        return measured_filter, default_center_class, "ok"

    # Map offset to center class
    offset_mhz = obs["offset_mhz"]
    if offset_mhz < -7.5:
        return measured_filter, 0, "ok"
    if offset_mhz > 7.5:
        return measured_filter, 2, "ok"

    return measured_filter, 1, "ok"


def symbolic_filter_classify(
    stft_complex: Union[Tensor, "np.ndarray"],
    sample_rate_hz: float = 125e6,
    n_fft: int = 2048,
    threshold_db: float = 6.0,
    boundary_low_mhz: float = 5.0,
    boundary_high_mhz: float = 15.0,
) -> int:
    """Symbolic bandwidth-based filter classification."""
    filter_class, _, _ = symbolic_coupled_filter_center_select(
        stft_complex=stft_complex,
        sample_rate_hz=sample_rate_hz,
        n_fft=n_fft,
        threshold_db=threshold_db,
        boundary_low_mhz=boundary_low_mhz,
        boundary_high_mhz=boundary_high_mhz,
    )
    return int(filter_class)


def symbolic_center_freq_classify(
    stft_complex: Union[Tensor, "np.ndarray"],
    filter_class: int,
    sample_rate_hz: float = 125e6,
    n_fft: int = 2048,
    threshold_db: float = 6.0,
    center_freqs_mhz: tuple = (2405, 2420, 2435),
    edge_margin_bins: int = 5,
) -> int:
    """Symbolic RF center-frequency classification."""
    _, center_class, _ = symbolic_coupled_filter_center_select(
        stft_complex=stft_complex,
        sample_rate_hz=sample_rate_hz,
        n_fft=n_fft,
        threshold_db=threshold_db,
        center_freqs_mhz=center_freqs_mhz,
        edge_margin_bins=edge_margin_bins,
    )
    return int(center_class)


def symbolic_filter_classify_batch(
    stft_batch: Union[Tensor, "np.ndarray"],
    **kwargs,
) -> Tensor:
    """Batch version of symbolic_filter_classify."""
    import numpy as np

    if isinstance(stft_batch, Tensor):
        data = stft_batch.detach().cpu().numpy()
    else:
        data = stft_batch

    if data.ndim == 2:
        cls = symbolic_filter_classify(data, **kwargs)
        return torch.tensor(cls, dtype=torch.long)

    results = []
    for i in range(data.shape[0]):
        cls = symbolic_filter_classify(data[i], **kwargs)
        results.append(cls)
    return torch.tensor(results, dtype=torch.long)
