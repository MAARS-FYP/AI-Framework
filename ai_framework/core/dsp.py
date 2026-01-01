"""
Digital Signal Processing utilities for RF signal analysis.

This module provides standalone functions for converting raw I/Q data
into structured features suitable for neural network processing.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch
from torch import Tensor

from ai_framework.config import DSPConfig, get_logger

logger = get_logger(__name__)


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
    
    Performs Short-Time Fourier Transform (STFT) on the input signal and
    stacks the real and imaginary components as separate channels.
    
    Args:
        iq_data: Complex-valued I/Q samples.
            Shape: [batch, time_steps] or [time_steps]
        config: DSP configuration object. If provided, other parameters
            override config values.
        n_fft: Number of FFT points. Default: 64
        hop_length: Hop length between frames. Default: 16
        win_length: Window length. Default: n_fft
        center: Whether to pad signal. Default: False
    
    Returns:
        Spectrogram tensor with real and imaginary channels.
        Shape: [batch, 2, freq_bins, time_frames]
        - Channel 0: Real component
        - Channel 1: Imaginary component
    
    Raises:
        ValueError: If iq_data has invalid dimensions.
        TypeError: If iq_data is not a torch.Tensor.
    
    Example:
        >>> iq = torch.randn(4, 256, dtype=torch.complex64)
        >>> spec = compute_spectrogram(iq, n_fft=64)
        >>> spec.shape
        torch.Size([4, 2, 33, 13])
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
    
    # Compute STFT
    spec = torch.stft(
        iq_data,
        n_fft=_n_fft,
        hop_length=_hop_length,
        win_length=_win_length,
        return_complex=True,
        center=_center,
    )
    
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
    """
    Calculate Error Vector Magnitude (EVM) for I/Q data.
    
    EVM measures the quality of a modulated signal by comparing received
    symbols against ideal reference points. For blind operation (no reference),
    uses dispersion from the unit circle as a heuristic.
    
    Args:
        iq_data: Received I/Q samples.
            Shape: [batch, symbols] or [symbols]
        reference_data: Ideal transmitted symbols (optional).
            Shape: Must match iq_data
        normalize: Whether to return EVM as percentage. Default: True
    
    Returns:
        EVM value(s) as percentage (if normalize=True) or raw ratio.
        Shape: [batch] or scalar tensor
    
    Raises:
        ValueError: If shapes of iq_data and reference_data don't match.
        TypeError: If inputs are not torch.Tensor.
    
    Example:
        >>> rx = torch.randn(4, 100, dtype=torch.complex64)
        >>> evm = calculate_evm(rx)
        >>> evm.shape
        torch.Size([4])
    """
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
        
        logger.debug(f"Computed reference-based EVM: {evm.mean():.4f}")
    else:
        # Blind EVM: Dispersion from unit circle (heuristic for QPSK/QAM)
        magnitude = torch.abs(iq_data)
        error = torch.abs(magnitude - 1.0)
        evm = torch.mean(error, dim=-1)
        
        logger.debug(f"Computed blind EVM (unit circle): {evm.mean():.4f}")
    
    if normalize:
        evm = evm * 100.0  # Convert to percentage
    
    if squeeze_output:
        evm = evm.squeeze(0)
    
    return evm


def calculate_power(iq_data: Tensor, db: bool = True) -> Tensor:
    """
    Calculate signal power from I/Q data.
    
    Args:
        iq_data: I/Q samples.
            Shape: [batch, samples] or [samples]
        db: Return power in dB (default) or linear.
    
    Returns:
        Power value(s).
        Shape: [batch] or scalar tensor
    
    Example:
        >>> iq = torch.randn(4, 256, dtype=torch.complex64)
        >>> power = calculate_power(iq)
        >>> power.shape
        torch.Size([4])
    """
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
