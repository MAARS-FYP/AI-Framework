"""
RF Chain Digital Twin Engine

Simulates the complete RF signal chain including:
- OFDM signal generation
- LNA processing with distortion
- PA distortion modeling
- Power measurements and EVM calculation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Configure root path to find rf_chain and ARVDTNN modules
import sys
from pathlib import Path
module_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(module_root))

from rf_chain import RFChain, OperatingPoint, Setting
from ai_framework.core.dsp import calculate_evm

logger = logging.getLogger(__name__)


@dataclass
class RFChainOutput:
    """Output from RF chain processing"""
    i_samples: np.ndarray  # Shape: (n_samples,)
    q_samples: np.ndarray  # Shape: (n_samples,)
    evm_percent: float
    power_pre_lna_dbm: float  # Input power (before LNA)
    power_post_pa_dbm: float  # Output power (after PA)
    processing_time_ms: float
    seq_id: int = 0
    status: str = "ok"


class RFChainEngine:
    """
    Simulates the digital twin of the RF signal chain.
    
    Generates variable-bandwidth OFDM signals and passes them through
    the complete RF chain model (LNA, mixer, filter, PA) with distortion.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the RF Chain Engine.
        
        Args:
            seed: Random seed for reproducible OFDM generation
        """
        self.rf_chain = RFChain()
        self.seed = seed
        logger.info(f"RF Chain Engine initialized with {len(self.rf_chain.available_voltages)} LNA voltages")

    def apply_pa_distortion(self, x: np.ndarray, gain_db: float = 10.0) -> np.ndarray:
        """
        Apply PA distortion model to signal.
        
        Simulates the LMH6401 DVGA with:
        - Rapp compression
        - Nonlinear k2, k3 terms
        - Memory effects (1st & 2nd order feedback)
        
        Args:
            x: Input complex signal (analytic IF)
            gain_db: PA gain in dB (-6 to 26)
        
        Returns:
            Distorted output signal
        """
        # Clamp gain to valid range
        gain_db_clamped = np.clip(np.round(gain_db), -6, 26)
        G = 10**(gain_db_clamped / 20.0)
        
        # Rapp compression parameters
        v_sat = 4.6
        smoothness = 4.0
        
        x_amp = np.abs(x)
        scaled_mag = (x_amp * G) / (v_sat + 1e-12)
        denominator = (1 + scaled_mag**(2 * smoothness)) ** (1 / (2 * smoothness))
        y_rapp = (x * G) / (denominator + 1e-12)

        # Polynomial distortion terms
        k2 = 0.00022  # 10^(-73/20)
        k3 = 0.00010  # 10^(-80/20)
        
        y_dist = y_rapp + k2 * (y_rapp**2) + k3 * (y_rapp**3)

        # Memory effects (1st and 2nd order feedback)
        pad = np.zeros(2, dtype=np.complex128)
        y_d1 = np.concatenate([pad[:1], y_dist[:-1]])
        y_d2 = np.concatenate([pad[:2], y_dist[:-2]])
        
        y = y_dist + (0.001 * y_d1) + (-0.0005 * y_d2)
        
        return y

    def calculate_power_dbm(self, signal: np.ndarray) -> float:
        """
        Calculate signal power in dBm.
        
        Assumes 50 Ohm impedance (standard RF systems).
        
        Args:
            signal: Complex or real signal
        
        Returns:
            Power in dBm
        """
        if np.iscomplexobj(signal):
            # For complex (analytic) signal: power is from I and Q
            power_linear = np.mean(np.abs(signal)**2)
        else:
            # For real signal
            power_linear = np.mean(signal**2)
        
        # Convert to dBm: P_dBm = 10*log10(P_W * 1000) = 10*log10(P_linear/50*1000)
        # Assuming 1V peak corresponds to power normalized to 50 Ohm
        power_dbm = 10 * np.log10(power_linear + 1e-12)
        return power_dbm

    def process(
        self,
        power_pre_lna_dbm: float = -40.0,
        bandwidth_hz: float = 10e6,
        center_freq_hz: float = 2420e6,
        lo_freq_hz: float = 0.0,
        lna_voltage: float = 3.0,
        lo_power_dbm: float = 0.0,
        pa_gain_db: float = 10.0,
        num_symbols: int = 30,
        seq_id: int = 0,
    ) -> RFChainOutput:
        """
        Process a random OFDM signal through the RF chain digital twin.
        
        Args:
            power_pre_lna_dbm: Input signal power before LNA (-60 to +20 dBm)
            bandwidth_hz: Signal bandwidth (1e6, 10e6, or 20e6 Hz)
            center_freq_hz: Center frequency (for reference, not used in IF processing)
            lna_voltage: LNA supply voltage (3.0 or 5.0 V)
            lo_power_dbm: Local oscillator power (-13.75 to +20 dBm)
            pa_gain_db: PA gain control (-6 to +26 dB)
            num_symbols: Number of OFDM symbols to generate
            seq_id: Sequence ID for tracking
        
        Returns:
            RFChainOutput with I/Q samples, EVM, and power measurements
        """
        start_time = time.time()
        
        try:
            # Use seed for reproducibility if set
            if self.seed is not None:
                np.random.seed(self.seed + seq_id)
            
            # Sampling rate (fixed from RF chain constants)
            FS = 125e6  # 125 MHz
            
            # Generate base OFDM signal with variable bandwidth
            logger.debug(f"Generating OFDM: BW={bandwidth_hz/1e6}MHz, {num_symbols} symbols")
            base_sig, qam_symbols, num_active, cp_len = self.rf_chain.generate_variable_bw_ofdm(
                bandwidth_hz, num_symbols=num_symbols
            )

            # Normalize generated OFDM signal to requested pre-LNA input power.
            base_power_dbm = self.calculate_power_dbm(base_sig)
            input_gain_db = power_pre_lna_dbm - base_power_dbm
            base_sig = base_sig * (10 ** (input_gain_db / 20.0))
            
            # Create operating point and setting
            op_point = OperatingPoint(
                power_pre_lna_dbm=power_pre_lna_dbm,
                bandwidth=bandwidth_hz,
                center_freq=center_freq_hz
            )
            
            setting = Setting(
                lna_voltage=lna_voltage,
                pa_drive_db=0.0,  # No additional PA drive from user control
                lo_power_dbm=lo_power_dbm,
                lo_freq_hz=lo_freq_hz
            )
            
            # Process through RF chain up to PA input
            logger.debug("Processing through RF chain (LNA → Mixer → Filter)")
            x_pa_in = self.rf_chain.process_chain_pre_pa(base_sig, op_point, setting)
            
            # Measure true input power after normalization (pre-LNA).
            power_pre_lna = self.calculate_power_dbm(base_sig)
            
            # Apply the final power scaling from the signal path to get accurate pre-PA measurement
            # The signal has been through LNA, mixer, filter
            power_at_pa_input = self.calculate_power_dbm(x_pa_in)
            
            logger.debug(f"Power before PA input: {power_at_pa_input:.2f} dBm")
            
            # Apply PA distortion with user-controlled gain
            y_pa_out = self.apply_pa_distortion(x_pa_in, gain_db=pa_gain_db)
            
            # Measure output power (post-PA)
            power_post_pa = self.calculate_power_dbm(y_pa_out)
            logger.debug(f"Power after PA output: {power_post_pa:.2f} dBm")
            
            # Extract I/Q samples (take real part of analytic signal for I channel)
            # For complex analytic signal: I = real, Q = imag
            i_samples = np.real(y_pa_out).astype(np.float32)
            q_samples = np.imag(y_pa_out).astype(np.float32)
            
            # Calculate EVM (blind EVM using unit circle metric)
            # Convert to torch for calculation
            iq_complex = torch.from_numpy(y_pa_out.astype(np.complex64))
            evm_tensor = calculate_evm(iq_complex, reference_data=None, normalize=True)
            evm_percent = float(evm_tensor.numpy())
            
            logger.debug(f"EVM: {evm_percent:.2f}%")
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000.0
            
            return RFChainOutput(
                i_samples=i_samples,
                q_samples=q_samples,
                evm_percent=evm_percent,
                power_pre_lna_dbm=power_pre_lna,
                power_post_pa_dbm=power_post_pa,
                processing_time_ms=processing_time_ms,
                seq_id=seq_id,
                status="ok"
            )
            
        except Exception as e:
            logger.error(f"Error processing RF chain: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000.0
            
            return RFChainOutput(
                i_samples=np.array([], dtype=np.float32),
                q_samples=np.array([], dtype=np.float32),
                evm_percent=0.0,
                power_pre_lna_dbm=0.0,
                power_post_pa_dbm=0.0,
                processing_time_ms=processing_time_ms,
                seq_id=seq_id,
                status=f"error: {str(e)}"
            )
