"""
MAARS AI Framework - Demo Script

Demonstrates the complete RF receiver control pipeline:
1. DSP: Raw I/Q → Spectrogram + Metrics
2. Backbone: Features → Latent Vector
3. Agents: Latent → Hardware Commands
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from ai_framework.config import (
    BackboneConfig,
    DSPConfig,
    FrameworkConfig,
    get_logger,
)
from ai_framework.core.dsp import calculate_evm, calculate_power, compute_spectrogram
from ai_framework.models.agents import FilterAgent, IFAmpAgent, LNAAgent, MixerAgent
from ai_framework.models.backbone import UnifiedBackbone

# Setup logging
logger = get_logger(__name__, level=logging.INFO)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(config: Optional[FrameworkConfig] = None) -> None:
    """
    Run the RF receiver control demo.
    
    Args:
        config: Framework configuration. Uses defaults if None.
    """
    if config is None:
        config = FrameworkConfig()
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # --- 1. Initialize Models ---
    logger.info("Initializing backbone and agents...")
    
    backbone_config = BackboneConfig(
        latent_dim=64,
        param_input_dim=3,  # EVM, P_LNA, P_IF
    )
    backbone = UnifiedBackbone(config=backbone_config).to(device)
    
    # Initialize agents with same latent_dim
    latent_size = backbone_config.latent_dim
    lna = LNAAgent(latent_dim=latent_size).to(device)
    mixer = MixerAgent(latent_dim=latent_size).to(device)
    filter_agent = FilterAgent(latent_dim=latent_size).to(device)
    if_amp = IFAmpAgent(latent_dim=latent_size).to(device)
    
    logger.info("Models initialized successfully")
    
    # --- 2. Simulate I/Q Data Processing ---
    batch_size = 4
    time_steps = 256
    
    # Mock raw I/Q data (complex valued)
    iq_data = torch.randn(batch_size, time_steps, dtype=torch.complex64, device=device)
    
    # DSP: Convert to spectrogram
    dsp_config = DSPConfig(n_fft=64, hop_length=16)
    spectrogram = compute_spectrogram(iq_data, config=dsp_config)
    logger.info(f"Spectrogram shape: {spectrogram.shape}")
    
    # DSP: Calculate metrics
    evm = calculate_evm(iq_data)
    power = calculate_power(iq_data)
    
    # Mock additional sensor metrics (P_LNA, P_IF)
    p_lna = torch.randn(batch_size, device=device) * 5 + 10  # ~10dBm
    p_if = torch.randn(batch_size, device=device) * 3 + 5   # ~5dBm
    
    # Stack into sensor_metrics tensor
    sensor_metrics = torch.stack([evm, p_lna, p_if], dim=1)
    logger.info(f"Sensor metrics shape: {sensor_metrics.shape}")
    
    # --- 3. Backbone: Extract Latent Features ---
    with torch.no_grad():
        z = backbone(spectrogram, sensor_metrics)
    logger.info(f"Latent vector shape: {z.shape}")
    
    # --- 4. Agents: Generate Hardware Commands ---
    with torch.no_grad():
        lna_current = lna.get_action(z)
        mixer_freq, mixer_amp = mixer.get_action(z)
        filter_id = filter_agent.get_action(z)
        if_gain = if_amp.get_action(z)
    
    # --- 5. Print Commands ---
    print("\n" + "=" * 70)
    print("HARDWARE COMMANDS (Batch Processing)")
    print("=" * 70)
    
    for i in range(batch_size):
        filter_name = filter_agent.filter_names[filter_id[i].item()]
        print(
            f"Sample {i}: "
            f"LNA={lna_current[i]:.2f}mA | "
            f"LO={mixer_freq[i]:.1f}MHz, {mixer_amp[i]:.3f}V | "
            f"Filter={filter_name} | "
            f"IF_Amp={if_gain[i]:.2f}V"
        )
    
    print("=" * 70)
    
    # --- 6. Demonstrate Exploration Mode ---
    print("\nEXPLORATION MODE (with noise):")
    with torch.no_grad():
        lna_explore = lna.get_action(z, deterministic=False)
        filter_explore = filter_agent.get_action(z, deterministic=False, return_names=True)
    
    print(f"LNA (exploration): {lna_explore}")
    print(f"Filter (exploration): {filter_explore}")


if __name__ == "__main__":
    main()