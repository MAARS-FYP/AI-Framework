"""
Inference script for the MAARS RF Control AI Framework.

Demonstrates how to load trained models with normalizer for deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ai_framework.config import BackboneConfig, get_logger
from ai_framework.core.dsp import compute_spectrogram, DSPConfig
from ai_framework.dataset.dataset import DataNormalizer
from ai_framework.dataset.dataloader import create_inference_dataloader
from ai_framework.models.agents import FilterAgent, IFAmpAgent, LNAAgent, MixerAgent
from ai_framework.models.backbone import UnifiedBackbone

logger = get_logger(__name__)


class RFControlInference:
    """
    Inference engine for RF receiver control.
    
    Loads trained models and normalizer, processes input data, and generates
    hardware control commands.
    
    Args:
        model_dir: Directory containing saved models (backbone, agents, normalizer).
        device: Device to run inference on ('cpu', 'cuda', 'mps').
    
    Example:
        >>> inference = RFControlInference("checkpoints/final_models")
        >>> predictions = inference.predict_from_dataloader(test_loader)
    """
    
    def __init__(
        self,
        model_dir: str | Path,
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        
        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self._load_models()
        
        # Load normalizer (critical for proper inference!)
        self._load_normalizer()
    
    def _load_models(self) -> None:
        """Load all trained models."""
        logger.info("Loading models...")
        
        # Initialize backbone
        backbone_config = BackboneConfig(latent_dim=64, param_input_dim=3)
        self.backbone = UnifiedBackbone(config=backbone_config)
        self.backbone.load_state_dict(
            torch.load(self.model_dir / "backbone.pt", map_location=self.device)
        )
        self.backbone.to(self.device)
        self.backbone.eval()
        
        # Initialize agents
        latent_dim = 64
        self.lna_agent = LNAAgent(latent_dim=latent_dim)
        self.lna_agent.load_state_dict(
            torch.load(self.model_dir / "lna_agent.pt", map_location=self.device)
        )
        self.lna_agent.to(self.device)
        self.lna_agent.eval()
        
        self.mixer_agent = MixerAgent(latent_dim=latent_dim)
        self.mixer_agent.load_state_dict(
            torch.load(self.model_dir / "mixer_agent.pt", map_location=self.device)
        )
        self.mixer_agent.to(self.device)
        self.mixer_agent.eval()
        
        self.filter_agent = FilterAgent(latent_dim=latent_dim)
        self.filter_agent.load_state_dict(
            torch.load(self.model_dir / "filter_agent.pt", map_location=self.device)
        )
        self.filter_agent.to(self.device)
        self.filter_agent.eval()
        
        self.if_amp_agent = IFAmpAgent(latent_dim=latent_dim)
        self.if_amp_agent.load_state_dict(
            torch.load(self.model_dir / "if_amp_agent.pt", map_location=self.device)
        )
        self.if_amp_agent.to(self.device)
        self.if_amp_agent.eval()
        
        logger.info("All models loaded successfully")
    
    def _load_normalizer(self) -> None:
        """Load normalizer with training statistics."""
        normalizer_path = self.model_dir / "normalizer.pt"
        if not normalizer_path.exists():
            raise FileNotFoundError(
                f"Normalizer not found at {normalizer_path}. "
                "Make sure to train with the updated training script that saves normalizer."
            )
        
        normalizer_params = torch.load(normalizer_path, map_location='cpu')
        self.normalizer = DataNormalizer.from_dict(normalizer_params)
        logger.info("Normalizer loaded successfully")
    
    @torch.no_grad()
    def predict(
        self,
        spectrogram: Tensor,
        metrics: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Generate control predictions for a single sample or batch.
        
        Args:
            spectrogram: Normalized spectrogram tensor [batch, 2, freq, time]
            metrics: Normalized metrics tensor [batch, 3]
        
        Returns:
            Dictionary with predictions for each agent (denormalized to real units):
            - 'lna_voltage': Voltage in Volts (3.0 or 5.0)
            - 'mixer_power': LO power in dBm
            - 'filter_bandwidth': Bandwidth in MHz (1.0, 10.0, or 20.0)
            - 'if_amp_gain': Gain in dB
        """
        # Move to device
        spectrogram = spectrogram.to(self.device)
        metrics = metrics.to(self.device)
        
        # Get latent representation
        z = self.backbone(spectrogram, metrics)
        
        # Get predictions from each agent
        lna_logits = self.lna_agent(z)
        lna_class = torch.argmax(lna_logits, dim=1)
        lna_voltage = torch.tensor([3.0 if c == 0 else 5.0 for c in lna_class],
                                   device=self.device)
        
        mixer_output = self.mixer_agent(z)
        mixer_power_norm = mixer_output[:, 1]  # Only power is trained
        mixer_power = self.normalizer.denormalize_mixer_power(mixer_power_norm)
        
        filter_logits = self.filter_agent(z)
        filter_class = torch.argmax(filter_logits, dim=1)
        filter_bw = torch.tensor([1.0, 10.0, 20.0], device=self.device)[filter_class]
        
        if_amp_norm = self.if_amp_agent(z).squeeze(-1)
        if_amp_gain = self.normalizer.denormalize_if_gain(if_amp_norm)
        
        return {
            'lna_voltage': lna_voltage,
            'mixer_power': mixer_power,
            'filter_bandwidth': filter_bw,
            'if_amp_gain': if_amp_gain,
        }
    
    def predict_from_dataloader(
        self,
        dataloader,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Run inference on entire dataset.
        
        Args:
            dataloader: DataLoader with test data (must use same normalizer).
        
        Returns:
            Tuple of (predictions, targets) as dictionaries.
        """
        all_predictions = {
            'lna_voltage': [],
            'mixer_power': [],
            'filter_bandwidth': [],
            'if_amp_gain': [],
        }
        all_targets = {
            'lna': [],
            'mixer_power': [],
            'filter': [],
            'if_amp': [],
        }
        
        with torch.no_grad():
            for (spectrograms, metrics), targets in dataloader:
                # Get predictions
                preds = self.predict(spectrograms, metrics)
                
                # Accumulate
                for key in all_predictions:
                    all_predictions[key].append(preds[key].cpu())
                
                for key in all_targets:
                    all_targets[key].append(targets[key].cpu())
        
        # Concatenate all batches
        for key in all_predictions:
            all_predictions[key] = torch.cat(all_predictions[key], dim=0)
        
        for key in all_targets:
            all_targets[key] = torch.cat(all_targets[key], dim=0)
        
        return all_predictions, all_targets


def load_from_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> RFControlInference:
    """
    Load inference engine from a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., best_checkpoint.pt).
        device: Device to run inference on.
    
    Returns:
        Configured RFControlInference instance.
    
    Example:
        >>> inference = load_from_checkpoint("checkpoints/best_checkpoint.pt")
        >>> predictions = inference.predict(spec, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Auto-select device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    
    # Initialize models
    backbone_config = BackboneConfig(latent_dim=64, param_input_dim=3)
    backbone = UnifiedBackbone(config=backbone_config)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    backbone.to(device)
    backbone.eval()
    
    latent_dim = 64
    lna_agent = LNAAgent(latent_dim=latent_dim)
    lna_agent.load_state_dict(checkpoint['lna_agent_state_dict'])
    lna_agent.to(device)
    lna_agent.eval()
    
    mixer_agent = MixerAgent(latent_dim=latent_dim)
    mixer_agent.load_state_dict(checkpoint['mixer_agent_state_dict'])
    mixer_agent.to(device)
    mixer_agent.eval()
    
    filter_agent = FilterAgent(latent_dim=latent_dim)
    filter_agent.load_state_dict(checkpoint['filter_agent_state_dict'])
    filter_agent.to(device)
    filter_agent.eval()
    
    if_amp_agent = IFAmpAgent(latent_dim=latent_dim)
    if_amp_agent.load_state_dict(checkpoint['if_amp_agent_state_dict'])
    if_amp_agent.to(device)
    if_amp_agent.eval()
    
    # Load normalizer
    normalizer = DataNormalizer.from_dict(checkpoint['normalizer'])
    
    # Create inference instance
    inference = object.__new__(RFControlInference)
    inference.device = device
    inference.backbone = backbone
    inference.lna_agent = lna_agent
    inference.mixer_agent = mixer_agent
    inference.filter_agent = filter_agent
    inference.if_amp_agent = if_amp_agent
    inference.normalizer = normalizer
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return inference


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="RF Control Inference")
    parser.add_argument('--model-dir', type=str, required=True,
                       help="Directory with saved models")
    parser.add_argument('--csv', type=str, required=True,
                       help="CSV file for inference")
    parser.add_argument('--data-root', type=str, default=None,
                       help="Root directory for data files")
    args = parser.parse_args()
    
    # Load inference engine
    inference = RFControlInference(args.model_dir)
    
    # Create inference dataloader (uses normalizer from models)
    test_loader = create_inference_dataloader(
        csv_path=args.csv,
        data_root=args.data_root,
        batch_size=8,
        normalizer=inference.normalizer,  # Use training normalizer
    )
    
    # Run inference
    logger.info("Running inference...")
    predictions, targets = inference.predict_from_dataloader(test_loader)
    
    # Display results
    logger.info(f"\nInference complete on {len(predictions['lna_voltage'])} samples")
    logger.info(f"LNA Voltage predictions: {predictions['lna_voltage'][:5]}")
    logger.info(f"Mixer Power predictions: {predictions['mixer_power'][:5]}")
    logger.info(f"Filter BW predictions: {predictions['filter_bandwidth'][:5]}")
    logger.info(f"IF Amp Gain predictions: {predictions['if_amp_gain'][:5]}")
