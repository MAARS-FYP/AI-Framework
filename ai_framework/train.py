"""
Training script for the MAARS RF Control AI Framework.

Trains the backbone + multi-agent system end-to-end with checkpoint saving.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from ai_framework.config import (
    BackboneConfig,
    FrameworkConfig,
    get_logger,
)
from ai_framework.core.engine import (
    AgentLossWeights,
    MultiAgentTrainer,
    TrainingMetrics,
    evaluate,
    train_one_epoch,
)
from ai_framework.dataset.dataloader import create_dataloaders, get_dataset_stats
from ai_framework.dataset.dataset import DataNormalizer
from ai_framework.models.agents import FilterAgent, IFAmpAgent, LNAAgent, MixerAgent
from ai_framework.models.backbone import UnifiedBackbone

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Data
    csv_path: str = "ai_framework/dataset/data/optimal_control_dataset.csv"
    data_root: Optional[str] = "ai_framework/dataset/data"
    
    # Training
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2
    
    # Scheduler
    scheduler_type: str = "plateau"  # "plateau" or "cosine"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10  # Save checkpoint every N epochs
    save_best: bool = True
    
    # Model
    latent_dim: int = 64
    
    # Hardware ranges (for normalization - IF Amp only, LNA is classification)
    if_amp_min_db: float = -6.0
    if_amp_max_db: float = 26.0
    mixer_power_min: float = 0.0  # Min LO power in dBm
    mixer_power_max: float = 25.0  # Max LO power in dBm
    
    # Loss weights
    lna_loss_weight: float = 1.0
    mixer_loss_weight: float = 1.0  # Only trains LO power
    filter_loss_weight: float = 1.0
    if_amp_loss_weight: float = 1.0


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    epoch: int,
    backbone: torch.nn.Module,
    lna_agent: torch.nn.Module,
    mixer_agent: torch.nn.Module,
    filter_agent: torch.nn.Module,
    if_amp_agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    train_metrics: TrainingMetrics,
    val_metrics: TrainingMetrics,
    config: TrainingConfig,
    normalizer: DataNormalizer,
) -> None:
    """Save training checkpoint with normalizer for inference deployment."""
    checkpoint = {
        'epoch': epoch,
        'backbone_state_dict': backbone.state_dict(),
        'lna_agent_state_dict': lna_agent.state_dict(),
        'mixer_agent_state_dict': mixer_agent.state_dict(),
        'filter_agent_state_dict': filter_agent.state_dict(),
        'if_amp_agent_state_dict': if_amp_agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_metrics': train_metrics.to_dict(),
        'val_metrics': val_metrics.to_dict(),
        'config': asdict(config),
        'normalizer': normalizer.to_dict(),  # Critical for inference!
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: Path,
    backbone: torch.nn.Module,
    lna_agent: torch.nn.Module,
    mixer_agent: torch.nn.Module,
    filter_agent: torch.nn.Module,
    if_amp_agent: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
) -> Tuple[int, Optional[DataNormalizer]]:
    """Load training checkpoint. Returns (epoch_number, normalizer)."""
    checkpoint = torch.load(path, map_location='cpu')
    
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    lna_agent.load_state_dict(checkpoint['lna_agent_state_dict'])
    mixer_agent.load_state_dict(checkpoint['mixer_agent_state_dict'])
    filter_agent.load_state_dict(checkpoint['filter_agent_state_dict'])
    if_amp_agent.load_state_dict(checkpoint['if_amp_agent_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load normalizer if available
    normalizer = None
    if 'normalizer' in checkpoint:
        normalizer = DataNormalizer.from_dict(checkpoint['normalizer'])
        logger.info("Normalizer loaded from checkpoint")
    
    logger.info(f"Checkpoint loaded: {path}, epoch {checkpoint['epoch']}")
    return checkpoint['epoch'], normalizer


def save_models(
    output_dir: Path,
    backbone: torch.nn.Module,
    lna_agent: torch.nn.Module,
    mixer_agent: torch.nn.Module,
    filter_agent: torch.nn.Module,
    if_amp_agent: torch.nn.Module,
    normalizer: DataNormalizer,
) -> None:
    """Save final trained models with normalizer for deployment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(backbone.state_dict(), output_dir / "backbone.pt")
    torch.save(lna_agent.state_dict(), output_dir / "lna_agent.pt")
    torch.save(mixer_agent.state_dict(), output_dir / "mixer_agent.pt")
    torch.save(filter_agent.state_dict(), output_dir / "filter_agent.pt")
    torch.save(if_amp_agent.state_dict(), output_dir / "if_amp_agent.pt")
    
    # Save normalizer separately for easy loading during inference
    torch.save(normalizer.to_dict(), output_dir / "normalizer.pt")
    
    logger.info(f"Models and normalizer saved to {output_dir}")


def train(config: TrainingConfig, resume_from: Optional[str] = None) -> None:
    """
    Main training function.
    
    Args:
        config: Training configuration.
        resume_from: Path to checkpoint to resume from.
    """
    device = get_device()
    logger.info(f"Training on device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training config
    config_path = checkpoint_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # --- Dataset Stats ---
    logger.info("Loading dataset statistics...")
    stats = get_dataset_stats(config.csv_path)
    logger.info(f"Dataset: {stats['num_samples']} samples")
    logger.info(f"Filter distribution: {stats['filter_class_distribution']}")
    
    # --- Data Loaders ---
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        csv_path=config.csv_path,
        data_root=config.data_root,
        batch_size=config.batch_size,
        val_split=config.val_split,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Extract normalizer from dataset for saving with model
    # The normalizer is critical for inference deployment
    normalizer = train_loader.dataset.dataset.normalizer
    logger.info("Normalizer extracted from training dataset")
    
    # --- Models ---
    logger.info("Initializing models...")
    
    # Backbone config - adjust param_input_dim for our 3 metrics
    backbone_config = BackboneConfig(
        latent_dim=config.latent_dim,
        param_input_dim=3,  # EVM, Power Post LNA, Power Post PA
    )
    
    # Need to check spectrogram shape to configure backbone properly
    # STFT complex is (1024, 5) -> 2 channels for (real, imag)
    backbone = UnifiedBackbone(config=backbone_config).to(device)
    
    lna_agent = LNAAgent(
        latent_dim=config.latent_dim,
        voltage_levels=(3.0, 5.0),  # Binary classification
    ).to(device)
    
    # Mixer agent: 2 outputs (freq + power), but only power is trained
    # Frequency output is placeholder for symbolic logic (user will implement)
    mixer_agent = MixerAgent(
        latent_dim=config.latent_dim,
        min_freq_mhz=2405.0,  # Placeholder - not trained
        max_freq_mhz=2483.0,  # Placeholder - not trained
        min_atten_db=config.mixer_power_min,  # Using power range
        max_atten_db=config.mixer_power_max,
    ).to(device)
    
    filter_agent = FilterAgent(
        latent_dim=config.latent_dim,
        filter_bandwidths_mhz=(1.0, 10.0, 20.0),
    ).to(device)
    
    if_amp_agent = IFAmpAgent(
        latent_dim=config.latent_dim,
        min_gain_db=config.if_amp_min_db,
        max_gain_db=config.if_amp_max_db,
    ).to(device)
    
    # --- Optimizer ---
    all_params = (
        list(backbone.parameters()) +
        list(lna_agent.parameters()) +
        list(mixer_agent.parameters()) +
        list(filter_agent.parameters()) +
        list(if_amp_agent.parameters())
    )
    optimizer = optim.AdamW(
        all_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # --- Scheduler ---
    if config.scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
    elif config.scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    else:
        scheduler = None
    
    # --- Trainer ---
    loss_weights = AgentLossWeights(
        lna=config.lna_loss_weight,
        mixer=config.mixer_loss_weight,
        filter=config.filter_loss_weight,
        if_amp=config.if_amp_loss_weight,
    )
    
    trainer = MultiAgentTrainer(
        backbone=backbone,
        lna_agent=lna_agent,
        mixer_agent=mixer_agent,
        filter_agent=filter_agent,
        if_amp_agent=if_amp_agent,
        device=device,
        loss_weights=loss_weights,
        if_amp_db_range=(config.if_amp_min_db, config.if_amp_max_db),
        mixer_power_range=(config.mixer_power_min, config.mixer_power_max),
    )
    
    # --- Resume from checkpoint ---
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from:
        epoch_num, loaded_normalizer = load_checkpoint(
            Path(resume_from),
            backbone, lna_agent, mixer_agent, filter_agent, if_amp_agent,
            optimizer, scheduler
        )
        start_epoch = epoch_num + 1
        if loaded_normalizer:
            normalizer = loaded_normalizer
            logger.info("Using normalizer from checkpoint")
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # --- Training Loop ---
    logger.info("Starting training...")
    history = {'train': [], 'val': []}
    
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_metrics = train_one_epoch(
            trainer, train_loader, optimizer, epoch + 1
        )
        
        # Validate
        val_metrics = evaluate(trainer, val_loader)
        
        # Log metrics
        logger.info(
            f"Train Loss: {train_metrics.total_loss:.4f} | "
            f"Val Loss: {val_metrics.total_loss:.4f} | "
            f"Val Filter Acc: {val_metrics.filter_accuracy*100:.1f}%"
        )
        
        # Update scheduler
        if scheduler:
            if config.scheduler_type == "plateau":
                scheduler.step(val_metrics.total_loss)
            else:
                scheduler.step()
        
        # Save history
        history['train'].append(train_metrics.to_dict())
        history['val'].append(val_metrics.to_dict())
        
        # Save best model
        if config.save_best and val_metrics.total_loss < best_val_loss:
            best_val_loss = val_metrics.total_loss
            save_checkpoint(
                checkpoint_dir / "best_checkpoint.pt",
                epoch, backbone, lna_agent, mixer_agent, filter_agent, if_amp_agent,
                optimizer, scheduler, train_metrics, val_metrics, config, normalizer
            )
            logger.info(f"New best model saved! Val loss: {best_val_loss:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt",
                epoch, backbone, lna_agent, mixer_agent, filter_agent, if_amp_agent,
                optimizer, scheduler, train_metrics, val_metrics, config, normalizer
            )
    
    # --- Save final models ---
    logger.info("\nTraining complete!")
    
    final_model_dir = checkpoint_dir / "final_models"
    save_models(
        final_model_dir,
        backbone, lna_agent, mixer_agent, filter_agent, if_amp_agent, normalizer
    )
    
    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    logger.info(f"\nBest validation loss: {best_val_loss:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MAARS RF Control AI")
    
    parser.add_argument('--csv', type=str, 
                        default="ai_framework/dataset/data/optimal_control_dataset.csv",
                        help="Path to CSV dataset")
    parser.add_argument('--data-root', type=str,
                        default="ai_framework/dataset/data",
                        help="Root directory for data files")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=4,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument('--latent-dim', type=int, default=64,
                        help="Latent dimension for backbone/agents")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    config = TrainingConfig(
        csv_path=args.csv,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        latent_dim=args.latent_dim,
    )
    
    train(config, resume_from=args.resume)
