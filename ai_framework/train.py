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
from typing import Optional

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
    
    # Hardware ranges (for normalization)
    lna_max_ma: float = 20.0
    mixer_min_freq_mhz: float = 2405.0
    mixer_max_freq_mhz: float = 2483.0
    mixer_max_amp_v: float = 1.0
    if_amp_max_v: float = 3.3
    
    # Loss weights
    lna_loss_weight: float = 1.0
    mixer_freq_loss_weight: float = 1.0
    mixer_amp_loss_weight: float = 1.0
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
) -> None:
    """Save training checkpoint."""
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
) -> int:
    """Load training checkpoint. Returns the epoch number."""
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
    
    logger.info(f"Checkpoint loaded: {path}, epoch {checkpoint['epoch']}")
    return checkpoint['epoch']


def save_models(
    output_dir: Path,
    backbone: torch.nn.Module,
    lna_agent: torch.nn.Module,
    mixer_agent: torch.nn.Module,
    filter_agent: torch.nn.Module,
    if_amp_agent: torch.nn.Module,
) -> None:
    """Save final trained models separately."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(backbone.state_dict(), output_dir / "backbone.pt")
    torch.save(lna_agent.state_dict(), output_dir / "lna_agent.pt")
    torch.save(mixer_agent.state_dict(), output_dir / "mixer_agent.pt")
    torch.save(filter_agent.state_dict(), output_dir / "filter_agent.pt")
    torch.save(if_amp_agent.state_dict(), output_dir / "if_amp_agent.pt")
    
    logger.info(f"Models saved to {output_dir}")


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
    
    # --- Models ---
    logger.info("Initializing models...")
    
    # Backbone config - adjust param_input_dim for our 3 metrics
    backbone_config = BackboneConfig(
        latent_dim=config.latent_dim,
        param_input_dim=3,  # EVM, Input Power, PA Input Power
    )
    
    # Need to check spectrogram shape to configure backbone properly
    # STFT complex is (1024, 8) -> 2 channels for (real, imag)
    backbone = UnifiedBackbone(config=backbone_config).to(device)
    
    lna_agent = LNAAgent(
        latent_dim=config.latent_dim,
        max_current_ma=config.lna_max_ma,
    ).to(device)
    
    mixer_agent = MixerAgent(
        latent_dim=config.latent_dim,
        min_freq_mhz=config.mixer_min_freq_mhz,
        max_freq_mhz=config.mixer_max_freq_mhz,
        max_amp_v=config.mixer_max_amp_v,
    ).to(device)
    
    filter_agent = FilterAgent(
        latent_dim=config.latent_dim,
        filter_bandwidths_mhz=(1.0, 10.0, 20.0),
    ).to(device)
    
    if_amp_agent = IFAmpAgent(
        latent_dim=config.latent_dim,
        max_gain_v=config.if_amp_max_v,
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
        mixer_freq=config.mixer_freq_loss_weight,
        mixer_amp=config.mixer_amp_loss_weight,
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
        mixer_freq_range=(config.mixer_min_freq_mhz, config.mixer_max_freq_mhz),
        lna_max_ma=config.lna_max_ma,
        mixer_max_amp=config.mixer_max_amp_v,
        if_amp_max_v=config.if_amp_max_v,
    )
    
    # --- Resume from checkpoint ---
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from:
        start_epoch = load_checkpoint(
            Path(resume_from),
            backbone, lna_agent, mixer_agent, filter_agent, if_amp_agent,
            optimizer, scheduler
        ) + 1
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
                optimizer, scheduler, train_metrics, val_metrics, config
            )
            logger.info(f"New best model saved! Val loss: {best_val_loss:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt",
                epoch, backbone, lna_agent, mixer_agent, filter_agent, if_amp_agent,
                optimizer, scheduler, train_metrics, val_metrics, config
            )
    
    # --- Save final models ---
    logger.info("\nTraining complete!")
    
    final_model_dir = checkpoint_dir / "final_models"
    save_models(
        final_model_dir,
        backbone, lna_agent, mixer_agent, filter_agent, if_amp_agent
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
