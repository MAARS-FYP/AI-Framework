"""
Training and evaluation engine for multi-agent RF control system.

Handles end-to-end training of the backbone + multiple agents simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ai_framework.config import get_logger

logger = get_logger(__name__)


@dataclass
class AgentLossWeights:
    """Weights for combining agent losses."""
    lna: float = 1.0
    mixer_freq: float = 1.0
    mixer_amp: float = 1.0
    filter: float = 1.0
    if_amp: float = 1.0


@dataclass 
class TrainingMetrics:
    """Container for training metrics."""
    total_loss: float
    lna_loss: float
    mixer_freq_loss: float
    mixer_amp_loss: float
    filter_loss: float
    filter_accuracy: float
    if_amp_loss: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'total_loss': self.total_loss,
            'lna_loss': self.lna_loss,
            'mixer_freq_loss': self.mixer_freq_loss,
            'mixer_amp_loss': self.mixer_amp_loss,
            'filter_loss': self.filter_loss,
            'filter_accuracy': self.filter_accuracy,
            'if_amp_loss': self.if_amp_loss,
        }


class MultiAgentTrainer:
    """
    Trainer for end-to-end backbone + multi-agent system.
    
    Trains all components together:
    - Backbone extracts latent features from spectrogram + metrics
    - Each agent predicts optimal control values from latent vector
    
    Args:
        backbone: UnifiedBackbone network.
        lna_agent: LNA control agent.
        mixer_agent: Mixer control agent.
        filter_agent: Filter selection agent.
        if_amp_agent: IF amplifier control agent.
        device: Device to train on.
        loss_weights: Weights for combining agent losses.
        mixer_freq_range: (min, max) frequency in MHz for normalization.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        lna_agent: nn.Module,
        mixer_agent: nn.Module,
        filter_agent: nn.Module,
        if_amp_agent: nn.Module,
        device: torch.device,
        loss_weights: Optional[AgentLossWeights] = None,
        mixer_freq_range: Tuple[float, float] = (2405.0, 2483.0),
        lna_max_ma: float = 20.0,
        mixer_max_amp: float = 1.0,
        if_amp_max_v: float = 3.3,
    ) -> None:
        self.backbone = backbone
        self.lna_agent = lna_agent
        self.mixer_agent = mixer_agent
        self.filter_agent = filter_agent
        self.if_amp_agent = if_amp_agent
        self.device = device
        self.loss_weights = loss_weights or AgentLossWeights()
        
        # Normalization ranges
        self.mixer_freq_min, self.mixer_freq_max = mixer_freq_range
        self.lna_max_ma = lna_max_ma
        self.mixer_max_amp = mixer_max_amp
        self.if_amp_max_v = if_amp_max_v
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def _normalize_lna_target(self, target: Tensor) -> Tensor:
        """Normalize LNA target to [0, 1] range."""
        return target / self.lna_max_ma
    
    def _normalize_mixer_freq_target(self, target: Tensor) -> Tensor:
        """Normalize mixer frequency target to [0, 1] range."""
        return (target - self.mixer_freq_min) / (self.mixer_freq_max - self.mixer_freq_min)
    
    def _normalize_mixer_amp_target(self, target: Tensor) -> Tensor:
        """Normalize mixer amplitude target to [0, 1] range."""
        return target / self.mixer_max_amp
    
    def _normalize_if_amp_target(self, target: Tensor) -> Tensor:
        """Normalize IF amp target to [0, 1] range."""
        # IF gain in dB can be negative, normalize to reasonable range
        # Assuming -20 to +20 dB range, shift to [0, 1]
        return (target + 20) / 40
    
    def compute_losses(
        self,
        z: Tensor,
        targets: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute losses for all agents.
        
        Args:
            z: Latent vector from backbone. Shape: [batch, latent_dim]
            targets: Dictionary of target values for each agent.
        
        Returns:
            Tuple of (total_loss, individual_losses_dict).
        """
        losses = {}
        
        # --- LNA Loss (Regression) ---
        lna_raw = self.lna_agent.forward(z)  # Raw output before sigmoid
        lna_pred = torch.sigmoid(lna_raw).squeeze(-1)  # [0, 1]
        lna_target_norm = self._normalize_lna_target(targets['lna'])
        losses['lna'] = self.mse_loss(lna_pred, lna_target_norm)
        
        # --- Mixer Loss (Regression, dual output) ---
        mixer_raw = self.mixer_agent.forward(z)  # [batch, 2]
        mixer_freq_pred = torch.sigmoid(mixer_raw[:, 0])  # [0, 1]
        mixer_amp_pred = torch.sigmoid(mixer_raw[:, 1])   # [0, 1]
        
        mixer_freq_target_norm = self._normalize_mixer_freq_target(targets['mixer_freq'])
        mixer_amp_target_norm = self._normalize_mixer_amp_target(targets['mixer_amp'])
        
        losses['mixer_freq'] = self.mse_loss(mixer_freq_pred, mixer_freq_target_norm)
        losses['mixer_amp'] = self.mse_loss(mixer_amp_pred, mixer_amp_target_norm)
        
        # --- Filter Loss (Classification) ---
        filter_logits = self.filter_agent.forward(z)  # [batch, num_classes]
        losses['filter'] = self.ce_loss(filter_logits, targets['filter'])
        
        # Calculate filter accuracy
        filter_preds = torch.argmax(filter_logits, dim=1)
        filter_correct = (filter_preds == targets['filter']).float().mean()
        losses['filter_accuracy'] = filter_correct
        
        # --- IF Amp Loss (Regression) ---
        if_amp_raw = self.if_amp_agent.forward(z)
        if_amp_pred = torch.sigmoid(if_amp_raw).squeeze(-1)  # [0, 1]
        if_amp_target_norm = self._normalize_if_amp_target(targets['if_amp'])
        losses['if_amp'] = self.mse_loss(if_amp_pred, if_amp_target_norm)
        
        # --- Total Weighted Loss ---
        total_loss = (
            self.loss_weights.lna * losses['lna'] +
            self.loss_weights.mixer_freq * losses['mixer_freq'] +
            self.loss_weights.mixer_amp * losses['mixer_amp'] +
            self.loss_weights.filter * losses['filter'] +
            self.loss_weights.if_amp * losses['if_amp']
        )
        
        return total_loss, losses
    
    def train_step(
        self,
        batch: Tuple[Tuple[Tensor, Tensor], Dict[str, Tensor]],
        optimizer: Optimizer,
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Tuple of ((spectrogram, metrics), targets).
            optimizer: Optimizer for all model parameters.
        
        Returns:
            Dictionary of loss values.
        """
        (spectrogram, metrics), targets = batch
        
        # Move to device
        spectrogram = spectrogram.to(self.device)
        metrics = metrics.to(self.device)
        targets = {k: v.to(self.device) for k, v in targets.items()}
        
        # Forward pass through backbone
        z = self.backbone(spectrogram, metrics)
        
        # Compute losses
        total_loss, losses = self.compute_losses(z, targets)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'lna_loss': losses['lna'].item(),
            'mixer_freq_loss': losses['mixer_freq'].item(),
            'mixer_amp_loss': losses['mixer_amp'].item(),
            'filter_loss': losses['filter'].item(),
            'filter_accuracy': losses['filter_accuracy'].item(),
            'if_amp_loss': losses['if_amp'].item(),
        }
    
    @torch.no_grad()
    def eval_step(
        self,
        batch: Tuple[Tuple[Tensor, Tensor], Dict[str, Tensor]],
    ) -> Dict[str, float]:
        """
        Perform a single evaluation step.
        
        Args:
            batch: Tuple of ((spectrogram, metrics), targets).
        
        Returns:
            Dictionary of loss values.
        """
        (spectrogram, metrics), targets = batch
        
        # Move to device
        spectrogram = spectrogram.to(self.device)
        metrics = metrics.to(self.device)
        targets = {k: v.to(self.device) for k, v in targets.items()}
        
        # Forward pass
        z = self.backbone(spectrogram, metrics)
        total_loss, losses = self.compute_losses(z, targets)
        
        return {
            'total_loss': total_loss.item(),
            'lna_loss': losses['lna'].item(),
            'mixer_freq_loss': losses['mixer_freq'].item(),
            'mixer_amp_loss': losses['mixer_amp'].item(),
            'filter_loss': losses['filter'].item(),
            'filter_accuracy': losses['filter_accuracy'].item(),
            'if_amp_loss': losses['if_amp'].item(),
        }


def train_one_epoch(
    trainer: MultiAgentTrainer,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    log_interval: int = 10,
) -> TrainingMetrics:
    """
    Train for one epoch.
    
    Args:
        trainer: MultiAgentTrainer instance.
        train_loader: Training data loader.
        optimizer: Optimizer for all parameters.
        epoch: Current epoch number.
        log_interval: How often to log progress.
    
    Returns:
        TrainingMetrics with averaged losses.
    """
    # Set all models to train mode
    trainer.backbone.train()
    trainer.lna_agent.train()
    trainer.mixer_agent.train()
    trainer.filter_agent.train()
    trainer.if_amp_agent.train()
    
    # Accumulators
    total_loss_sum = 0.0
    lna_loss_sum = 0.0
    mixer_freq_loss_sum = 0.0
    mixer_amp_loss_sum = 0.0
    filter_loss_sum = 0.0
    filter_acc_sum = 0.0
    if_amp_loss_sum = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        losses = trainer.train_step(batch, optimizer)
        
        total_loss_sum += losses['total_loss']
        lna_loss_sum += losses['lna_loss']
        mixer_freq_loss_sum += losses['mixer_freq_loss']
        mixer_amp_loss_sum += losses['mixer_amp_loss']
        filter_loss_sum += losses['filter_loss']
        filter_acc_sum += losses['filter_accuracy']
        if_amp_loss_sum += losses['if_amp_loss']
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {losses['total_loss']:.4f} "
                f"(LNA: {losses['lna_loss']:.4f}, "
                f"Mixer: {losses['mixer_freq_loss']:.4f}/{losses['mixer_amp_loss']:.4f}, "
                f"Filter: {losses['filter_loss']:.4f} acc={losses['filter_accuracy']*100:.1f}%, "
                f"IF: {losses['if_amp_loss']:.4f})"
            )
    
    return TrainingMetrics(
        total_loss=total_loss_sum / num_batches,
        lna_loss=lna_loss_sum / num_batches,
        mixer_freq_loss=mixer_freq_loss_sum / num_batches,
        mixer_amp_loss=mixer_amp_loss_sum / num_batches,
        filter_loss=filter_loss_sum / num_batches,
        filter_accuracy=filter_acc_sum / num_batches,
        if_amp_loss=if_amp_loss_sum / num_batches,
    )


@torch.no_grad()
def evaluate(
    trainer: MultiAgentTrainer,
    val_loader: DataLoader,
) -> TrainingMetrics:
    """
    Evaluate on validation set.
    
    Args:
        trainer: MultiAgentTrainer instance.
        val_loader: Validation data loader.
    
    Returns:
        TrainingMetrics with averaged losses.
    """
    # Set all models to eval mode
    trainer.backbone.eval()
    trainer.lna_agent.eval()
    trainer.mixer_agent.eval()
    trainer.filter_agent.eval()
    trainer.if_amp_agent.eval()
    
    # Accumulators
    total_loss_sum = 0.0
    lna_loss_sum = 0.0
    mixer_freq_loss_sum = 0.0
    mixer_amp_loss_sum = 0.0
    filter_loss_sum = 0.0
    filter_acc_sum = 0.0
    if_amp_loss_sum = 0.0
    num_batches = 0
    
    for batch in val_loader:
        losses = trainer.eval_step(batch)
        
        total_loss_sum += losses['total_loss']
        lna_loss_sum += losses['lna_loss']
        mixer_freq_loss_sum += losses['mixer_freq_loss']
        mixer_amp_loss_sum += losses['mixer_amp_loss']
        filter_loss_sum += losses['filter_loss']
        filter_acc_sum += losses['filter_accuracy']
        if_amp_loss_sum += losses['if_amp_loss']
        num_batches += 1
    
    return TrainingMetrics(
        total_loss=total_loss_sum / num_batches,
        lna_loss=lna_loss_sum / num_batches,
        mixer_freq_loss=mixer_freq_loss_sum / num_batches,
        mixer_amp_loss=mixer_amp_loss_sum / num_batches,
        filter_loss=filter_loss_sum / num_batches,
        filter_accuracy=filter_acc_sum / num_batches,
        if_amp_loss=if_amp_loss_sum / num_batches,
    )
