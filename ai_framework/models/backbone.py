"""Dual-branch backbone with gated multimodal fusion and residual refinement."""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Learn a per-latent gate between spectrogram and metric embeddings."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, spec_embed: torch.Tensor, metric_embed: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([spec_embed, metric_embed], dim=1))
        return gate * spec_embed + (1.0 - gate) * metric_embed


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block for latent-space refinement."""

    def __init__(self, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = latent_dim * 2
        self.norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class Backbone(nn.Module):
    """
    Fuses spectrogram features (CNN) with scalar metrics (MLP)
    into a latent vector for downstream agent heads.

    Input:  spectrogram [B, 2, freq, time], metrics [B, 3]
    Output: latent vector [B, latent_dim]
    """

    def __init__(self, latent_dim=64, metric_dim=3, residual_blocks: int = 2, dropout: float = 0.1):
        super().__init__()

        # CNN for 2-channel (real/imag) spectrograms
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 1)),
            nn.Flatten(),
            nn.Linear(32 * 4, 32), nn.ReLU(),
        )

        # MLP for scalar metrics
        self.mlp = nn.Sequential(
            nn.Linear(metric_dim, 16), nn.LayerNorm(16), nn.ReLU(),
            nn.Linear(16, 32), nn.LayerNorm(32), nn.ReLU(),
        )

        self.spec_proj = nn.Sequential(
            nn.Linear(32, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )
        self.metric_proj = nn.Sequential(
            nn.Linear(32, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )
        self.gated_fusion = GatedFusion(latent_dim)
        self.trunk = nn.Sequential(
            *[ResidualMLPBlock(latent_dim=latent_dim, dropout=dropout) for _ in range(residual_blocks)]
        )
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, spectrogram, metrics):
        spec_embed = self.spec_proj(self.cnn(spectrogram))
        metric_embed = self.metric_proj(self.mlp(metrics))
        fused = self.gated_fusion(spec_embed, metric_embed)
        return self.out_norm(self.trunk(fused))
