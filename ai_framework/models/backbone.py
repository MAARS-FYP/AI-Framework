"""Dual-branch backbone: CNN (spectrogram) + MLP (metrics) → latent vector."""

import torch
import torch.nn as nn


class Backbone(nn.Module):
    """
    Fuses spectrogram features (CNN) with scalar metrics (MLP)
    into a latent vector for downstream agent heads.

    Input:  spectrogram [B, 2, freq, time], metrics [B, 3]
    Output: latent vector [B, latent_dim]
    """

    def __init__(self, latent_dim=64, metric_dim=3):
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

        # Fusion → latent
        self.fusion = nn.Sequential(
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh(),
        )

    def forward(self, spectrogram, metrics):
        v = self.cnn(spectrogram)
        p = self.mlp(metrics)
        return self.fusion(torch.cat([v, p], dim=1))
