from __future__ import annotations

import torch
import torch.nn as nn


class ReducedHardwareFFTNet(nn.Module):
    def __init__(self, input_length: int = 16384, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_length = int(input_length)
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(64 * 64, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.center_head = nn.Linear(hidden_dim, 3)
        self.bandwidth_head = nn.Linear(hidden_dim, 3)

    def forward(self, inputs: torch.Tensor):
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        elif inputs.dim() != 3:
            raise ValueError(f"Expected [B, N] or [B, 1, N], got shape {tuple(inputs.shape)}")

        latent = self.trunk(self.features(inputs))
        return {
            "center_logits": self.center_head(latent),
            "bandwidth_logits": self.bandwidth_head(latent),
        }
