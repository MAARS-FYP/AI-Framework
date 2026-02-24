"""Agent heads for RF hardware control."""

import torch.nn as nn


class LNAAgent(nn.Module):
    """Binary classification: 3V (class 0) or 5V (class 1)."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, z):
        return self.net(z)  # [B, 2] logits


class FilterAgent(nn.Module):
    """3-class classification: 1MHz (0), 10MHz (1), 20MHz (2)."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, z):
        return self.net(z)  # [B, 3] logits


class MixerAgent(nn.Module):
    """Regression: predicts normalized LO power."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)  # [B]


class IFAmpAgent(nn.Module):
    """Regression: predicts normalized IF gain."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)  # [B]
