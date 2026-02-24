"""Agent heads for RF hardware control."""

import torch
import torch.nn as nn

from ai_framework.core.dsp import symbolic_filter_classify


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


class SymbolicFilterAgent:
    """
    Symbolic (non-neural) filter classifier.

    Uses PSD histogram with tail removal to extract bandwidth from raw
    complex STFT data, then classifies into filter class via fixed
    boundaries. No learnable parameters — purely rule-based.

    Replaces the neural FilterAgent with 97.3% accuracy on the dataset
    (limited by 8 ambiguous samples whose optimal filter cannot be
    determined from spectral features alone).
    """

    FILTER_NAMES = {0: "1MHz", 1: "10MHz", 2: "20MHz"}

    def __call__(self, stft_raw_batch: torch.Tensor) -> torch.Tensor:
        """
        Predict filter class from raw complex STFT data.

        Args:
            stft_raw_batch: Real-viewed complex STFT, shape [B, freq, time, 2].
                Convert back to complex via torch.view_as_complex.

        Returns:
            Tensor of class indices, shape [B], dtype=long.
        """
        # Reconstruct complex from real view
        stft_complex = torch.view_as_complex(stft_raw_batch)  # [B, freq, time]
        stft_np = stft_complex.detach().cpu().numpy()

        preds = []
        for i in range(stft_np.shape[0]):
            cls = symbolic_filter_classify(stft_np[i])
            preds.append(cls)
        return torch.tensor(preds, dtype=torch.long)

    def parameters(self):
        """No learnable parameters."""
        return iter([])

    def train(self, mode=True):
        return self

    def eval(self):
        return self
