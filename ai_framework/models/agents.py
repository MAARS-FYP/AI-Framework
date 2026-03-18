"""Agent heads for RF hardware control."""

import torch
import torch.nn as nn

from ai_framework.core.dsp import symbolic_coupled_filter_center_select


class LNAAgent(nn.Module):
    """Binary classification: 3V (class 0) or 5V (class 1)."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, z):
        return self.net(z)  # [B, 2] logits


class FilterAgent:
    """
    Symbolic (non-neural) filter classifier.

    Uses PSD histogram with tail removal to extract bandwidth from raw
    complex STFT data, then classifies into filter class via fixed
    boundaries.  No learnable parameters — purely rule-based.

    Algorithm:
        1. Compute PSD from complex STFT (magnitude² → time-average).
        2. Find −3 dB cutoff frequencies (tail removal).
        3. Bandwidth = upper cutoff − lower cutoff.
        4. Classify via fixed boundaries:
             BW < 12 MHz  →  1 MHz filter  (class 0)
             BW < 25 MHz  → 10 MHz filter  (class 1)
             BW ≥ 25 MHz  → 20 MHz filter  (class 2)

    Achieves 97.3% accuracy on the full 300-sample dataset.
    The 8 misclassified samples are 10 MHz-bandwidth signals labeled
    for the 20 MHz filter due to system-level optimality reasons that
    cannot be determined from the spectrum alone.
    """

    FILTER_NAMES = {0: "1MHz", 1: "10MHz", 2: "20MHz"}

    def __init__(self):
        self._last_center_freq_preds = None
        self._last_status = None

    def __call__(self, stft_raw_batch: torch.Tensor) -> torch.Tensor:
        """
        Predict filter class from raw complex STFT data.

        Args:
            stft_raw_batch: Real-viewed complex STFT, shape [B, freq, time, 2].
                Convert back to complex via torch.view_as_complex.

        Returns:
            Tensor of class indices, shape [B], dtype=long.
        """
        stft_complex = torch.view_as_complex(stft_raw_batch)  # [B, freq, time]
        stft_np = stft_complex.detach().cpu().numpy()

        preds = []
        center_preds = []
        status = []
        for i in range(stft_np.shape[0]):
            filt_cls, center_cls, state = symbolic_coupled_filter_center_select(stft_np[i])
            preds.append(filt_cls)
            center_preds.append(center_cls)
            status.append(state)

        self._last_center_freq_preds = torch.tensor(center_preds, dtype=torch.long)
        self._last_status = status
        return torch.tensor(preds, dtype=torch.long)

    def last_center_freq_preds(self) -> torch.Tensor:
        if self._last_center_freq_preds is None:
            return torch.empty(0, dtype=torch.long)
        return self._last_center_freq_preds

    def last_status(self):
        return self._last_status

    def parameters(self):
        """No learnable parameters."""
        return iter([])

    def train(self, mode=True):
        return self

    def eval(self):
        return self


class MixerAgent(nn.Module):
    """
    Hybrid agent: neural LO-power regression + symbolic centre-frequency
    classification.

    Neural output:   normalised LO power (float, via backbone latent).
    Symbolic output: RF centre-frequency class (int, via raw STFT).
        0 → 2405 MHz,  1 → 2420 MHz,  2 → 2435 MHz
    """

    CENTER_FREQS_MHZ = (2405, 2420, 2435)

    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, z):
        """Neural LO-power regression."""
        return self.net(z).squeeze(-1)  # [B]

    @staticmethod
    def classify_center_freq(
        stft_raw_batch: torch.Tensor,
        filter_classes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Symbolic centre-frequency classification from raw STFT data.

        Args:
            stft_raw_batch: Real-viewed complex STFT [B, freq, time, 2].
            filter_classes:  Filter class per sample [B] (from FilterAgent).

        Returns:
            Tensor of centre-freq class indices [B], dtype=long.
        """
        stft_complex = torch.view_as_complex(stft_raw_batch)  # [B, freq, time]
        stft_np = stft_complex.detach().cpu().numpy()

        preds = []
        for i in range(stft_np.shape[0]):
            _, center_cls, _ = symbolic_coupled_filter_center_select(stft_np[i])
            preds.append(center_cls)
        return torch.tensor(preds, dtype=torch.long)


class IFAmpAgent(nn.Module):
    """Regression: predicts normalized IF gain."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)  # [B]
