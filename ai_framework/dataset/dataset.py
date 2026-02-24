"""Dataset and DataLoader for RF control training."""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler

FILTER_CLASS_MAP = {"1MHz": 0, "10MHz": 1, "20MHz": 2}


class RFDataset(Dataset):
    """
    Loads pre-computed STFT spectrograms + sensor metrics for training.

    Inputs per sample:
        spectrogram: [2, 1024, 5] (real/imag channels, per-sample normalized)
        metrics:     [3]          (EVM, P_LNA, P_PA — StandardScaler'd)

    Targets per sample:
        lna:         class index  (0=3V, 1=5V)
        filter:      class index  (0=1MHz, 1=10MHz, 2=20MHz)
        if_amp:      float        (StandardScaler'd IF gain)
        mixer_power: float        (StandardScaler'd LO power)
    """

    def __init__(self, csv_path, data_root=None, scalers=None):
        self.data_root = Path(data_root) if data_root else Path(csv_path).parent
        df = pd.read_csv(csv_path)

        # Classification targets
        self.lna_targets = (df["Optimal_LNA_Voltage_V"] > 4.0).astype(int).values # Binary classification: 3V vs 5V
        self.filter_targets = df["Detected_BW_Class"].map(FILTER_CLASS_MAP).values
        self.stft_files = df["STFT_Complex_File"].values

        # Fit or reuse scalers
        metric_cols = ["Best_EVM_dB", "Measured_Power_Post_LNA_dBm", "Measured_Power_Post_PA_dBm"]
        if scalers is None:
            scalers = {
                "metrics": StandardScaler().fit(df[metric_cols]),
                "if_gain": StandardScaler().fit(df[["Optimal_IF_Gain_dB"]]),
                "mixer_power": StandardScaler().fit(df[["Optimal_LO_Power_dBm"]]),
            }
        self.scalers = scalers

        # Pre-scale all tabular data (avoids repeated transform in __getitem__)
        self.metrics = scalers["metrics"].transform(df[metric_cols]).astype(np.float32)
        self.if_gain = scalers["if_gain"].transform(df[["Optimal_IF_Gain_dB"]]).ravel().astype(np.float32)
        self.mixer_power = scalers["mixer_power"].transform(df[["Optimal_LO_Power_dBm"]]).ravel().astype(np.float32)

    def __len__(self):
        return len(self.lna_targets)

    def __getitem__(self, idx):
        # Load complex STFT → 2-channel (real/imag), per-sample z-score
        stft = np.load(self.data_root / "stft_complex" / self.stft_files[idx])
        real, imag = stft.real, stft.imag
        spec = np.stack([
            (real - real.mean()) / (real.std() + 1e-8),
            (imag - imag.mean()) / (imag.std() + 1e-8),
        ], axis=0).astype(np.float32)

        inputs = (
            torch.from_numpy(spec),
            torch.from_numpy(self.metrics[idx]),
        )
        targets = {
            "lna": torch.tensor(self.lna_targets[idx], dtype=torch.long),
            "filter": torch.tensor(self.filter_targets[idx], dtype=torch.long),
            "if_amp": torch.tensor(self.if_gain[idx], dtype=torch.float32),
            "mixer_power": torch.tensor(self.mixer_power[idx], dtype=torch.float32),
        }
        return inputs, targets


def collate_fn(batch):
    """Stack samples into batches."""
    specs = torch.stack([b[0][0] for b in batch])
    metrics = torch.stack([b[0][1] for b in batch])
    targets = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    return (specs, metrics), targets


def create_dataloaders(csv_path, data_root=None, batch_size=8, val_split=0.2, seed=42):
    """
    Create train/val DataLoaders. Scalers are fit on training split only.

    Returns: (train_loader, val_loader, scalers)
    """
    df = pd.read_csv(csv_path)
    n = len(df)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    val_size = int(n * val_split)
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    # Fit scalers on training data only (no data leakage)
    train_df = df.iloc[train_idx]
    metric_cols = ["Best_EVM_dB", "Measured_Power_Post_LNA_dBm", "Measured_Power_Post_PA_dBm"]
    scalers = {
        "metrics": StandardScaler().fit(train_df[metric_cols]),
        "if_gain": StandardScaler().fit(train_df[["Optimal_IF_Gain_dB"]]),
        "mixer_power": StandardScaler().fit(train_df[["Optimal_LO_Power_dBm"]]),
    }

    dataset = RFDataset(csv_path, data_root, scalers=scalers)

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    return train_loader, val_loader, scalers
