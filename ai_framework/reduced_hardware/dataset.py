from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ai_framework.reduced_hardware.config import ReducedHardwareConfig
from ai_framework.reduced_hardware.features import (
    bandwidth_to_class,
    center_frequency_to_class,
    compute_fft_feature,
    load_ila_adc_samples,
)


@dataclass(frozen=True)
class ReducedHardwareSample:
    capture_csv_path: Path
    center_class: int
    bandwidth_class: int


class ReducedHardwareManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv_path: str | Path,
        data_root: Optional[str | Path] = None,
        config: Optional[ReducedHardwareConfig] = None,
        sample_column: Optional[str] = None,
    ):
        self.manifest_csv_path = Path(manifest_csv_path)
        self.data_root = Path(data_root) if data_root is not None else self.manifest_csv_path.parent
        self.config = config or ReducedHardwareConfig()
        self.sample_column = sample_column

        df = pd.read_csv(self.manifest_csv_path)
        required_columns = {"capture_csv_path"}
        missing = required_columns - set(df.columns)
        if missing:
            raise KeyError(f"Manifest CSV is missing required columns: {sorted(missing)}")

        if "center_class" in df.columns:
            self.center_targets = df["center_class"].astype(int).to_numpy()
        elif "center_frequency_mhz" in df.columns:
            self.center_targets = df["center_frequency_mhz"].apply(center_frequency_to_class).astype(int).to_numpy()
        else:
            raise KeyError("Manifest CSV must include either center_class or center_frequency_mhz")

        if "bandwidth_class" in df.columns:
            self.bandwidth_targets = df["bandwidth_class"].astype(int).to_numpy()
        elif "bandwidth_mhz" in df.columns:
            self.bandwidth_targets = df["bandwidth_mhz"].apply(bandwidth_to_class).astype(int).to_numpy()
        else:
            raise KeyError("Manifest CSV must include either bandwidth_class or bandwidth_mhz")

        self.capture_paths = df["capture_csv_path"].astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.capture_paths)

    def __getitem__(self, idx: int):
        capture_path = Path(self.capture_paths[idx])
        if not capture_path.is_absolute():
            capture_path = self.data_root / capture_path

        adc_samples = load_ila_adc_samples(capture_path, sample_column=self.sample_column, max_samples=self.config.n_fft)
        feature, _ = compute_fft_feature(adc_samples, config=self.config)

        inputs = torch.from_numpy(feature).unsqueeze(0)
        targets = {
            "center_class": torch.tensor(int(self.center_targets[idx]), dtype=torch.long),
            "bandwidth_class": torch.tensor(int(self.bandwidth_targets[idx]), dtype=torch.long),
        }
        metadata = {"capture_path": str(capture_path)}
        return inputs, targets, metadata


def collate_fn(batch):
    inputs = torch.stack([item[0] for item in batch], dim=0)
    targets = {key: torch.stack([item[1][key] for item in batch], dim=0) for key in batch[0][1]}
    metadata = [item[2] for item in batch]
    return inputs, targets, metadata


def create_dataloaders(
    manifest_csv_path: str | Path,
    data_root: Optional[str | Path] = None,
    config: Optional[ReducedHardwareConfig] = None,
    batch_size: int = 8,
    val_split: float = 0.2,
    seed: int = 42,
    sample_column: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, ReducedHardwareManifestDataset]:
    dataset = ReducedHardwareManifestDataset(
        manifest_csv_path=manifest_csv_path,
        data_root=data_root,
        config=config,
        sample_column=sample_column,
    )
    n_items = len(dataset)
    indices = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed)).tolist()
    val_size = max(1, int(n_items * val_split)) if n_items > 1 else 0
    train_idx = indices[val_size:] if val_size < n_items else indices[:1]
    val_idx = indices[:val_size] if val_size > 0 else indices[:1]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, dataset
