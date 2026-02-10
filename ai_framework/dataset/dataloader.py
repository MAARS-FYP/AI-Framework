"""
DataLoader utilities for RF Control training.

Provides functions to create train/validation data loaders with proper splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Subset, random_split

from ai_framework.config import get_logger
from ai_framework.dataset.dataset import RFControlDataset, DataNormalizer, collate_fn

logger = get_logger(__name__)


def create_dataloaders(
    csv_path: Union[str, Path],
    data_root: Optional[Union[str, Path]] = None,
    batch_size: int = 4,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders with random splitting.
    
    Normalization statistics are computed from the TRAINING split only
    to prevent data leakage into validation. Normalization is always enabled.
    
    Args:
        csv_path: Path to the CSV file with dataset metadata.
        data_root: Root directory for STFT data files.
        batch_size: Batch size for both loaders.
        val_split: Fraction of data to use for validation (0.0 to 1.0).
        num_workers: Number of worker processes for data loading.
        seed: Random seed for reproducible splitting.
        shuffle_train: Whether to shuffle training data.
    
    Returns:
        Tuple of (train_loader, val_loader).
    
    Example:
        >>> train_loader, val_loader = create_dataloaders(
        ...     csv_path="data/optimal_control_dataset.csv",
        ...     batch_size=4,
        ...     val_split=0.2
        ... )
    """
    import pandas as pd
    
    # Load CSV to compute split indices
    df = pd.read_csv(csv_path)
    total_size = len(df)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    logger.info(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Generate train/val indices
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Compute normalization statistics from TRAINING data only (prevents data leakage)
    train_df = df.iloc[train_indices]
    normalizer = DataNormalizer(train_df)
    logger.info("Computed normalization statistics from training split only")
    
    # Create datasets with shared normalizer
    full_dataset = RFControlDataset(
        csv_path=csv_path,
        data_root=data_root,
        normalizer=normalizer,
    )
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    return train_loader, val_loader


def create_inference_dataloader(
    csv_path: Union[str, Path],
    data_root: Optional[Union[str, Path]] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    normalizer: Optional[DataNormalizer] = None,
) -> DataLoader:
    """
    Create a DataLoader for inference (no shuffling, no splitting).
    
    Args:
        csv_path: Path to the CSV file.
        data_root: Root directory for STFT data files.
        batch_size: Batch size.
        num_workers: Number of worker processes.
        normalizer: Pre-computed normalizer from training. Required for 
            consistent normalization at inference time.
    
    Returns:
        DataLoader for inference.
        
    Note:
        The normalizer should be loaded from training checkpoints to ensure
        inference uses the same normalization statistics as training.
    """
    dataset = RFControlDataset(
        csv_path=csv_path,
        data_root=data_root,
        normalizer=normalizer,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    
    return loader


def get_dataset_stats(csv_path: Union[str, Path]) -> dict:
    """
    Get statistics about the dataset for normalization or analysis.
    
    Args:
        csv_path: Path to the CSV file.
    
    Returns:
        Dictionary with dataset statistics.
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    stats = {
        'num_samples': len(df),
        'input_power': {
            'min': df['Input_Power_dBm'].min(),
            'max': df['Input_Power_dBm'].max(),
            'mean': df['Input_Power_dBm'].mean(),
        },
        'lna_target': {
            'min': df['Optimal_LNA_Voltage_V'].min(),
            'max': df['Optimal_LNA_Voltage_V'].max(),
            'mean': df['Optimal_LNA_Voltage_V'].mean(),
        },
        'mixer_power_target': {
            'min': df['Optimal_LO_Power_dBm'].min(),
            'max': df['Optimal_LO_Power_dBm'].max(),
            'mean': df['Optimal_LO_Power_dBm'].mean(),
        },
        'if_gain_target': {
            'min': df['Optimal_IF_Gain_dB'].min(),
            'max': df['Optimal_IF_Gain_dB'].max(),
            'mean': df['Optimal_IF_Gain_dB'].mean(),
        },
        'filter_class_distribution': df['Detected_BW_Class'].value_counts().to_dict(),
    }
    
    return stats
