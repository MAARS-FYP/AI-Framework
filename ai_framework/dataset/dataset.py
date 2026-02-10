"""
PyTorch Dataset for RF Signal Control Training.

Loads STFT spectrograms and sensor metrics, along with optimal control targets
for multi-agent training (LNA, Mixer, Filter, IF Amp).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ai_framework.config import get_logger

logger = get_logger(__name__)


# Filter bandwidth class mapping
# Maps Detected_BW_Class labels to discrete filter indices
# Available filters: 1MHz (narrow), 10MHz (medium), 20MHz (wide)
FILTER_CLASS_MAP = {
    "1MHz": 0,
    "10MHz": 1,
    "20MHz": 2,
    # Alternative numeric-only labels (for robustness)
    "1": 0,
    "10": 1,
    "20": 2,
}

# LNA voltage class mapping
# Binary classification: 3V (low power) or 5V (high power)
LNA_VOLTAGE_CLASS_MAP = {
    3.0: 0,  # 3V -> class 0
    5.0: 1,  # 5V -> class 1
}
LNA_VOLTAGE_LEVELS = (3.0, 5.0)


def normalize_spectrogram(stft_data: np.ndarray) -> np.ndarray:
    """
    Normalize complex STFT spectrogram using per-sample Z-score normalization.
    
    Applies Z-score normalization to real and imaginary components independently
    to standardize the input distribution for the CNN.
    
    Args:
        stft_data: Complex-valued STFT array.
            Shape: (freq_bins, time_frames), complex dtype
    
    Returns:
        Normalized 2-channel array [real, imag].
        Shape: (2, freq_bins, time_frames), float32
    
    Note:
        Per-sample normalization is used rather than dataset-wide to handle
        varying signal characteristics. Add small epsilon to prevent division by zero.
    """
    real = stft_data.real
    imag = stft_data.imag
    
    # Z-score normalize each channel independently
    real_mean, real_std = real.mean(), real.std()
    imag_mean, imag_std = imag.mean(), imag.std()
    
    real_norm = (real - real_mean) / (real_std + 1e-8)
    imag_norm = (imag - imag_mean) / (imag_std + 1e-8)
    
    return np.stack([real_norm, imag_norm], axis=0).astype(np.float32)


class DataNormalizer:
    """
    Compute and apply Z-score normalization to dataset features and targets.
    
    Stores mean and standard deviation computed from the training set,
    enabling consistent normalization across train/val/test splits.
    
    Args:
        df: DataFrame containing the dataset.
    
    Attributes:
        metrics_mean: Mean of sensor metrics [EVM, P_LNA, P_PA].
        metrics_std: Std of sensor metrics.
        if_gain_mean: Mean of IF amplifier gain target.
        if_gain_std: Std of IF amplifier gain target.
        mixer_power_mean: Mean of mixer LO power target.
        mixer_power_std: Std of mixer LO power target.
    """
    
    def __init__(self, df: pd.DataFrame) -> None:
        # Compute statistics for continuous input features (sensor metrics)
        metric_cols = ['Best_EVM_dB', 'Measured_Power_Post_LNA_dBm', 'Measured_Power_Post_PA_dBm']
        self.metrics_mean = df[metric_cols].mean().values.astype(np.float32)
        self.metrics_std = df[metric_cols].std().values.astype(np.float32)
        
        # Compute statistics for regression targets
        self.if_gain_mean = float(df['Optimal_IF_Gain_dB'].mean())
        self.if_gain_std = float(df['Optimal_IF_Gain_dB'].std())
        
        self.mixer_power_mean = float(df['Optimal_LO_Power_dBm'].mean())
        self.mixer_power_std = float(df['Optimal_LO_Power_dBm'].std())
        
        logger.info(
            f"DataNormalizer initialized:\n"
            f"  Metrics mean: {self.metrics_mean}\n"
            f"  Metrics std: {self.metrics_std}\n"
            f"  IF gain: μ={self.if_gain_mean:.2f}, σ={self.if_gain_std:.2f}\n"
            f"  Mixer power: μ={self.mixer_power_mean:.2f}, σ={self.mixer_power_std:.2f}"
        )
    
    def normalize_metrics(self, metrics: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization to sensor metrics.
        
        Args:
            metrics: Raw sensor metrics array.
        
        Returns:
            Normalized metrics: (x - μ) / σ
        """
        return (metrics - self.metrics_mean) / (self.metrics_std + 1e-8)
    
    def normalize_if_gain(self, value: float) -> float:
        """Normalize IF amplifier gain target."""
        return (value - self.if_gain_mean) / (self.if_gain_std + 1e-8)
    
    def normalize_mixer_power(self, value: float) -> float:
        """Normalize mixer LO power target."""
        return (value - self.mixer_power_mean) / (self.mixer_power_std + 1e-8)
    
    def denormalize_if_gain(self, value: Union[float, Tensor]) -> Union[float, Tensor]:
        """Convert normalized IF gain prediction back to dB."""
        return value * self.if_gain_std + self.if_gain_mean
    
    def denormalize_mixer_power(self, value: Union[float, Tensor]) -> Union[float, Tensor]:
        """Convert normalized mixer power prediction back to dBm."""
        return value * self.mixer_power_std + self.mixer_power_mean
    
    def to_dict(self) -> Dict:
        """Export normalizer parameters for saving/loading."""
        return {
            'metrics_mean': self.metrics_mean.tolist(),
            'metrics_std': self.metrics_std.tolist(),
            'if_gain_mean': self.if_gain_mean,
            'if_gain_std': self.if_gain_std,
            'mixer_power_mean': self.mixer_power_mean,
            'mixer_power_std': self.mixer_power_std,
        }
    
    @classmethod
    def from_dict(cls, params: Dict) -> 'DataNormalizer':
        """Load normalizer from saved parameters."""
        normalizer = object.__new__(cls)
        normalizer.metrics_mean = np.array(params['metrics_mean'], dtype=np.float32)
        normalizer.metrics_std = np.array(params['metrics_std'], dtype=np.float32)
        normalizer.if_gain_mean = params['if_gain_mean']
        normalizer.if_gain_std = params['if_gain_std']
        normalizer.mixer_power_mean = params['mixer_power_mean']
        normalizer.mixer_power_std = params['mixer_power_std']
        return normalizer


class RFControlDataset(Dataset):
    """
    Dataset for RF receiver control training.
    
    Loads pre-computed STFT spectrograms and associated metrics/targets
    for training the backbone + multi-agent system.
    
    Args:
        csv_path: Path to the CSV file containing metadata and targets.
        data_root: Root directory for STFT data files. If None, uses CSV directory.
        transform: Optional transform to apply to spectrograms.
    
    CSV Expected Columns:
        - Input_Power_dBm: Input power metric
        - Bandwidth_Hz: Signal bandwidth  
        - Best_EVM_dB: Error Vector Magnitude
        - Resulting_NF_dB: Noise Figure
        - Measured_Power_Post_LNA_dBm: Power after LNA
        - Measured_Power_Post_PA_dBm: Power after PA
        - STFT_Complex_File: Path to complex STFT numpy file
        - Optimal_LNA_Voltage_V: Target for LNA agent (2.7V to 5.25V)
        - Detected_BW_Class: Target for Filter agent (1MHz, 10MHz, 20MHz)
        - Optimal_IF_Gain_dB: Target for IF Amp agent (-20dB to 0dB)
    
    Returns:
        Tuple of (inputs, targets) where:
        - inputs: (spectrogram, metrics) tuple
        - targets: dict with keys 'lna', 'filter', 'if_amp'
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        transform: Optional[callable] = None,
        normalizer: Optional[DataNormalizer] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root) if data_root else self.csv_path.parent
        self.transform = transform
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded dataset with {len(self.df)} samples from {self.csv_path}")
        
        # Validate required columns
        self._validate_columns()
        
        # Pre-compute class indices for classification targets
        self._prepare_filter_targets()
        self._prepare_lna_targets()
        
        # Initialize normalization (always enabled)
        if normalizer is not None:
            self.normalizer = normalizer
            logger.info("Using provided normalizer")
        else:
            self.normalizer = DataNormalizer(self.df)
            logger.info("Created new normalizer from dataset")
    
    def _validate_columns(self) -> None:
        """Validate that required columns exist."""
        required_cols = [
            'Input_Power_dBm', 'Bandwidth_Hz', 'Best_EVM_dB',
            'Optimal_LNA_Voltage_V', 'Optimal_IF_Gain_dB',
            'Optimal_LO_Power_dBm',  # Mixer LO power
            'Detected_BW_Class', 'STFT_Complex_File'
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _prepare_filter_targets(self) -> None:
        """Convert filter class names to indices, inferring from bandwidth if needed."""
        def get_filter_class(row):
            bw_class = row.get('Detected_BW_Class', '')
            if pd.notna(bw_class) and str(bw_class).strip() in FILTER_CLASS_MAP:
                return FILTER_CLASS_MAP[str(bw_class).strip()]
            
            # Infer from Bandwidth_Hz if label is missing
            bw_hz = row.get('Bandwidth_Hz', 0)
            if bw_hz <= 1_000_000:
                return 0  # 1MHz or less -> narrow (1MHz filter)
            elif bw_hz <= 10_000_000:
                return 1  # 10MHz
            else:
                return 2  # 20MHz
        
        self.df['filter_class_idx'] = self.df.apply(get_filter_class, axis=1)
    
    def _prepare_lna_targets(self) -> None:
        """Convert LNA voltage to class index (binary: 3V=0, 5V=1)."""
        def get_lna_class(voltage):
            # Round to nearest available voltage level
            if voltage <= 4.0:  # Closer to 3V
                return 0  # 3V class
            else:  # Closer to 5V
                return 1  # 5V class
        
        self.df['lna_class_idx'] = self.df['Optimal_LNA_Voltage_V'].apply(get_lna_class)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], Dict[str, Tensor]]:
        row = self.df.iloc[idx]
        
        # --- 1. Load Spectrogram (Visual Input) ---
        stft_file = self.data_root / 'stft_complex' / row['STFT_Complex_File']
        stft_data = np.load(stft_file)  # Shape: (1024, 5), complex128
        
        # Convert complex STFT to 2-channel (real, imag) format for CNN
        # Always apply normalization
        spectrogram = normalize_spectrogram(stft_data)  # Returns (2, freq_bins, time_frames)
        
        spectrogram = torch.from_numpy(spectrogram)
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        # --- 2. Prepare Sensor Metrics (Parametric Input) ---
        # Metrics fed to backbone's parametric branch
        # Using: EVM, Power after LNA, Power after PA
        metrics = np.array([
            row['Best_EVM_dB'],
            row['Measured_Power_Post_LNA_dBm'],
            row['Measured_Power_Post_PA_dBm'],
        ], dtype=np.float32)
        
        # Always apply Z-score normalization to metrics
        metrics = self.normalizer.normalize_metrics(metrics)
        metrics = torch.from_numpy(metrics)
        
        # --- 3. Prepare Target Labels for Each Agent ---
        # Always normalize regression targets (if_amp, mixer_power)
        # Classification targets (lna, filter) remain as class indices
        if_amp_target = self.normalizer.normalize_if_gain(row['Optimal_IF_Gain_dB'])
        mixer_power_target = self.normalizer.normalize_mixer_power(row['Optimal_LO_Power_dBm'])
        
        targets = {
            'lna': torch.tensor(row['lna_class_idx'], dtype=torch.long),  # Binary class
            'mixer_power': torch.tensor(mixer_power_target, dtype=torch.float32),  # Normalized LO power
            'filter': torch.tensor(row['filter_class_idx'], dtype=torch.long),  # 3-class
            'if_amp': torch.tensor(if_amp_target, dtype=torch.float32),  # Normalized gain
        }
        
        return (spectrogram, metrics), targets
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get human-readable info about a sample (for debugging)."""
        row = self.df.iloc[idx]
        return {
            'input_power': row['Input_Power_dBm'],
            'bandwidth': row['Bandwidth_Hz'],
            'evm': row['Best_EVM_dB'],
            'targets': {
                'lna_voltage_v': row['Optimal_LNA_Voltage_V'],
                'mixer_power_dbm': row['Optimal_LO_Power_dBm'],
                'filter_class': row['Detected_BW_Class'],
                'if_gain_db': row['Optimal_IF_Gain_dB'],
            }
        }


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Stacks spectrograms and metrics, and collates target dicts.
    """
    spectrograms = torch.stack([item[0][0] for item in batch])
    metrics = torch.stack([item[0][1] for item in batch])
    
    targets = {
        'lna': torch.stack([item[1]['lna'] for item in batch]),
        'mixer_power': torch.stack([item[1]['mixer_power'] for item in batch]),
        'filter': torch.stack([item[1]['filter'] for item in batch]),
        'if_amp': torch.stack([item[1]['if_amp'] for item in batch]),
    }
    
    return (spectrograms, metrics), targets
