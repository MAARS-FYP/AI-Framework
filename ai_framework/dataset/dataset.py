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


class RFControlDataset(Dataset):
    """
    Dataset for RF receiver control training.
    
    Loads pre-computed STFT spectrograms and associated metrics/targets
    for training the backbone + multi-agent system.
    
    Args:
        csv_path: Path to the CSV file containing metadata and targets.
        data_root: Root directory for STFT data files. If None, uses CSV directory.
        transform: Optional transform to apply to spectrograms.
        use_complex_stft: If True, use complex STFT files. If False, use magnitude.
    
    CSV Expected Columns:
        - Input_Power_dBm: Input power metric
        - Bandwidth_Hz: Signal bandwidth  
        - Best_EVM_dB: Error Vector Magnitude
        - Resulting_NF_dB: Noise Figure
        - Measured_PA_Input_Power_dBm: PA input power
        - STFT_Complex_File: Path to complex STFT numpy file
        - Optimal_LNA_Bias_mA: Target for LNA agent
        - Optimal_Mixer_Frequency_MHz: Target for Mixer agent (frequency)
        - Optimal_Mixer_Amplitude_V: Target for Mixer agent (amplitude)
        - Detected_BW_Class: Target for Filter agent (1MHz, 10MHz, 20MHz)
        - Optimal_IF_Gain_dB: Target for IF Amp agent
    
    Returns:
        Tuple of (inputs, targets) where:
        - inputs: (spectrogram, metrics) tuple
        - targets: dict with keys 'lna', 'mixer_freq', 'mixer_amp', 'filter', 'if_amp'
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        transform: Optional[callable] = None,
        use_complex_stft: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root) if data_root else self.csv_path.parent
        self.transform = transform
        self.use_complex_stft = use_complex_stft
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded dataset with {len(self.df)} samples from {self.csv_path}")
        
        # Validate required columns
        self._validate_columns()
        
        # Pre-compute filter class indices
        self._prepare_filter_targets()
    
    def _validate_columns(self) -> None:
        """Validate that required columns exist."""
        required_cols = [
            'Input_Power_dBm', 'Bandwidth_Hz', 'Best_EVM_dB',
            'Optimal_LNA_Bias_mA', 'Optimal_IF_Gain_dB',
            'Optimal_Mixer_Frequency_MHz', 'Optimal_Mixer_Amplitude_V',
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
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], Dict[str, Tensor]]:
        row = self.df.iloc[idx]
        
        # --- 1. Load Spectrogram (Visual Input) ---
        stft_file = self.data_root / row['STFT_Complex_File']
        stft_data = np.load(stft_file)  # Shape: (1024, 8), complex64
        
        # Convert complex STFT to 2-channel (real, imag) format for CNN
        # Shape: (2, freq_bins, time_frames)
        spectrogram = np.stack([stft_data.real, stft_data.imag], axis=0)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        # --- 2. Prepare Sensor Metrics (Parametric Input) ---
        # Metrics fed to backbone's parametric branch
        # Using: EVM, Input Power (as proxy for P_LNA), PA Input Power (as proxy for P_IF)
        metrics = torch.tensor([
            row['Best_EVM_dB'],
            row['Input_Power_dBm'],
            row['Measured_PA_Input_Power_dBm'],
        ], dtype=torch.float32)
        
        # --- 3. Prepare Target Labels for Each Agent ---
        targets = {
            'lna': torch.tensor(row['Optimal_LNA_Bias_mA'], dtype=torch.float32),
            'mixer_freq': torch.tensor(row['Optimal_Mixer_Frequency_MHz'], dtype=torch.float32),
            'mixer_amp': torch.tensor(row['Optimal_Mixer_Amplitude_V'], dtype=torch.float32),
            'filter': torch.tensor(row['filter_class_idx'], dtype=torch.long),
            'if_amp': torch.tensor(row['Optimal_IF_Gain_dB'], dtype=torch.float32),
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
                'lna_ma': row['Optimal_LNA_Bias_mA'],
                'mixer_freq_mhz': row['Optimal_Mixer_Frequency_MHz'],
                'mixer_amp_v': row['Optimal_Mixer_Amplitude_V'],
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
        'mixer_freq': torch.stack([item[1]['mixer_freq'] for item in batch]),
        'mixer_amp': torch.stack([item[1]['mixer_amp'] for item in batch]),
        'filter': torch.stack([item[1]['filter'] for item in batch]),
        'if_amp': torch.stack([item[1]['if_amp'] for item in batch]),
    }
    
    return (spectrograms, metrics), targets
