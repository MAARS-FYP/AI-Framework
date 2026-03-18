from dataclasses import dataclass
from typing import Tuple


@dataclass
class InferenceConfig:
    n_fft: int = 1024
    hop_length: int = 512
    win_length: int = 1024
    center: bool = True
    sample_rate_hz: float = 25e6

    threshold_db: float = 3.0
    boundary_low_mhz: float = 12.0
    boundary_high_mhz: float = 25.0
    center_freqs_mhz: Tuple[int, int, int] = (2405, 2420, 2435)
    edge_margin_bins: int = 5
    center_tolerance_bins: float = 8.0
    energy_floor_db: float = -90.0
    min_span_bins: int = 2
    allow_center_shift: bool = False

    modulation: str = "QPSK"
    evm_mode: str = "blind"
