from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class ReducedHardwareConfig:
    sample_rate_hz: float = 125e6
    n_fft: int = 16384
    use_log_magnitude: bool = True
    normalize_feature: bool = True
    center_frequencies_mhz: Tuple[int, int, int] = (2405, 2420, 2435)
    bandwidths_mhz: Tuple[int, int, int] = (1, 10, 20)
    center_frequency_class_to_mhz: Dict[int, int] = field(
        default_factory=lambda: {0: 2405, 1: 2420, 2: 2435}
    )
    bandwidth_class_to_mhz: Dict[int, int] = field(default_factory=lambda: {0: 1, 1: 10, 2: 20})
