"""Minimal configuration for the MAARS AI Framework."""

import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class DSPConfig:
    """Config for STFT computation (used by core/dsp.py for real-time inference)."""
    n_fft: int = 1024
    hop_length: int = 512
    win_length: Optional[int] = None
    center: bool = True
    sample_rate_hz: float = 25e6

    def __post_init__(self):
        if self.win_length is None:
            self.win_length = self.n_fft


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
