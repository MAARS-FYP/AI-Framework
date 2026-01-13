"""
Configuration dataclasses for the MAARS AI Framework.

Provides centralized, type-safe configuration management for all components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging


@dataclass
class DSPConfig:
    """Configuration for Digital Signal Processing operations.
    
    Attributes:
        n_fft: Number of FFT points for spectrogram computation.
        hop_length: Hop length between successive frames.
        win_length: Window length for STFT. Defaults to n_fft if None.
        center: Whether to pad signal to center frames.
    """
    n_fft: int = 64
    hop_length: int = 16
    win_length: Optional[int] = None
    center: bool = False
    
    def __post_init__(self) -> None:
        if self.win_length is None:
            self.win_length = self.n_fft


@dataclass
class BackboneConfig:
    """Configuration for the UnifiedBackbone network.
    
    Attributes:
        latent_dim: Dimension of the output latent vector.
        visual_output_dim: Output dimension of the visual branch.
        param_output_dim: Output dimension of the parametric branch.
        param_input_dim: Number of input sensor metrics (e.g., EVM, P_LNA, P_IF).
        conv_channels: List of channel sizes for conv layers in visual branch.
        adaptive_pool_size: Output size for adaptive average pooling.
    """
    latent_dim: int = 64
    visual_output_dim: int = 32
    param_output_dim: int = 32
    param_input_dim: int = 3
    conv_channels: Tuple[int, int] = (16, 32)
    adaptive_pool_size: Tuple[int, int] = (4, 1)  # (4, 1) for narrow time dimension inputs


@dataclass
class AgentConfig:
    """Base configuration for hardware control agents.
    
    Attributes:
        latent_dim: Dimension of input latent vector from backbone.
        hidden_dim: Dimension of hidden layers in decision network.
        output_dim: Dimension of output (action space).
    """
    latent_dim: int = 64
    hidden_dim: int = 32
    output_dim: int = 1


@dataclass
class LNAConfig(AgentConfig):
    """Configuration for the LNA (Low Noise Amplifier) Agent.
    
    Controls supply voltage for the LNA.
    Binary classification: 3V or 5V only.
    
    Attributes:
        voltage_levels: Available voltage levels (3V, 5V).
        voltage_names: Human-readable names for each level.
    """
    voltage_levels: Tuple[float, ...] = (3.0, 5.0)
    voltage_names: Tuple[str, ...] = ("3V", "5V")
    
    def __post_init__(self) -> None:
        # Output dim = number of voltage choices (binary classification)
        self.output_dim = len(self.voltage_levels)


@dataclass
class FilterConfig(AgentConfig):
    """Configuration for the Filter (Bandwidth Selection) Agent.
    
    Discrete selection of hardware filters.
    
    Attributes:
        filter_bandwidths_mhz: Available filter bandwidths in MHz.
        filter_names: Human-readable names for each filter.
        temperature: Temperature for Gumbel-Softmax during training.
    """
    filter_bandwidths_mhz: Tuple[float, ...] = (1.0, 10.0, 20.0)
    filter_names: Tuple[str, ...] = ("1MHz", "10MHz", "20MHz")
    temperature: float = 1.0
    
    def __post_init__(self) -> None:
        # Output dim = number of filter choices
        self.output_dim = len(self.filter_bandwidths_mhz)
    
    @property
    def bandwidth_map(self) -> Dict[int, float]:
        """Returns mapping from index to bandwidth in MHz."""
        return {i: bw for i, bw in enumerate(self.filter_bandwidths_mhz)}


@dataclass
class MixerConfig(AgentConfig):
    """Configuration for the Mixer (Local Oscillator) Agent.
    
    Controls LO frequency and attenuation.
    
    Attributes:
        min_freq_mhz: Minimum LO frequency in MHz.
        max_freq_mhz: Maximum LO frequency in MHz.
        min_atten_db: Minimum attenuation in dB (most attenuation).
        max_atten_db: Maximum attenuation in dB (no attenuation).
    """
    output_dim: int = 2  # frequency + attenuation
    min_freq_mhz: float = 2405.0
    max_freq_mhz: float = 2483.0
    min_atten_db: float = -26.0  # Maximum attenuation
    max_atten_db: float = 0.0    # No attenuation


@dataclass
class IFAmpConfig(AgentConfig):
    """Configuration for the IF Amplifier Agent.
    
    Controls IF amplifier gain in dB.
    
    Attributes:
        min_gain_db: Minimum gain in dB.
        max_gain_db: Maximum gain in dB.
    """
    output_dim: int = 1
    min_gain_db: float = -6.0
    max_gain_db: float = 26.0


@dataclass
class FrameworkConfig:
    """Master configuration containing all sub-configurations.
    
    Attributes:
        dsp: DSP configuration.
        backbone: Backbone network configuration.
        lna: LNA agent configuration.
        mixer: Mixer agent configuration.
        filter: Filter agent configuration.
        if_amp: IF amplifier agent configuration.
        device: PyTorch device string ('cuda', 'cpu', 'mps').
        log_level: Logging level.
    """
    dsp: DSPConfig = field(default_factory=DSPConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    lna: LNAConfig = field(default_factory=LNAConfig)
    mixer: MixerConfig = field(default_factory=MixerConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    if_amp: IFAmpConfig = field(default_factory=IFAmpConfig)
    device: str = "cpu"
    log_level: int = logging.INFO


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger instance.
    
    Args:
        name: Name for the logger (typically __name__).
        level: Logging level (default: INFO).
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger
