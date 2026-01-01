"""
MAARS AI Framework - Intelligent RF Receiver Control System

This framework provides neural network-based control for RF hardware components
including LNA, Mixer, Filter, and IF Amplifier agents.
"""

from ai_framework.config import (
    AgentConfig,
    BackboneConfig,
    DSPConfig,
    FilterConfig,
    FrameworkConfig,
    IFAmpConfig,
    LNAConfig,
    MixerConfig,
    get_logger,
)
from ai_framework.core.dsp import calculate_evm, calculate_power, compute_spectrogram
from ai_framework.models.agents import (
    BaseAgent,
    FilterAgent,
    IFAmpAgent,
    LNAAgent,
    MixerAgent,
)
from ai_framework.models.backbone import ParametricBranch, UnifiedBackbone, VisualBranch

__version__ = "0.1.0"
__all__ = [
    # DSP Functions
    "compute_spectrogram",
    "calculate_evm",
    "calculate_power",
    # Backbone
    "UnifiedBackbone",
    "VisualBranch",
    "ParametricBranch",
    # Agents
    "BaseAgent",
    "LNAAgent",
    "MixerAgent",
    "FilterAgent",
    "IFAmpAgent",
    # Config
    "BackboneConfig",
    "AgentConfig",
    "LNAConfig",
    "MixerConfig",
    "FilterConfig",
    "IFAmpConfig",
    "DSPConfig",
    "FrameworkConfig",
    "get_logger",
]
