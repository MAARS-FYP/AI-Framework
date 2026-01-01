"""
Hardware control agents for RF receiver components.

Each agent processes a latent vector from the backbone and outputs
control signals for specific hardware components (LNA, Mixer, Filter, IF Amp).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ai_framework.config import AgentConfig, get_logger

logger = get_logger(__name__)


class BaseAgent(nn.Module):
    """
    Base class for all hardware control agents.
    
    Defines the standard decision network (MLP) that processes the latent
    vector 'z' from the backbone. Child classes implement specific action
    formatting for their hardware component.
    
    Args:
        config: Agent configuration with network dimensions.
        latent_dim: Override for latent dimension (deprecated, use config).
        output_dim: Override for output dimension (deprecated, use config).
        hidden_dim: Override for hidden dimension (deprecated, use config).
    
    Input Shape:
        [batch, latent_dim]
    
    Output Shape:
        [batch, output_dim] (raw logits/activations)
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        latent_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        # Handle legacy arguments
        if config is None:
            config = AgentConfig()
        
        if latent_dim is not None:
            config.latent_dim = latent_dim
        if output_dim is not None:
            config.output_dim = output_dim
        if hidden_dim is not None:
            config.hidden_dim = hidden_dim
        
        self.config = config
        self._latent_dim = config.latent_dim
        self._output_dim = config.output_dim
        
        # Lightweight MLP for ultra-fast inference (microseconds)
        self.decision_network = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        
        logger.debug(
            f"{self.__class__.__name__} initialized: "
            f"latent_dim={config.latent_dim}, hidden_dim={config.hidden_dim}, "
            f"output_dim={config.output_dim}"
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Process latent vector through decision network.
        
        Args:
            z: Latent vector from backbone.
                Shape: [batch, latent_dim]
        
        Returns:
            Raw logits/activations.
            Shape: [batch, output_dim]
        
        Raises:
            ValueError: If input shape is invalid.
        """
        # Input validation
        if z.dim() != 2:
            raise ValueError(
                f"Expected 2D input [batch, latent_dim], got {z.dim()}D tensor"
            )
        if z.size(1) != self._latent_dim:
            raise ValueError(
                f"Expected latent_dim={self._latent_dim}, got {z.size(1)}"
            )
        
        return self.decision_network(z)

    def get_action(
        self,
        z: Tensor,
        deterministic: bool = True,
    ) -> Union[Tensor, int, float]:
        """
        Get hardware action from latent vector.
        
        Must be implemented by child classes to format output
        into actual hardware units (dB, Hz, etc.).
        
        Args:
            z: Latent vector from backbone.
            deterministic: If True, return deterministic action.
                If False, sample from action distribution (for exploration).
        
        Returns:
            Action value(s) in hardware units.
        """
        raise NotImplementedError("Child agents must define get_action()")


# --- 1. LNA Agent (Continuous Bias Current) ---
class LNAAgent(BaseAgent):
    """
    Low Noise Amplifier bias current control agent.
    
    Outputs continuous bias current value between 0 and max_current_ma.
    Uses sigmoid activation for bounded positive output.
    
    Args:
        config: Agent configuration (optional).
        latent_dim: Override for latent dimension.
        max_current_ma: Maximum bias current in mA. Default: 20.0
    
    Input Shape:
        [batch, latent_dim]
    
    Output Shape:
        [batch] - Current values in mA
    
    Example:
        >>> lna = LNAAgent(latent_dim=64, max_current_ma=20.0)
        >>> z = torch.randn(4, 64)
        >>> currents = lna.get_action(z)  # Shape: [4]
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        latent_dim: Optional[int] = None,
        max_current_ma: float = 20.0,
    ) -> None:
        if config is None:
            config = AgentConfig()
        
        if latent_dim is not None:
            config.latent_dim = latent_dim
        
        config.output_dim = 1
        super().__init__(config)
        
        self.max_ma = max_current_ma
        
        logger.info(f"LNAAgent: max_current_ma={self.max_ma}")

    def get_action(
        self,
        z: Tensor,
        deterministic: bool = True,
        as_tensor: bool = True,
        noise_scale: float = 0.1,
    ) -> Union[Tensor, List[float]]:
        """
        Get LNA bias current(s) from latent vector.
        
        Args:
            z: Latent vector. Shape: [batch, latent_dim]
            deterministic: If True, return direct output.
                If False, add exploration noise.
            as_tensor: If True, return tensor. If False, return list.
            noise_scale: Scale of exploration noise (fraction of max_ma).
        
        Returns:
            Current value(s) in mA.
            Shape: [batch] if as_tensor=True, else list of floats.
        """
        raw_val = self.forward(z)
        
        # Sigmoid bounds output to [0, 1] (Current cannot be negative)
        norm_val = torch.sigmoid(raw_val)
        
        # Scale to max hardware current
        currents = norm_val.squeeze(-1) * self.max_ma
        
        if not deterministic:
            noise = torch.randn_like(currents) * self.max_ma * noise_scale
            currents = torch.clamp(currents + noise, 0.0, self.max_ma)
        
        if as_tensor:
            return currents
        else:
            return [float(c.item()) for c in currents]


# --- 2. Mixer Agent (Dual Continuous Controls) ---
class MixerAgent(BaseAgent):
    """
    Local Oscillator (Mixer) control agent with dual outputs.
    
    Outputs both LO frequency and amplitude simultaneously.
    Frequency range: 2405 MHz to 2483 MHz (absolute, not offset).
    Amplitude uses sigmoid (positive only).
    
    Args:
        config: Agent configuration (optional).
        latent_dim: Override for latent dimension.
        min_freq_mhz: Minimum LO frequency in MHz. Default: 2405.0
        max_freq_mhz: Maximum LO frequency in MHz. Default: 2483.0
        max_amp_v: Maximum amplitude in Volts. Default: 1.0
    
    Input Shape:
        [batch, latent_dim]
    
    Output Shape:
        Tuple of ([batch], [batch]) - (frequency MHz, amplitude V)
    
    Example:
        >>> mixer = MixerAgent(latent_dim=64)
        >>> z = torch.randn(4, 64)
        >>> freq, amp = mixer.get_action(z)  # freq in MHz, amp in V
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        latent_dim: Optional[int] = None,
        min_freq_mhz: float = 2405.0,
        max_freq_mhz: float = 2483.0,
        max_amp_v: float = 1.0,
    ) -> None:
        if config is None:
            config = AgentConfig()
        
        if latent_dim is not None:
            config.latent_dim = latent_dim
        
        # Output 2 values: frequency and amplitude
        config.output_dim = 2
        super().__init__(config)
        
        self.min_freq = min_freq_mhz
        self.max_freq = max_freq_mhz
        self.max_amp = max_amp_v
        
        logger.info(
            f"MixerAgent: freq_range={self.min_freq}-{self.max_freq}MHz, "
            f"max_amp_v={self.max_amp}"
        )

    def get_action(
        self,
        z: Tensor,
        deterministic: bool = True,
        as_tensor: bool = True,
        noise_scale: float = 0.1,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[List[float], List[float]]]:
        """
        Get mixer LO frequency and amplitude from latent vector.
        
        Args:
            z: Latent vector. Shape: [batch, latent_dim]
            deterministic: If True, return direct output.
                If False, add exploration noise.
            as_tensor: If True, return tensors. If False, return lists.
            noise_scale: Scale of exploration noise.
        
        Returns:
            Tuple of (frequency_mhz, amplitude_v).
            Each has shape [batch] if as_tensor=True, else list of floats.
        """
        raw_val = self.forward(z)  # Shape: [batch, 2]
        
        # Split the two outputs
        raw_freq = raw_val[:, 0]  # Shape: [batch]
        raw_amp = raw_val[:, 1]   # Shape: [batch]
        
        # Frequency: sigmoid to [0,1], then scale to [min_freq, max_freq]
        freq_norm = torch.sigmoid(raw_freq)
        freq_mhz = self.min_freq + freq_norm * (self.max_freq - self.min_freq)
        
        # Amplitude must be positive -> Sigmoid
        amp_volts = torch.sigmoid(raw_amp) * self.max_amp
        
        if not deterministic:
            freq_range = self.max_freq - self.min_freq
            freq_noise = torch.randn_like(freq_mhz) * freq_range * noise_scale
            amp_noise = torch.randn_like(amp_volts) * self.max_amp * noise_scale
            freq_mhz = torch.clamp(freq_mhz + freq_noise, self.min_freq, self.max_freq)
            amp_volts = torch.clamp(amp_volts + amp_noise, 0.0, self.max_amp)
        
        if as_tensor:
            return freq_mhz, amp_volts
        else:
            return (
                [float(f.item()) for f in freq_mhz],
                [float(a.item()) for a in amp_volts],
            )


# --- 3. Filter Agent (Discrete Selection) ---
class FilterAgent(BaseAgent):
    """
    Hardware filter selection agent.
    
    Outputs discrete filter selection from predefined filter bandwidths.
    Uses Gumbel-Softmax for differentiable training, argmax for inference.
    
    Args:
        config: Agent configuration (optional).
        latent_dim: Override for latent dimension.
        filter_bandwidths_mhz: Tuple of available bandwidths in MHz.
        filter_names: Human-readable names for each filter.
        temperature: Temperature for Gumbel-Softmax during training.
    
    Input Shape:
        [batch, latent_dim]
    
    Output Shape:
        [batch] - Filter indices (0, 1, 2, ...)
    
    Example:
        >>> filter_agent = FilterAgent(latent_dim=64)
        >>> z = torch.randn(4, 64)
        >>> filter_ids = filter_agent.get_action(z)  # Shape: [4]
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        latent_dim: Optional[int] = None,
        filter_bandwidths_mhz: Optional[Tuple[float, ...]] = None,
        filter_names: Optional[Tuple[str, ...]] = None,
        temperature: float = 1.0,
    ) -> None:
        if config is None:
            config = AgentConfig()
        
        if latent_dim is not None:
            config.latent_dim = latent_dim
        
        if filter_bandwidths_mhz is None:
            filter_bandwidths_mhz = (5.0, 10.0, 20.0)
        
        if filter_names is None:
            filter_names = tuple(f"{bw}MHz" for bw in filter_bandwidths_mhz)
        
        # Output dim = number of filter choices
        config.output_dim = len(filter_bandwidths_mhz)
        super().__init__(config)
        
        self.filter_bandwidths_mhz = filter_bandwidths_mhz
        self.filter_names = filter_names
        self.temperature = temperature
        
        # For convenience: map index to bandwidth
        self.bandwidth_map = {i: bw for i, bw in enumerate(filter_bandwidths_mhz)}
        
        logger.info(f"FilterAgent: bandwidths={self.filter_bandwidths_mhz}MHz")

    def get_action(
        self,
        z: Tensor,
        deterministic: bool = True,
        as_tensor: bool = True,
        return_names: bool = False,
    ) -> Union[Tensor, List[int], List[str]]:
        """
        Get filter selection(s) from latent vector.
        
        Args:
            z: Latent vector. Shape: [batch, latent_dim]
            deterministic: If True, use argmax. If False, sample from softmax.
            as_tensor: If True, return tensor. If False, return list.
            return_names: If True, return filter names instead of indices.
        
        Returns:
            Filter index/indices or names.
            Shape: [batch] if as_tensor=True, else list.
        """
        logits = self.forward(z)
        
        if deterministic:
            indices = torch.argmax(logits, dim=1)
        else:
            # Sample from categorical distribution for exploration
            probs = F.softmax(logits / self.temperature, dim=1)
            indices = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        if return_names:
            return [self.filter_names[i.item()] for i in indices]
        
        if as_tensor:
            return indices
        else:
            return [int(i.item()) for i in indices]

    def get_action_differentiable(self, z: Tensor) -> Tensor:
        """
        Get differentiable filter selection using Gumbel-Softmax.
        
        Enables gradient flow through discrete selection during training.
        
        Args:
            z: Latent vector. Shape: [batch, latent_dim]
        
        Returns:
            Soft selection probabilities.
            Shape: [batch, num_filters]
        """
        logits = self.forward(z)
        return F.gumbel_softmax(logits, tau=self.temperature, hard=False)


# --- 4. IF Amplifier Agent (Continuous Gain) ---
class IFAmpAgent(BaseAgent):
    """
    IF Amplifier gain control agent.
    
    Outputs continuous gain/voltage value between 0 and max_gain_v.
    Uses sigmoid activation for bounded positive output.
    
    Args:
        config: Agent configuration (optional).
        latent_dim: Override for latent dimension.
        max_gain_v: Maximum gain voltage. Default: 3.3
    
    Input Shape:
        [batch, latent_dim]
    
    Output Shape:
        [batch] - Gain values in Volts
    
    Example:
        >>> if_amp = IFAmpAgent(latent_dim=64, max_gain_v=3.3)
        >>> z = torch.randn(4, 64)
        >>> gains = if_amp.get_action(z)  # Shape: [4]
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        latent_dim: Optional[int] = None,
        max_gain_v: float = 3.3,
    ) -> None:
        if config is None:
            config = AgentConfig()
        
        if latent_dim is not None:
            config.latent_dim = latent_dim
        
        config.output_dim = 1
        super().__init__(config)
        
        self.max_gain = max_gain_v
        
        logger.info(f"IFAmpAgent: max_gain_v={self.max_gain}")

    def get_action(
        self,
        z: Tensor,
        deterministic: bool = True,
        as_tensor: bool = True,
        noise_scale: float = 0.1,
    ) -> Union[Tensor, List[float]]:
        """
        Get IF amplifier gain(s) from latent vector.
        
        Args:
            z: Latent vector. Shape: [batch, latent_dim]
            deterministic: If True, return direct output.
                If False, add exploration noise.
            as_tensor: If True, return tensor. If False, return list.
            noise_scale: Scale of exploration noise (fraction of max_gain).
        
        Returns:
            Gain value(s) in mA.
            Shape: [batch] if as_tensor=True, else list of floats.
        """
        raw_val = self.forward(z)
        
        # Sigmoid bounds output to [0, 1] (Gain is positive)
        norm_val = torch.sigmoid(raw_val)
        
        # Scale to max gain
        gains = norm_val.squeeze(-1) * self.max_gain
        
        if not deterministic:
            noise = torch.randn_like(gains) * self.max_gain * noise_scale
            gains = torch.clamp(gains + noise, 0.0, self.max_gain)
        
        if as_tensor:
            return gains
        else:
            return [float(g.item()) for g in gains]