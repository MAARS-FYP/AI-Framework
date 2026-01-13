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


# --- 1. LNA Agent (Binary Classification: 3V or 5V) ---
class LNAAgent(BaseAgent):
    """
    Low Noise Amplifier voltage control agent.
    
    Binary classification for voltage selection: 3V or 5V.
    Uses softmax activation for discrete selection.
    
    Args:
        config: Agent configuration (optional).
        latent_dim: Override for latent dimension.
        voltage_levels: Tuple of available voltages (default: 3V, 5V).
        voltage_names: Human-readable names for each level.
    
    Input Shape:
        [batch, latent_dim]
    
    Output Shape:
        [batch] - Voltage class indices (0=3V, 1=5V)
    
    Example:
        >>> lna = LNAAgent(latent_dim=64)
        >>> z = torch.randn(4, 64)
        >>> voltage_idx = lna.get_action(z)  # Shape: [4], values 0 or 1
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        latent_dim: Optional[int] = None,
        voltage_levels: Tuple[float, ...] = (3.0, 5.0),
        voltage_names: Tuple[str, ...] = ("3V", "5V"),
    ) -> None:
        if config is None:
            config = AgentConfig()
        
        if latent_dim is not None:
            config.latent_dim = latent_dim
        
        # Output dim = number of voltage choices (binary classification)
        config.output_dim = len(voltage_levels)
        super().__init__(config)
        
        self.voltage_levels = voltage_levels
        self.voltage_names = voltage_names
        
        # For convenience: map index to voltage
        self.voltage_map = {i: v for i, v in enumerate(voltage_levels)}
        
        logger.info(f"LNAAgent: voltage_levels={self.voltage_levels}V (binary classification)")

    def get_action(
        self,
        z: Tensor,
        deterministic: bool = True,
        as_tensor: bool = True,
        return_voltage: bool = False,
    ) -> Union[Tensor, List[int], List[float]]:
        """
        Get LNA voltage selection(s) from latent vector.
        
        Args:
            z: Latent vector. Shape: [batch, latent_dim]
            deterministic: If True, use argmax. If False, sample from softmax.
            as_tensor: If True, return tensor. If False, return list.
            return_voltage: If True, return voltage values instead of indices.
        
        Returns:
            Voltage index/indices or voltage values.
            Shape: [batch] if as_tensor=True, else list.
        """
        logits = self.forward(z)  # [batch, 2]
        
        if deterministic:
            indices = torch.argmax(logits, dim=1)
        else:
            # Sample from categorical distribution for exploration
            probs = F.softmax(logits, dim=1)
            indices = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        if return_voltage:
            voltages = torch.tensor(
                [self.voltage_levels[i.item()] for i in indices],
                device=z.device
            )
            if as_tensor:
                return voltages
            else:
                return [float(v.item()) for v in voltages]
        
        if as_tensor:
            return indices
        else:
            return [int(i.item()) for i in indices]
    
    def get_voltage_from_index(self, idx: int) -> float:
        """Convert class index to voltage value."""
        return self.voltage_levels[idx]


# --- 2. Filter Agent (Discrete Selection) ---
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
            filter_bandwidths_mhz = (1.0, 10.0, 20.0)
        
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


# --- 3. Mixer Agent (LO Frequency + Attenuation) ---
class MixerAgent(BaseAgent):
    """
    Mixer / Local Oscillator control agent.
    
    Outputs two values:
    1. LO frequency in MHz
    2. Attenuation in dB (0 to -26dB)
    
    Args:
        config: Agent configuration (optional).
        latent_dim: Override for latent dimension.
        min_freq_mhz: Minimum LO frequency. Default: 2405.0
        max_freq_mhz: Maximum LO frequency. Default: 2483.0
        min_atten_db: Minimum (most) attenuation. Default: -26.0
        max_atten_db: Maximum (no) attenuation. Default: 0.0
    
    Input Shape:
        [batch, latent_dim]
    
    Output Shape:
        Tuple of ([batch], [batch]) - (frequency_MHz, attenuation_dB)
    
    Example:
        >>> mixer = MixerAgent(latent_dim=64)
        >>> z = torch.randn(4, 64)
        >>> freq, atten = mixer.get_action(z)
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        latent_dim: Optional[int] = None,
        min_freq_mhz: float = 2405.0,
        max_freq_mhz: float = 2483.0,
        min_atten_db: float = -26.0,
        max_atten_db: float = 0.0,
    ) -> None:
        if config is None:
            config = AgentConfig()
        
        if latent_dim is not None:
            config.latent_dim = latent_dim
        
        config.output_dim = 2  # frequency + attenuation
        super().__init__(config)
        
        self.min_freq = min_freq_mhz
        self.max_freq = max_freq_mhz
        self.min_atten = min_atten_db
        self.max_atten = max_atten_db
        
        logger.info(
            f"MixerAgent: freq_range={self.min_freq}-{self.max_freq}MHz, "
            f"atten_range={self.min_atten} to {self.max_atten}dB"
        )

    def get_action(
        self,
        z: Tensor,
        deterministic: bool = True,
        as_tensor: bool = True,
        noise_scale: float = 0.1,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[List[float], List[float]]]:
        """
        Get mixer LO frequency and attenuation from latent vector.
        
        Args:
            z: Latent vector. Shape: [batch, latent_dim]
            deterministic: If True, return direct output.
                If False, add exploration noise.
            as_tensor: If True, return tensors. If False, return lists.
            noise_scale: Scale of exploration noise.
        
        Returns:
            Tuple of (frequency_MHz, attenuation_dB).
            Shape: ([batch], [batch]) if as_tensor=True.
        """
        raw_val = self.forward(z)  # Shape: [batch, 2]
        
        # Split into frequency and attenuation outputs
        raw_freq = raw_val[:, 0]
        raw_atten = raw_val[:, 1]
        
        # Sigmoid bounds outputs to [0, 1], then scale to respective ranges
        norm_freq = torch.sigmoid(raw_freq)
        norm_atten = torch.sigmoid(raw_atten)
        
        frequencies = self.min_freq + norm_freq * (self.max_freq - self.min_freq)
        attenuations = self.min_atten + norm_atten * (self.max_atten - self.min_atten)
        
        if not deterministic:
            freq_range = self.max_freq - self.min_freq
            atten_range = self.max_atten - self.min_atten
            
            freq_noise = torch.randn_like(frequencies) * freq_range * noise_scale
            atten_noise = torch.randn_like(attenuations) * atten_range * noise_scale
            
            frequencies = torch.clamp(frequencies + freq_noise, self.min_freq, self.max_freq)
            attenuations = torch.clamp(attenuations + atten_noise, self.min_atten, self.max_atten)
        
        if as_tensor:
            return frequencies, attenuations
        else:
            return (
                [float(f.item()) for f in frequencies],
                [float(a.item()) for a in attenuations],
            )


# --- 4. IF Amplifier Agent (Continuous Gain in dB) ---
class IFAmpAgent(BaseAgent):
    """
    IF Amplifier gain control agent.
    
    Outputs continuous gain value in dB between min_gain_db and max_gain_db.
    Uses sigmoid activation scaled to dB range.
    
    Args:
        config: Agent configuration (optional).
        latent_dim: Override for latent dimension.
        min_gain_db: Minimum gain in dB. Default: -6.0
        max_gain_db: Maximum gain in dB. Default: 26.0
    
    Input Shape:
        [batch, latent_dim]
    
    Output Shape:
        [batch] - Gain values in dB
    
    Example:
        >>> if_amp = IFAmpAgent(latent_dim=64)
        >>> z = torch.randn(4, 64)
        >>> gains = if_amp.get_action(z)  # Shape: [4], values in dB
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        latent_dim: Optional[int] = None,
        min_gain_db: float = -6.0,
        max_gain_db: float = 26.0,
    ) -> None:
        if config is None:
            config = AgentConfig()
        
        if latent_dim is not None:
            config.latent_dim = latent_dim
        
        config.output_dim = 1
        super().__init__(config)
        
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        
        logger.info(f"IFAmpAgent: gain_range={self.min_gain_db} to {self.max_gain_db} dB")

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
            noise_scale: Scale of exploration noise.
        
        Returns:
            Gain value(s) in dB.
            Shape: [batch] if as_tensor=True, else list of floats.
        """
        raw_val = self.forward(z)
        
        # Sigmoid bounds output to [0, 1], then scale to [min_gain_db, max_gain_db]
        norm_val = torch.sigmoid(raw_val)
        gains_db = self.min_gain_db + norm_val.squeeze(-1) * (self.max_gain_db - self.min_gain_db)
        
        if not deterministic:
            gain_range = self.max_gain_db - self.min_gain_db
            noise = torch.randn_like(gains_db) * gain_range * noise_scale
            gains_db = torch.clamp(gains_db + noise, self.min_gain_db, self.max_gain_db)
        
        if as_tensor:
            return gains_db
        else:
            return [float(g.item()) for g in gains_db]