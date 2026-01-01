"""
Neural network backbone for multi-modal RF signal feature extraction.

This module implements a dual-branch architecture that fuses visual features
(from spectrograms) with parametric features (from sensor metrics) into a
unified latent representation.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ai_framework.config import BackboneConfig, get_logger

logger = get_logger(__name__)


class VisualBranch(nn.Module):
    """
    Convolutional branch for processing spectrogram images.
    
    Extracts spatial features from 2-channel (Real/Imaginary) spectrograms
    using a lightweight CNN architecture.
    
    Args:
        output_dim: Dimension of the output feature vector. Default: 32
        conv_channels: Tuple of channel sizes for conv layers. Default: (16, 32)
        adaptive_pool_size: Output size for adaptive pooling. Default: (4, 4)
    
    Input Shape:
        [batch, 2, freq_bins, time_frames]
        - Channel 0: Real component of spectrogram
        - Channel 1: Imaginary component of spectrogram
    
    Output Shape:
        [batch, output_dim]
    """
    
    def __init__(
        self,
        output_dim: int = 32,
        conv_channels: Tuple[int, int] = (16, 32),
        adaptive_pool_size: Tuple[int, int] = (4, 1),  # (4, 1) for narrow time dimension
    ) -> None:
        super().__init__()
        
        self.output_dim = output_dim
        self.conv_channels = conv_channels
        self.adaptive_pool_size = adaptive_pool_size
        
        # Input is a 2-channel image (Real/Imag Spectrogram)
        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=conv_channels[0],
            kernel_size=3,
            padding=1,
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=3,
            padding=1,
        )
        self.global_pool = nn.AdaptiveAvgPool2d(adaptive_pool_size)
        
        fc_input_dim = conv_channels[1] * adaptive_pool_size[0] * adaptive_pool_size[1]
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
        logger.debug(
            f"VisualBranch initialized: conv_channels={conv_channels}, "
            f"pool_size={adaptive_pool_size}, output_dim={output_dim}"
        )

    def forward(self, spectrogram: Tensor) -> Tensor:
        """
        Process spectrogram through convolutional layers.
        
        Args:
            spectrogram: 2-channel spectrogram tensor.
                Shape: [batch, 2, freq_bins, time_frames]
        
        Returns:
            Feature vector.
            Shape: [batch, output_dim]
        
        Raises:
            ValueError: If input shape is invalid.
        """
        # Input validation
        if spectrogram.dim() != 4:
            raise ValueError(
                f"Expected 4D input [batch, channels, freq, time], "
                f"got {spectrogram.dim()}D tensor"
            )
        if spectrogram.size(1) != 2:
            raise ValueError(
                f"Expected 2 channels (Real/Imag), got {spectrogram.size(1)}"
            )
        
        x = self.pool(F.relu(self.conv1(spectrogram)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return F.relu(self.fc(x))


class ParametricBranch(nn.Module):
    """
    MLP branch for processing scalar sensor metrics.
    
    Processes numerical features like EVM, power levels, etc.
    through a simple feedforward network.
    
    Args:
        input_dim: Number of input sensor metrics. Default: 3
        output_dim: Dimension of output feature vector. Default: 32
        hidden_dims: Hidden layer dimensions. Default: (16, 32)
    
    Input Shape:
        [batch, input_dim]
        Example metrics: [EVM, P_LNA, P_IF]
    
    Output Shape:
        [batch, output_dim]
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (16, 32),
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP dynamically
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.ReLU(),
        ])
        
        self.net = nn.Sequential(*layers)
        
        logger.debug(
            f"ParametricBranch initialized: input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}, output_dim={output_dim}"
        )

    def forward(self, sensor_metrics: Tensor) -> Tensor:
        """
        Process sensor metrics through MLP.
        
        Args:
            sensor_metrics: Scalar sensor measurements.
                Shape: [batch, input_dim]
        
        Returns:
            Feature vector.
            Shape: [batch, output_dim]
        
        Raises:
            ValueError: If input dimension doesn't match expected.
        """
        # Input validation
        if sensor_metrics.dim() != 2:
            raise ValueError(
                f"Expected 2D input [batch, features], "
                f"got {sensor_metrics.dim()}D tensor"
            )
        if sensor_metrics.size(1) != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {sensor_metrics.size(1)}. "
                f"Consider updating param_input_dim in BackboneConfig."
            )
        
        return self.net(sensor_metrics)


class UnifiedBackbone(nn.Module):
    """
    Dual-branch feature extractor for RF signal analysis.
    
    Fuses visual features (from spectrograms) with parametric features
    (from sensor metrics) into a unified latent vector suitable for
    downstream agent decision-making.
    
    Args:
        config: BackboneConfig with architecture parameters.
        latent_dim: Override for latent dimension (deprecated, use config).
        param_input_dim: Override for parametric input dim (deprecated, use config).
    
    Input:
        spectrogram: Shape [batch, 2, freq_bins, time_frames]
        sensor_metrics: Shape [batch, param_input_dim]
    
    Output:
        Latent vector z: Shape [batch, latent_dim]
    
    Example:
        >>> config = BackboneConfig(latent_dim=64, param_input_dim=3)
        >>> backbone = UnifiedBackbone(config)
        >>> spec = torch.randn(4, 2, 33, 13)
        >>> metrics = torch.randn(4, 3)
        >>> z = backbone(spec, metrics)
        >>> z.shape
        torch.Size([4, 64])
    """
    
    def __init__(
        self,
        config: Optional[BackboneConfig] = None,
        latent_dim: Optional[int] = None,
        param_input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        # Handle legacy arguments
        if config is None:
            config = BackboneConfig()
        
        if latent_dim is not None:
            logger.warning(
                "latent_dim argument is deprecated, use BackboneConfig instead"
            )
            config.latent_dim = latent_dim
        
        if param_input_dim is not None:
            logger.warning(
                "param_input_dim argument is deprecated, use BackboneConfig instead"
            )
            config.param_input_dim = param_input_dim
        
        self.config = config
        
        self.visual_branch = VisualBranch(
            output_dim=config.visual_output_dim,
            conv_channels=config.conv_channels,
            adaptive_pool_size=config.adaptive_pool_size,
        )
        self.param_branch = ParametricBranch(
            input_dim=config.param_input_dim,
            output_dim=config.param_output_dim,
        )
        
        fusion_input_dim = config.visual_output_dim + config.param_output_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.Tanh(),
        )
        
        logger.info(
            f"UnifiedBackbone initialized: latent_dim={config.latent_dim}, "
            f"param_input_dim={config.param_input_dim}"
        )

    def forward(
        self,
        spectrogram: Tensor,
        sensor_metrics: Tensor,
    ) -> Tensor:
        """
        Fuse spectrogram and sensor features into latent vector.
        
        Args:
            spectrogram: 2-channel spectrogram from DSP.
                Shape: [batch, 2, freq_bins, time_frames]
            sensor_metrics: Scalar measurements from sensors.
                Shape: [batch, param_input_dim]
        
        Returns:
            Latent vector z for downstream agents.
            Shape: [batch, latent_dim]
        """
        visual_feats = self.visual_branch(spectrogram)
        param_feats = self.param_branch(sensor_metrics)
        
        combined = torch.cat((visual_feats, param_feats), dim=1)
        return self.fusion_layer(combined)
    
    @property
    def latent_dim(self) -> int:
        """Return the latent dimension for this backbone."""
        return self.config.latent_dim
    
    @property
    def device(self) -> torch.device:
        """Return the device this model is on."""
        return next(self.parameters()).device
