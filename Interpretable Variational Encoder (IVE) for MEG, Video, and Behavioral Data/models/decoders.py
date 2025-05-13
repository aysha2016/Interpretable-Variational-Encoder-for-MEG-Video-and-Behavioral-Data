import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MEGDecoder(nn.Module):
    """
    Decoder for MEG data reconstruction.
    Converts latent representations back to MEG signal space.
    """
    def __init__(
        self,
        latent_dim: int,
        output_shape: Tuple[int, int, int, int],  # (batch, channels, time, sensors)
        base_channels: int = 32,
        dropout_rate: float = 0.1
    ):
        super(MEGDecoder, self).__init__()
        
        # Store shapes
        self.output_shape = output_shape
        _, out_channels, time_steps, n_sensors = output_shape
        
        # Initial projection
        self.initial_projection = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Temporal upsampling
        self.temporal_upsample = nn.Sequential(
            nn.ConvTranspose1d(1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.ConvTranspose1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Final projection to MEG space
        self.final_projection = nn.Sequential(
            nn.Linear(base_channels * 2, n_sensors * out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_sensors * out_channels, n_sensors * out_channels)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        
        # Initial projection
        x = self.initial_projection(z)
        
        # Reshape for temporal upsampling
        x = x.view(batch_size, 1, -1)
        x = self.temporal_upsample(x)
        
        # Final projection to MEG space
        x = x.permute(0, 2, 1)  # (batch, time, features)
        x = self.final_projection(x)
        
        # Reshape to output shape
        x = x.view(batch_size, *self.output_shape[1:])
        
        return x

class VideoDecoder(nn.Module):
    """
    Decoder for video data reconstruction.
    Converts latent representations back to video space.
    """
    def __init__(
        self,
        latent_dim: int,
        output_shape: Tuple[int, int, int, int, int],  # (batch, channels, time, height, width)
        base_channels: int = 32,
        dropout_rate: float = 0.1
    ):
        super(VideoDecoder, self).__init__()
        
        # Store shapes
        self.output_shape = output_shape
        _, out_channels, time_steps, height, width = output_shape
        
        # Initial projection
        self.initial_projection = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate initial spatial dimensions
        self.initial_height = height // 8
        self.initial_width = width // 8
        self.initial_channels = base_channels * 8
        
        # 3D upsampling layers
        self.upsample = nn.Sequential(
            # First upsampling block
            nn.ConvTranspose3d(1, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(),
            nn.Dropout3d(dropout_rate),
            
            # Second upsampling block
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(),
            nn.Dropout3d(dropout_rate),
            
            # Third upsampling block
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.Dropout3d(dropout_rate),
            
            # Final convolution
            nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        
        # Initial projection
        x = self.initial_projection(z)
        
        # Reshape for 3D upsampling
        x = x.view(batch_size, 1, self.initial_channels, self.initial_height, self.initial_width)
        x = self.upsample(x)
        
        return x

class BehavioralDecoder(nn.Module):
    """
    Decoder for behavioral data reconstruction.
    Converts latent representations back to behavioral feature space.
    """
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 512),
        dropout_rate: float = 0.1
    ):
        super(BehavioralDecoder, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z) 