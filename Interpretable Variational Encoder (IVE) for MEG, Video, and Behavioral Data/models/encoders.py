import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class MEGEncoder(nn.Module):
    """
    Encoder for MEG data using 3D CNNs and attention mechanisms.
    Processes spatiotemporal neural data to extract meaningful features.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],  # (batch, channels, time, sensors)
        output_dim: int,
        base_channels: int = 32,
        dropout_rate: float = 0.1
    ):
        super(MEGEncoder, self).__init__()
        
        # Store input shape
        self.input_shape = input_shape
        _, in_channels, time_steps, n_sensors = input_shape
        
        # Temporal convolution layers
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Linear(n_sensors, n_sensors // 2),
            nn.ReLU(),
            nn.Linear(n_sensors // 2, n_sensors),
            nn.Sigmoid()
        )
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(base_channels * 2 * time_steps, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, channels, time, sensors)
        batch_size = x.size(0)
        
        # Apply temporal convolutions
        x = x.permute(0, 1, 3, 2)  # (batch, channels, sensors, time)
        x = x.reshape(batch_size, -1, x.size(-1))  # Combine channels
        x = self.temporal_conv(x)
        
        # Apply spatial attention
        x = x.permute(0, 2, 1)  # (batch, time, features)
        attention_weights = self.spatial_attention(x)
        x = x * attention_weights
        
        # Flatten and extract features
        x = x.reshape(batch_size, -1)
        x = self.feature_extractor(x)
        
        return x

class VideoEncoder(nn.Module):
    """
    Encoder for video data using 3D CNNs and transformer attention.
    Processes spatiotemporal visual data to extract meaningful features.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int, int],  # (batch, channels, time, height, width)
        output_dim: int,
        base_channels: int = 32,
        dropout_rate: float = 0.1
    ):
        super(VideoEncoder, self).__init__()
        
        # Store input shape
        self.input_shape = input_shape
        _, in_channels, time_steps, height, width = input_shape
        
        # 3D CNN layers
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(dropout_rate),
            
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(dropout_rate),
            
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(dropout_rate)
        )
        
        # Calculate the size after convolutions
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Transformer attention
        self.transformer = nn.TransformerEncoderLayer(
            d_model=conv_output_size,
            nhead=8,
            dim_feedforward=conv_output_size * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_dim)
        )

    def _get_conv_output_size(self, input_shape: Tuple[int, int, int, int, int]) -> int:
        """Calculate the output size of the convolutional layers."""
        x = torch.randn(1, *input_shape[1:])
        x = self.conv3d(x)
        return int(torch.prod(torch.tensor(x.size()[1:])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, channels, time, height, width)
        batch_size = x.size(0)
        
        # Apply 3D convolutions
        x = self.conv3d(x)
        
        # Reshape for transformer
        x = x.permute(0, 2, 1, 3, 4)  # (batch, time, channels, height, width)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, features)
        
        # Apply transformer attention
        x = self.transformer(x)
        
        # Global average pooling over time
        x = x.mean(dim=1)
        
        # Extract final features
        x = self.feature_extractor(x)
        
        return x

class BehavioralEncoder(nn.Module):
    """
    Encoder for behavioral data using fully connected layers.
    Processes behavioral features to extract meaningful representations.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout_rate: float = 0.1
    ):
        super(BehavioralEncoder, self).__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) 