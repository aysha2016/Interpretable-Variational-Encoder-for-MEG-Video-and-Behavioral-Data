import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class MultiModalVAE(nn.Module):
    """
    Interpretable Variational Encoder (IVE) for multi-modal data analysis.
    Processes MEG, video, and behavioral data to learn shared and modality-specific
    latent representations.
    """
    def __init__(
        self,
        meg_input_shape: Tuple[int, int, int, int],  # (batch, channels, time, sensors)
        video_input_shape: Tuple[int, int, int, int, int],  # (batch, channels, time, height, width)
        behavioral_input_dim: int,
        latent_dim: int = 32,
        shared_dim: int = 16,
        modality_specific_dim: int = 8,
        dropout_rate: float = 0.1
    ):
        super(MultiModalVAE, self).__init__()
        
        # Store dimensions
        self.latent_dim = latent_dim
        self.shared_dim = shared_dim
        self.modality_specific_dim = modality_specific_dim
        
        # Initialize encoders
        self.meg_encoder = MEGEncoder(
            input_shape=meg_input_shape,
            output_dim=shared_dim + modality_specific_dim
        )
        
        self.video_encoder = VideoEncoder(
            input_shape=video_input_shape,
            output_dim=shared_dim + modality_specific_dim
        )
        
        self.behavioral_encoder = BehavioralEncoder(
            input_dim=behavioral_input_dim,
            output_dim=shared_dim + modality_specific_dim
        )
        
        # Shared latent space
        self.shared_encoder = nn.Sequential(
            nn.Linear(shared_dim * 3, shared_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(shared_dim * 2, shared_dim)
        )
        
        # Modality-specific latent spaces
        self.meg_specific_encoder = nn.Sequential(
            nn.Linear(modality_specific_dim, modality_specific_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.video_specific_encoder = nn.Sequential(
            nn.Linear(modality_specific_dim, modality_specific_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.behavioral_specific_encoder = nn.Sequential(
            nn.Linear(modality_specific_dim, modality_specific_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(shared_dim + modality_specific_dim * 3, latent_dim)
        self.fc_logvar = nn.Linear(shared_dim + modality_specific_dim * 3, latent_dim)
        
        # Decoders
        self.meg_decoder = MEGDecoder(
            latent_dim=latent_dim,
            output_shape=meg_input_shape
        )
        
        self.video_decoder = VideoDecoder(
            latent_dim=latent_dim,
            output_shape=video_input_shape
        )
        
        self.behavioral_decoder = BehavioralDecoder(
            latent_dim=latent_dim,
            output_dim=behavioral_input_dim
        )

    def encode(self, meg: torch.Tensor, video: torch.Tensor, behavioral: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode inputs from all modalities into latent space.
        
        Args:
            meg: MEG data tensor
            video: Video data tensor
            behavioral: Behavioral data tensor
            
        Returns:
            Tuple of (mu, logvar) for the latent space
        """
        # Encode each modality
        meg_features = self.meg_encoder(meg)
        video_features = self.video_encoder(video)
        behavioral_features = self.behavioral_encoder(behavioral)
        
        # Split into shared and modality-specific features
        meg_shared, meg_specific = torch.split(meg_features, [self.shared_dim, self.modality_specific_dim], dim=1)
        video_shared, video_specific = torch.split(video_features, [self.shared_dim, self.modality_specific_dim], dim=1)
        behavioral_shared, behavioral_specific = torch.split(behavioral_features, [self.shared_dim, self.modality_specific_dim], dim=1)
        
        # Process shared features
        shared_features = torch.cat([meg_shared, video_shared, behavioral_shared], dim=1)
        shared_latent = self.shared_encoder(shared_features)
        
        # Process modality-specific features
        meg_specific = self.meg_specific_encoder(meg_specific)
        video_specific = self.video_specific_encoder(video_specific)
        behavioral_specific = self.behavioral_specific_encoder(behavioral_specific)
        
        # Combine all features
        combined = torch.cat([
            shared_latent,
            meg_specific,
            video_specific,
            behavioral_specific
        ], dim=1)
        
        # Generate latent space parameters
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode from latent space to all modalities.
        """
        return {
            'meg': self.meg_decoder(z),
            'video': self.video_decoder(z),
            'behavioral': self.behavioral_decoder(z)
        }

    def forward(self, meg: torch.Tensor, video: torch.Tensor, behavioral: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through the entire model.
        
        Returns:
            Dictionary containing:
            - Reconstructions for each modality
            - Latent space parameters (mu, logvar)
            - Latent representation (z)
        """
        mu, logvar = self.encode(meg, video, behavioral)
        z = self.reparameterize(mu, logvar)
        reconstructions = self.decode(z)
        
        return {
            'reconstructions': reconstructions,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    def compute_loss(self, outputs: Dict[str, Any], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the VAE loss including reconstruction and KL divergence.
        """
        # Reconstruction losses
        meg_loss = F.mse_loss(outputs['reconstructions']['meg'], targets['meg'])
        video_loss = F.mse_loss(outputs['reconstructions']['video'], targets['video'])
        behavioral_loss = F.mse_loss(outputs['reconstructions']['behavioral'], targets['behavioral'])
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
        
        # Total loss
        total_loss = meg_loss + video_loss + behavioral_loss + kl_loss
        
        return {
            'total_loss': total_loss,
            'meg_loss': meg_loss,
            'video_loss': video_loss,
            'behavioral_loss': behavioral_loss,
            'kl_loss': kl_loss
        } 