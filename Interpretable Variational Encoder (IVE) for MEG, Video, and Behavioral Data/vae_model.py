import torch
import torch.nn as nn
from encoders import MEGEncoder, VideoEncoder, BehavioralEncoder

class VAE(nn.Module):
    def __init__(self, behavioral_input_dim, latent_dim=32):
        super(VAE, self).__init__()
        self.meg_encoder = MEGEncoder()
        self.video_encoder = VideoEncoder()
        self.behavioral_encoder = BehavioralEncoder(behavioral_input_dim, latent_dim)

        self.fc_mu = nn.Linear(32 * 8 + latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(32 * 8 + latent_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 1)  # example output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, meg, video, behavioral):
        meg_feat = self.meg_encoder(meg).view(meg.size(0), -1)
        video_feat = self.video_encoder(video).view(video.size(0), -1)
        behavioral_feat = self.behavioral_encoder(behavioral)

        combined = torch.cat([meg_feat, video_feat, behavioral_feat], dim=1)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        z = self.reparameterize(mu, logvar)
        output = self.fc_decode(z)
        return output, mu, logvar