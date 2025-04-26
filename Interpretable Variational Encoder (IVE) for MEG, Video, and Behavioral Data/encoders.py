import torch
import torch.nn as nn

class MEGEncoder(nn.Module):
    def __init__(self):
        super(MEGEncoder, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        return self.conv3d(x)

class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

    def forward(self, x):
        x = self.conv3d(x)
        b, c, d, h, w = x.size()
        x = x.view(b, c, -1).permute(2, 0, 1)  # (sequence_len, batch, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.mean(dim=0)

class BehavioralEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(BehavioralEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)