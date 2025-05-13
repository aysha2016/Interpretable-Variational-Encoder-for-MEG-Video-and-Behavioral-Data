# @title Install Required Packages
!pip install torch torchvision numpy mne opencv-python scikit-learn matplotlib pandas tqdm PyYAML

# @title Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from typing import Dict, Any, Tuple
import mne
import cv2
from sklearn.manifold import TSNE

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# @title Define Model Architecture
class MEGEncoder(nn.Module):
    def __init__(self, input_shape, output_dim, base_channels=32, dropout_rate=0.1):
        super(MEGEncoder, self).__init__()
        _, in_channels, time_steps, n_sensors = input_shape
        
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
        
        self.spatial_attention = nn.Sequential(
            nn.Linear(n_sensors, n_sensors // 2),
            nn.ReLU(),
            nn.Linear(n_sensors // 2, n_sensors),
            nn.Sigmoid()
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(base_channels * 2 * time_steps, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(batch_size, -1, x.size(-1))
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)
        attention_weights = self.spatial_attention(x)
        x = x * attention_weights
        x = x.reshape(batch_size, -1)
        x = self.feature_extractor(x)
        return x

class VideoEncoder(nn.Module):
    def __init__(self, input_shape, output_dim, base_channels=32, dropout_rate=0.1):
        super(VideoEncoder, self).__init__()
        _, in_channels, time_steps, height, width = input_shape
        
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
        
        conv_output_size = self._get_conv_output_size(input_shape)
        
        self.transformer = nn.TransformerEncoderLayer(
            d_model=conv_output_size,
            nhead=8,
            dim_feedforward=conv_output_size * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_dim)
        )

    def _get_conv_output_size(self, input_shape):
        x = torch.randn(1, *input_shape[1:])
        x = self.conv3d(x)
        return int(torch.prod(torch.tensor(x.size()[1:])))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv3d(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size, x.size(1), -1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.feature_extractor(x)
        return x

class BehavioralEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 128), dropout_rate=0.1):
        super(BehavioralEncoder, self).__init__()
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
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MultiModalVAE(nn.Module):
    def __init__(self, meg_input_shape, video_input_shape, behavioral_input_dim,
                 latent_dim=32, shared_dim=16, modality_specific_dim=8, dropout_rate=0.1):
        super(MultiModalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.shared_dim = shared_dim
        self.modality_specific_dim = modality_specific_dim
        
        # Encoders
        self.meg_encoder = MEGEncoder(meg_input_shape, shared_dim + modality_specific_dim)
        self.video_encoder = VideoEncoder(video_input_shape, shared_dim + modality_specific_dim)
        self.behavioral_encoder = BehavioralEncoder(behavioral_input_dim, shared_dim + modality_specific_dim)
        
        # Shared latent space
        self.shared_encoder = nn.Sequential(
            nn.Linear(shared_dim * 3, shared_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(shared_dim * 2, shared_dim)
        )
        
        # Modality-specific encoders
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
        
        # Decoders (simplified for demo)
        self.meg_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(meg_input_shape[1:])),
            nn.Sigmoid()
        )
        
        self.video_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, np.prod(video_input_shape[1:])),
            nn.Sigmoid()
        )
        
        self.behavioral_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, behavioral_input_dim)
        )

    def encode(self, meg, video, behavioral):
        meg_features = self.meg_encoder(meg)
        video_features = self.video_encoder(video)
        behavioral_features = self.behavioral_encoder(behavioral)
        
        meg_shared, meg_specific = torch.split(meg_features, [self.shared_dim, self.modality_specific_dim], dim=1)
        video_shared, video_specific = torch.split(video_features, [self.shared_dim, self.modality_specific_dim], dim=1)
        behavioral_shared, behavioral_specific = torch.split(behavioral_features, [self.shared_dim, self.modality_specific_dim], dim=1)
        
        shared_features = torch.cat([meg_shared, video_shared, behavioral_shared], dim=1)
        shared_latent = self.shared_encoder(shared_features)
        
        meg_specific = self.meg_specific_encoder(meg_specific)
        video_specific = self.video_specific_encoder(video_specific)
        behavioral_specific = self.behavioral_specific_encoder(behavioral_specific)
        
        combined = torch.cat([shared_latent, meg_specific, video_specific, behavioral_specific], dim=1)
        
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        meg_recon = self.meg_decoder(z).view(-1, *self.meg_input_shape[1:])
        video_recon = self.video_decoder(z).view(-1, *self.video_input_shape[1:])
        behavioral_recon = self.behavioral_decoder(z)
        
        return {
            'meg': meg_recon,
            'video': video_recon,
            'behavioral': behavioral_recon
        }

    def forward(self, meg, video, behavioral):
        self.meg_input_shape = meg.shape
        self.video_input_shape = video.shape
        
        mu, logvar = self.encode(meg, video, behavioral)
        z = self.reparameterize(mu, logvar)
        reconstructions = self.decode(z)
        
        return {
            'reconstructions': reconstructions,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    def compute_loss(self, outputs, targets):
        meg_loss = nn.MSELoss()(outputs['reconstructions']['meg'], targets['meg'])
        video_loss = nn.MSELoss()(outputs['reconstructions']['video'], targets['video'])
        behavioral_loss = nn.MSELoss()(outputs['reconstructions']['behavioral'], targets['behavioral'])
        
        kl_loss = -0.5 * torch.mean(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
        
        total_loss = meg_loss + video_loss + behavioral_loss + kl_loss
        
        return {
            'total_loss': total_loss,
            'meg_loss': meg_loss,
            'video_loss': video_loss,
            'behavioral_loss': behavioral_loss,
            'kl_loss': kl_loss
        }

# @title Generate Synthetic Data
def generate_synthetic_data(n_samples=1000):
    # MEG data (samples, channels, time, sensors)
    meg_data = np.random.randn(n_samples, 1, 100, 102)
    t = np.linspace(0, 2*np.pi, 100)
    for i in range(n_samples):
        meg_data[i, 0, :, :] += np.sin(t[:, None] + np.random.randn(102))
    
    # Video data (samples, channels, time, height, width)
    video_data = np.random.randn(n_samples, 3, 30, 64, 64)
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y)
    for i in range(n_samples):
        for t in range(30):
            video_data[i, :, t] += np.exp(-(X**2 + Y**2)/0.5)[None, :, :]
    
    # Behavioral data (samples, features)
    behavioral_data = np.random.randn(n_samples, 10)
    behavioral_data[:, 0] = np.mean(meg_data[:, 0, :, :], axis=(1, 2))
    
    return meg_data, video_data, behavioral_data

# @title Create Dataset and DataLoader
class MultiModalDataset(Dataset):
    def __init__(self, meg_data, video_data, behavioral_data, transform=None):
        self.meg_data = torch.FloatTensor(meg_data)
        self.video_data = torch.FloatTensor(video_data)
        self.behavioral_data = torch.FloatTensor(behavioral_data)
        self.transform = transform
    
    def __len__(self):
        return len(self.meg_data)
    
    def __getitem__(self, idx):
        meg = self.meg_data[idx]
        video = self.video_data[idx]
        behavioral = self.behavioral_data[idx]
        
        if self.transform:
            meg = self.transform(meg)
            video = self.transform(video)
            behavioral = self.transform(behavioral)
        
        return {
            'meg': meg,
            'video': video,
            'behavioral': behavioral
        }

# @title Training Function
def train_model(model, train_loader, val_loader, n_epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_kl_loss': [],
        'val_kl_loss': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        train_kl_losses = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}'):
            meg = batch['meg'].to(device)
            video = batch['video'].to(device)
            behavioral = batch['behavioral'].to(device)
            
            optimizer.zero_grad()
            outputs = model(meg, video, behavioral)
            
            targets = {'meg': meg, 'video': video, 'behavioral': behavioral}
            losses = model.compute_loss(outputs, targets)
            
            losses['total_loss'].backward()
            optimizer.step()
            
            train_losses.append(losses['total_loss'].item())
            train_kl_losses.append(losses['kl_loss'].item())
        
        # Validation
        model.eval()
        val_losses = []
        val_kl_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                meg = batch['meg'].to(device)
                video = batch['video'].to(device)
                behavioral = batch['behavioral'].to(device)
                
                outputs = model(meg, video, behavioral)
                losses = model.compute_loss(outputs, targets)
                
                val_losses.append(losses['total_loss'].item())
                val_kl_losses.append(losses['kl_loss'].item())
        
        # Update learning rate
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_kl_loss'].append(np.mean(train_kl_losses))
        history['val_kl_loss'].append(np.mean(val_kl_losses))
        
        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}, KL Loss: {history['train_kl_loss'][-1]:.4f}")
        print(f"Val Loss: {history['val_loss'][-1]:.4f}, KL Loss: {history['val_kl_loss'][-1]:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return history

# @title Visualization Functions
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_kl_loss'], label='Train')
    plt.plot(history['val_kl_loss'], label='Validation')
    plt.title('KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_latent_space(model, dataloader):
    model.eval()
    latent_representations = []
    
    with torch.no_grad():
        for batch in dataloader:
            meg = batch['meg'].to(device)
            video = batch['video'].to(device)
            behavioral = batch['behavioral'].to(device)
            
            outputs = model(meg, video, behavioral)
            z = outputs['z'].cpu().numpy()
            latent_representations.append(z)
    
    latent_representations = np.concatenate(latent_representations, axis=0)
    
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_representations)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5)
    plt.title('t-SNE Visualization of Latent Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

def visualize_reconstructions(model, dataloader, n_samples=5):
    model.eval()
    
    batch = next(iter(dataloader))
    meg = batch['meg'][:n_samples].to(device)
    video = batch['video'][:n_samples].to(device)
    behavioral = batch['behavioral'][:n_samples].to(device)
    
    with torch.no_grad():
        outputs = model(meg, video, behavioral)
        reconstructions = outputs['reconstructions']
    
    # Plot MEG reconstructions
    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(meg[i, 0].cpu().numpy(), aspect='auto')
        plt.title(f'Original MEG {i+1}')
        
        plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(reconstructions['meg'][i, 0].cpu().numpy(), aspect='auto')
        plt.title(f'Reconstructed MEG {i+1}')
    plt.tight_layout()
    plt.show()
    
    # Plot video reconstructions (first frame only)
    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(video[i, :, 0].permute(1, 2, 0).cpu().numpy())
        plt.title(f'Original Video {i+1}')
        
        plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(reconstructions['video'][i, :, 0].permute(1, 2, 0).cpu().numpy())
        plt.title(f'Reconstructed Video {i+1}')
    plt.tight_layout()
    plt.show()

# @title Main Training Loop
# Generate data
meg_data, video_data, behavioral_data = generate_synthetic_data(n_samples=1000)

# Create datasets
train_size = int(0.8 * len(meg_data))
train_dataset = MultiModalDataset(
    meg_data=meg_data[:train_size],
    video_data=video_data[:train_size],
    behavioral_data=behavioral_data[:train_size]
)

val_dataset = MultiModalDataset(
    meg_data=meg_data[train_size:],
    video_data=video_data[train_size:],
    behavioral_data=behavioral_data[train_size:]
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
model = MultiModalVAE(
    meg_input_shape=(1, 1, 100, 102),
    video_input_shape=(1, 3, 30, 64, 64),
    behavioral_input_dim=10,
    latent_dim=32,
    shared_dim=16,
    modality_specific_dim=8
).to(device)

# Train model
history = train_model(model, train_loader, val_loader, n_epochs=50)

# Visualize results
plot_training_history(history)
visualize_latent_space(model, val_loader)
visualize_reconstructions(model, val_loader)

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'meg_input_shape': (1, 1, 100, 102),
        'video_input_shape': (1, 3, 30, 64, 64),
        'behavioral_input_dim': 10,
        'latent_dim': 32,
        'shared_dim': 16,
        'modality_specific_dim': 8
    }
}, 'ive_model.pth') 