import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

from models.vae import MultiModalVAE
from models.encoders import MEGEncoder, VideoEncoder, BehavioralEncoder
from models.decoders import MEGDecoder, VideoDecoder, BehavioralDecoder

class MultiModalDataset(Dataset):
    """
    Dataset class for multi-modal data (MEG, video, and behavioral).
    """
    def __init__(
        self,
        meg_data: np.ndarray,  # (n_samples, channels, time, sensors)
        video_data: np.ndarray,  # (n_samples, channels, time, height, width)
        behavioral_data: np.ndarray,  # (n_samples, features)
        transform: Any = None
    ):
        self.meg_data = torch.FloatTensor(meg_data)
        self.video_data = torch.FloatTensor(video_data)
        self.behavioral_data = torch.FloatTensor(behavioral_data)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.meg_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    losses = {
        'total_loss': 0,
        'meg_loss': 0,
        'video_loss': 0,
        'behavioral_loss': 0,
        'kl_loss': 0
    }

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        # Move data to device
        meg = batch['meg'].to(device)
        video = batch['video'].to(device)
        behavioral = batch['behavioral'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(meg, video, behavioral)
        
        # Compute loss
        targets = {
            'meg': meg,
            'video': video,
            'behavioral': behavioral
        }
        batch_losses = model.compute_loss(outputs, targets)
        
        # Backward pass
        batch_losses['total_loss'].backward()
        optimizer.step()

        # Update statistics
        total_loss += batch_losses['total_loss'].item()
        for k, v in batch_losses.items():
            losses[k] += v.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{batch_losses['total_loss'].item():.4f}",
            'kl_loss': f"{batch_losses['kl_loss'].item():.4f}"
        })

    # Average losses
    n_batches = len(dataloader)
    for k in losses:
        losses[k] /= n_batches

    return losses

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    losses = {
        'total_loss': 0,
        'meg_loss': 0,
        'video_loss': 0,
        'behavioral_loss': 0,
        'kl_loss': 0
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move data to device
            meg = batch['meg'].to(device)
            video = batch['video'].to(device)
            behavioral = batch['behavioral'].to(device)

            # Forward pass
            outputs = model(meg, video, behavioral)
            
            # Compute loss
            targets = {
                'meg': meg,
                'video': video,
                'behavioral': behavioral
            }
            batch_losses = model.compute_loss(outputs, targets)

            # Update statistics
            for k, v in batch_losses.items():
                losses[k] += v.item()

    # Average losses
    n_batches = len(dataloader)
    for k in losses:
        losses[k] /= n_batches

    return losses

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = MultiModalVAE(
        meg_input_shape=config['data']['meg_shape'],
        video_input_shape=config['data']['video_shape'],
        behavioral_input_dim=config['data']['behavioral_dim'],
        latent_dim=config['model']['latent_dim'],
        shared_dim=config['model']['shared_dim'],
        modality_specific_dim=config['model']['modality_specific_dim'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Load data
    # Note: Replace these with your actual data loading code
    meg_data = np.random.randn(100, 1, 100, 102)  # Example MEG data
    video_data = np.random.randn(100, 3, 30, 64, 64)  # Example video data
    behavioral_data = np.random.randn(100, 10)  # Example behavioral data
    
    # Create datasets
    train_dataset = MultiModalDataset(
        meg_data=meg_data[:80],
        video_data=video_data[:80],
        behavioral_data=behavioral_data[:80]
    )
    
    val_dataset = MultiModalDataset(
        meg_data=meg_data[80:],
        video_data=video_data[80:],
        behavioral_data=behavioral_data[80:]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_losses = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_losses['total_loss'])
        
        # Print epoch statistics
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print("Training losses:", {k: f"{v:.4f}" for k, v in train_losses.items()})
        print("Validation losses:", {k: f"{v:.4f}" for k, v in val_losses.items()})
        
        # Save best model
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, Path(config['training']['checkpoint_dir']) / 'best_model.pth')

if __name__ == '__main__':
    main() 