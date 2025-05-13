# Interpretable Variational Encoder (IVE) for Multi-Modal Data Analysis

This repository implements a deep learning framework for modeling MEG (magnetoencephalography), video, and behavioral data using an interpretable variational encoder. The architecture is designed to uncover shared and modality-specific latent factors across brain signals, visual input, and observable behavior.

## Features

- 🧬 **Variational Inference**: Captures uncertainty and latent dynamics across multi-modal data
- 🧠 **MEG Processing**: Extracts spatiotemporal neural representations using 3D CNNs and attention mechanisms
- 🎥 **Video Feature Encoder**: Processes facial, motion, or contextual cues using transformer-based attention
- 🧍 **Behavioral Decoder**: Links latent variables to reaction times, responses, or emotional states
- 🔍 **Interpretability Layer**: Disentangles latent variables for neuroscientific insight and model transparency

## Architecture

The framework consists of several key components:

1. **Multi-Modal VAE**: Core model that processes and combines data from all modalities
2. **Modality-Specific Encoders**:
   - MEG Encoder: 3D CNNs with spatial attention
   - Video Encoder: 3D CNNs with transformer attention
   - Behavioral Encoder: Fully connected layers
3. **Shared Latent Space**: Learns common representations across modalities
4. **Modality-Specific Latent Spaces**: Captures unique features of each modality
5. **Decoders**: Reconstructs original data from latent representations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aysha2016/interpretable-variational-encoder.git
cd interpretable-variational-encoder
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**:
   - MEG data should be preprocessed and formatted as (samples, channels, time, sensors)
   - Video data should be preprocessed and formatted as (samples, channels, time, height, width)
   - Behavioral data should be formatted as (samples, features)

2. **Configuration**:
   - Modify `config.yaml` to match your data dimensions and training requirements
   - Adjust model architecture parameters as needed

3. **Training**:
```bash
python train.py
```

4. **Model Evaluation**:
   - The training script automatically saves the best model based on validation loss
   - Checkpoints are saved in the `checkpoints` directory
   - Training logs are saved in the `logs` directory

## Project Structure

```
.
├── models/
│   ├── vae.py           # Main VAE model
│   ├── encoders.py      # Modality-specific encoders
│   └── decoders.py      # Reconstruction decoders
├── train.py             # Training script
├── config.yaml          # Configuration file
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Model Architecture Details

### MEG Encoder
- Temporal convolutions for feature extraction
- Spatial attention mechanism for sensor importance
- Batch normalization and dropout for regularization

### Video Encoder
- 3D CNNs for spatiotemporal feature extraction
- Transformer attention for temporal dependencies
- Multi-scale feature aggregation

### Behavioral Encoder
- Fully connected layers with batch normalization
- ReLU activation and dropout for regularization
- Configurable architecture based on input dimensions

### Shared Latent Space
- Combines features from all modalities
- Learns common representations
- Enables cross-modal analysis

 






   
