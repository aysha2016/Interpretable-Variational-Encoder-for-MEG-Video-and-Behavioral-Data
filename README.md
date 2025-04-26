# Interpretable-Variational-Encoder-for-MEG-Video-and-Behavioral-Data
Project Overview
This project implements a multimodal Interpretable Variational Encoder (IVE) combining MEG signals, video frames, and behavioral covariates into a unified prediction framework.
We leverage 3D CNNs, Variational Autoencoders (VAE), and attention mechanisms (Transformers/Conformers) to extract compact, interpretable latent spaces related to physical and cognitive tasks.

The goal is to model how the human brain processes physical stimuli like motion, gravity, and air resistance, and apply these ideas toward broader cognitive and neuro-AI research.
 
IVE_Project/
│
├── data_preprocessing/
│   ├── meg_preprocessing.py
│   ├── video_preprocessing.py
│
├── encoders/
│   ├── meg_encoder.py
│   ├── video_encoder.py
│   ├── behavioral_encoder.py
│
├── models/
│   ├── multimodal_vae.py
│
├── callbacks/
│   ├── early_stopping.py
│   ├── latent_drift_monitor.py
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
Setup Instructions
1. Install Requirements
 
pip install -r requirements.txt
2. Prepare Data
MEG data: Place .fif MEG files under a data/meg/ folder.

Video data: Place video files (e.g., .mp4) under a data/video/ folder.

Behavioral covariates: Store behavioral data as .csv files under data/behavior/.

3. Preprocess Data
 
python data_preprocessing/meg_preprocessing.py
python data_preprocessing/video_preprocessing.py
4. Train Model
 
python main.py
Key Components
MEG Encoder: 3D CNN that processes spatiotemporal MEG signals.

Video Encoder: 3D CNN + Attention layers to model video frames dynamically.

Behavioral Encoder: Lightweight dense network for covariates.

VAE Structure: Learns a shared latent space.

Callbacks:

Early stopping

Latent space drift monitoring for stability.

Notes
We use MNE for MEG data handling.

PyTorch powers the model architecture.

Attention layers are optional and can be disabled via the config file.

