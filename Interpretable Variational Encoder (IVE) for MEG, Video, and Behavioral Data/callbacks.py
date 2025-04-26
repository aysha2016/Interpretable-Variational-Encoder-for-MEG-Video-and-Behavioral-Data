import numpy as np

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class LatentDriftMonitor:
    def __init__(self, threshold=0.1):
        self.previous_latent = None
        self.threshold = threshold

    def __call__(self, current_latent):
        if self.previous_latent is not None:
            drift = np.linalg.norm(current_latent - self.previous_latent)
            if drift > self.threshold:
                print(f"Warning: Latent drift detected. Drift: {drift}")
        self.previous_latent = current_latent