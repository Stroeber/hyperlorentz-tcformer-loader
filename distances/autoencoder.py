import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class EEGAutoencoder(nn.Module):
    def __init__(self, in_channels, sequence_length, latent_dim=64, hidden_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # --- ENCODER ---
        # 1. Spatial Phase: Convolve across the channel dimension first
        # Input shape: (Batch, 1, Channels, Time) -> Output shape: (Batch, hidden_dim, 1, Time)
        self.spatial_conv = nn.Conv2d(1, hidden_dim, kernel_size=(in_channels, 1))
        
        # 2. Temporal Phase: Convolve across the time dimension
        # Input shape: (Batch, hidden_dim, Time)
        self.temporal_conv1 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)
        self.temporal_conv2 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1)
        
        # Mathematically track the reduced temporal length after strided convolutions
        l = sequence_length
        l = (l + 2*1 - 3) // 2 + 1  # length after temporal_conv1
        l = (l + 2*1 - 3) // 2 + 1  # length after temporal_conv2
        self.reduced_time_len = l
        
        # Linear bottleneck to fixed latent space size
        self.flat_features = (hidden_dim * 4) * self.reduced_time_len
        self.fc_encode = nn.Linear(self.flat_features, latent_dim)
        
        # --- DECODER ---
        self.fc_decode = nn.Linear(latent_dim, self.flat_features)
        
        # Temporal upsampling layers (keep size constant via padding=1, handled by interpolation)
        self.temporal_deconv1 = nn.Conv1d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1)
        self.temporal_deconv2 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        
        # Spatial expansion back to original channel count
        self.spatial_deconv = nn.ConvTranspose2d(hidden_dim, 1, kernel_size=(in_channels, 1))

    def encode(self, x):
        # Input 'x' shape: (Batch, Channels, Time)
        # x = x.unsqueeze(1)  # Add single image-channel dimension -> (Batch, 1, Channels, Time)
        x = F.relu(self.spatial_conv(x))
        x = x.squeeze(2)    # Remove the collapsed channel dimension -> (Batch, hidden_dim, Time)
        
        # Temporal processing
        x = F.relu(self.temporal_conv1(x))
        x = F.relu(self.temporal_conv2(x))
        
        # Flatten and project to latent space
        x = x.view(x.size(0), -1)
        return self.fc_encode(x)

    def decode(self, latent):
        x = F.relu(self.fc_decode(latent))
        x = x.view(x.size(0), self.hidden_dim * 4, self.reduced_time_len)
        
        # Use bilinear/linear interpolation to scale back up to the exact original sequence length.
        # This completely avoids target shape mismatch bugs with odd sequence lengths.
        x = F.interpolate(x, size=self.sequence_length, mode='linear', align_corners=False)
        
        x = F.relu(self.temporal_deconv1(x))
        x = F.relu(self.temporal_deconv2(x))
        
        x = x.unsqueeze(2)  # Prepare for spatial expansion -> (Batch, hidden_dim, 1, Time)
        x = self.spatial_deconv(x)  # -> (Batch, 1, Channels, Time)
        return x#.squeeze(1)  # -> (Batch, Channels, Time)

    def forward(self, x):
        return self.decode(self.encode(x))


    def train_eeg_autoencoder(self, model, dataloader, epochs=100, lr=1e-3, device='cuda:0'):
        """
        Trains the EEGAutoencoder using MSE loss.
        Assumes dataloader yields (eeg_batch, subject_id, label).
        """
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        print(f"Starting training on device: {device}")
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0
            
            for eeg_batch, _, _ in dataloader:
                # Send data to device
                eeg_batch = eeg_batch.to(device).float()
                batch_size = eeg_batch.size(0)
                
                # Forward pass
                optimizer.zero_grad()
                reconstructed = model(eeg_batch)
                loss = criterion(reconstructed, eeg_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track statistics
                running_loss += loss.item() * batch_size
                total_samples += batch_size
                
            epoch_loss = running_loss / total_samples
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {epoch_loss:.6f}")
            
        print("Training Complete!")
        return model

    def LatentEuclideanDistance(self, data_A, data_B):
        """
        Calculates the Euclidean distance between the mean latent profiles of Subject A and Subject B.
        Shapes of data_A / data_B: (num_trials, channels, time_steps)
        """
        with torch.no_grad():
            # Extract latent states for all trials
            latent_A = self.encode(data_A)  # Shape: (num_trials_A, latent_dim)
            latent_B = self.encode(data_B)  # Shape: (num_trials_B, latent_dim)
            
            # Compute average subject profile across trials
            mean_profile_A = latent_A.mean(dim=0)   # Shape: (latent_dim,)
            mean_profile_B = latent_B.mean(dim=0)   # Shape: (latent_dim,)
            
            # Calculate standard Euclidean distance
            distance = torch.dist(mean_profile_A, mean_profile_B, p=2)
            
        return distance