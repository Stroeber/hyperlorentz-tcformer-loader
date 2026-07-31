import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dAutoencoder(nn.Module):
    """
    Flexible 1D convolutional autoencoder.

    Input shape:
        (batch_size, in_channels, sequence_length)

    Works with arbitrary sequence lengths.
    """

    def __init__(
        self,
        in_channels=1,
        latent_dim=128,
        base_channels=32,
        bottleneck_length=16,
    ):
        super().__init__()

        self.bottleneck_length = bottleneck_length

        # -------------------------
        # Encoder
        # -------------------------
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(base_channels, base_channels * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(base_channels * 2, base_channels * 4,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(bottleneck_length)
        )

        self.fc_encoder = nn.Linear(
            base_channels * 4 * bottleneck_length,
            latent_dim
        )

        self.fc_decoder = nn.Linear(
            latent_dim,
            base_channels * 4 * bottleneck_length
        )

        # -------------------------
        # Decoder
        # -------------------------
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 2,
                      kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(base_channels * 2, base_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(base_channels, in_channels,
                      kernel_size=3, padding=1)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        z = self.fc_encoder(x)
        return z

    def decode(self, z, output_length):
        x = self.fc_decoder(z)

        batch_size = x.shape[0]

        x = x.view(
            batch_size,
            -1,
            self.bottleneck_length
        )

        # Upsample back to original sequence length
        x = F.interpolate(
            x,
            size=output_length,
            mode="linear",
            align_corners=False
        )

        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        output_length = x.shape[-1]
        z = self.encode(x)
        reconstruction = self.decode(z, output_length)
        return reconstruction