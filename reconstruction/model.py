import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class SimpleDecoder(nn.Module):
    """
      Takes a flattened embedding and reconstructs the full sequence.
      Input:  z  (B, N)
      Output: y  (B, T, D_in)
      """

    def __init__(self, n_in: int, T: int, d_in: int,
                 hidden_mult: int = 2, p_drop: float = 0.1, use_ln: bool = True):
        super().__init__()
        self.T, self.d_in = T, d_in
        hidden = hidden_mult
        self.norm = nn.LayerNorm(n_in) if use_ln else nn.Identity()
        # self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            # nn.Dropout(p_drop),
            nn.Linear(hidden, T * d_in),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(1, 2)),  # local smoothing
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1),
        )

        self.init_weights()

    def forward(self, z):  # z: (B, N)
        # z = self.drop(self.norm(z))  # (B, N)
        # z = self.norm(z)  # (B, N)
        y = self.mlp(z)  # (B, T*D_in)
        y = y.view(z.size(0), 1, self.T, self.d_in)  # (B, T, D_in)
        return self.refine(y)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class TimeSeriesPretrainer(nn.Module):
    """
    Wraps an existing encoder + tiny decoder for self-supervised pretraining.
    """
    def __init__(self, encoder: nn.Module, d_model: int, d_in: int,
                 p_drop: float = 0.0, use_ln: bool = True):
        super().__init__()
        self.encoder = encoder
        self.decoder = SimpleDecoder(d_model, d_in, p_drop=p_drop, use_ln=use_ln)

    def forward(self, x):
        """
        x: (B, T, D_in)
        returns reconstruction: (B, T, D_in)
        """
        h = self.encoder(x)        # (B, T, D_model)
        out = self.decoder(h)      # (B, T, D_in)
        return out
