import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from hyperbolic_lib.lib.lorentz.layers import (
    LorentzFullyConnected,
    LorentzLayerNorm,
    LorentzProjection,
    LorentzAct,
)

from hyperbolic_lib.lib.lorentz.layers_1d.LConv import (
    LorentzPureConv1d,
    LorentzConv1d,
    HyperbolicCayleyConv1D,
    HyperbolicStiefelConv1D,
)


class LorentzMovingAverageFast(nn.Module):
    def __init__(self, manifold, window_size=31, stride=1, padding='same'):
        super().__init__()
        self.manifold = manifold
        self.window_size = window_size
        self.stride = stride

        if padding == 'same':
            self.pad = (window_size - 1) // 2
        else:
            self.pad = 0

        # Fixed uniform weights
        self.register_buffer('weights', torch.ones(window_size) / window_size)

    def forward(self, x):
        # Pad if needed
        if self.pad > 0:
            x = torch.cat([
                x[..., :self.pad, :],
                x,
                x[..., -self.pad:, :]
            ], dim=-2)

        # Unfold into windows
        x_unfolded = x.unfold(-2, self.window_size, self.stride).permute(0, 1, 3, 2)

        # Reshape for batch processing
        orig_shape = x_unfolded.shape

        means = self.manifold.centroid(x_unfolded)

        # Reshape back
        return means.reshape(orig_shape[:-2] + (-1, orig_shape[-1])).squeeze()


class LorentzCrossAttention(nn.Module):
    def __init__(self, manifold, embed_dim, num_heads, dropout=0):
        super().__init__()
        self.manifold = manifold
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head projections
        self.q_proj = LorentzFullyConnected(manifold, embed_dim, embed_dim, nheads=num_heads)
        self.k_proj = LorentzFullyConnected(manifold, embed_dim, embed_dim, nheads=num_heads)
        self.v_proj = LorentzFullyConnected(manifold, embed_dim, embed_dim, nheads=num_heads)

        # Attention dropout
        self.attn_drop = nn.Dropout(dropout)

        # Output projection
        self.out_proj = LorentzFullyConnected(manifold, embed_dim, embed_dim, dropout=dropout)

        # Learnable curvature
        self.curvature = nn.Parameter(torch.tensor(1.0))
        self.scale = nn.Parameter(num_heads** (-0.5) * torch.ones((1)), requires_grad=False)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, query, key, value):
        # Project to query/key/value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Compute hyperbolic attention scores
        dist_matrix = -self.manifold.csqdist(q, k)
        attn_weights = F.softmax(dist_matrix / math.sqrt(self.embed_dim), dim=-1)
        # attn_weights = self.attn_drop(attn_weights)

        # Hyperbolic weighted aggregation
        attn_output = self.manifold.centroid(v, attn_weights)


        # cs_dist = self.manifold.csqdist(q, k) * self.scale.abs()
        # if self.num_heads == 1:
        #     cs_dist = cs_dist.unsqueeze(1)
        #
        # score = nn.Softmax(dim=-2)(1/(1+torch.log(1 + cs_dist))).permute(0, 1, 3, 2)

        # attn_output = self.manifold.centroid(v.unsqueeze(1), score)

        return self.out_proj(attn_output).squeeze(1)


class EEGBaselineExtractor(nn.Module):
    def __init__(self, manifold, channels, features, time_samples):
        super().__init__()



        # Temporal attention for neutral-like segments
        self.temporal_attn = nn.Sequential(
            LorentzConv1d(manifold, channels+1, 64, kernel_size=15, padding=7),
            LorentzAct(nn.ReLU(), manifold),
            LorentzConv1d(manifold, 64, 1, kernel_size=15, padding=7),
            nn.Sigmoid()
        )

        # Frequency-based baseline extraction
        self.freq_encoder = nn.Sequential(
            LorentzConv1d(manifold,channels+1, 128, kernel_size=3, stride=2, padding=1),
            LorentzAct(nn.ReLU(), manifold),
            LorentzConv1d(manifold,128, channels, kernel_size=3, stride=2, padding=1),
            # LorentzAct(nn.AdaptiveAvgPool1d(6), manifold)
        )

        self.baseline_decoder = nn.Sequential(
            LorentzFullyConnected(manifold, channels+1, 128),
            LorentzAct(nn.ReLU(), manifold),
            LorentzFullyConnected(manifold, 128, features * time_samples)
        )

    def forward(self, x):
        """x: [batch, channels, time_samples]"""
        # 1. Temporal attention weights
        attn_weights = self.temporal_attn(x)  # [batch, 1, time_samples]

        # 2. Frequency-based baseline features
        freq_features = self.freq_encoder(x).squeeze(-1)  # [batch, 64]

        # 3. Combine attention and frequency features
        baseline_flat = self.baseline_decoder(freq_features)
        baseline = baseline_flat.view(x.shape[0], x.shape[1], -1)

        # 4. Apply attention to create dynamic baseline
        return attn_weights * baseline


def segment_eeg(eeg_sequence, segment_length, overlap):
    """
    eeg_sequence: (batch_size, channels, total_length)
    segment_length: number of samples per segment
    overlap: number of overlapping samples between segments
    """
    stride = segment_length - overlap
    segments = eeg_sequence.unfold(dimension=-1, size=segment_length, step=stride)
    # segments shape: (batch_size, channels, num_segments, segment_length)
    segments = segments.permute(0, 2, 1, 3)  # (batch_size, num_segments, channels, segment_length)
    return segments

