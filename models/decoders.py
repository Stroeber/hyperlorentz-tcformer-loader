import torch
import torch.nn as nn

from hyperbolic_lib.lib.lorentz.layers import LorentzMLR
from hyperbolic_lib.lib.geoopt import ManifoldParameter
from hyperbolic_lib.lib.lorentz.layers.Kernels import get_learned_kernels
from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz
from models.blocks import LorentzInceptionBlock
from utils.utils_h import (Conv1dSamePadding,
                                       CONV1D_TYPES,
                                       BATCH1D_TYPES,
                                       POOL1D_TYPES)


INIT_EPS = 1e-3


class LorentzMLRDecoder(nn.Module):
    def __init__(self, manifold, in_channels, n_classes):
        super(LorentzMLRDecoder, self).__init__()

        self.manifold = manifold
        self.mlr = LorentzMLR(self.manifold, in_channels, n_classes)

    def forward(self, x):
        return self.mlr(x)


class LorentzPrototypeDecoder(nn.Module):
    def __init__(self, manifold, in_channels, n_classes):
        super(LorentzPrototypeDecoder, self).__init__()

        self.manifold = manifold

        self.prototypes = self.manifold.add_time(torch.randn((n_classes, in_channels), device=self.manifold.k.device))
        self.prototypes = ManifoldParameter(self.prototypes, self.manifold)

    def forward(self, x):

        prototypes = self.manifold.projx(self.prototypes)
        # prototypes = self.manifold.rescale_to_max(prototypes)

        distances = self.manifold.sqdist(x.unsqueeze(-2), prototypes)

        return -distances.squeeze()


def dist2planebm(manifold, x, a, p, keepdim=False, dim=-1):

    diff = manifold.mobius_add(manifold.projx(-p), x, dim=dim)

    diff_norm = diff.norm(dim=dim, keepdim=keepdim, p=2)

    diff_norm2 = diff.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(1e-15)

    dist_p_x = 2 * torch.atanh(diff_norm)

    b_func_nume = torch.log(1 - diff_norm2)

    diff_minus_a = diff - a / a.norm(dim=dim, keepdim=keepdim, p=2)

    diff_minus_a_norm2 = diff_minus_a.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(1e-15)

    b_func_deno = torch.log(diff_minus_a_norm2)

    b_func = b_func_nume - b_func_deno

    distance = dist_p_x * b_func / diff_norm

    return distance


class LorentzBusmanDecoder(nn.Module):
    def __init__(self, manifold, in_channels, n_classes):
        super(LorentzBusmanDecoder, self).__init__()

        self.manifold = manifold
        self.n_classes = n_classes

        self.tangent = nn.Parameter(torch.randn(n_classes, in_channels) * INIT_EPS)

        self.point = self.manifold.add_time(torch.randn((n_classes, in_channels), device=self.manifold.k.device) * INIT_EPS)
        self.point = ManifoldParameter(self.prototypes, self.manifold)

    def forward(self, x):
        bs = x.shape[0]
        d = x.shape[1]

        input = torch.reshape(x, (-1, self.feat_dim))
        distances = torch.zeros_like(torch.empty(input.shape[0], self.n_classes), device="cuda:0",
                                     requires_grad=False)

        for i in range(self.num_outcome):
            point_i = self.point[i]
            tangent_i = self.tangent[i]

            distances[:, i] = self.ball.dist2planebm(
                x=input, a=tangent_i, p=point_i
            )
        return distances


class BaselineInceptionVAEDecoder(nn.Module):
    def __init__(self,
                 manifold,
                 in_channel=1,
                 features=18 * 18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0):
        super().__init__()

        self.manifold = manifold
        self.baseline_block = LorentzInceptionBlock(self.manifold,
                                                             in_channels=in_channel,
                                                             n_filters=int(features / 4),
                                                             kernel_sizes=kernel_sizes,
                                                             bottleneck_channels=inception_channels,
                                                             activation=None,
                                                             return_indices=False,
                                                             conv_type=conv_type,
                                                             batch_type=None,
                                                             pool_type="average",
                                                             dropout=dropout)


        self.layer_norm = BATCH1D_TYPES["original"](self.manifold, features + 1) if batch_type is not None else nn.Sequential()
        self.activation = nn.ReLU(inplace=True)


    def forward(self, x):

        processed = self.baseline_block(x)
        normed = self.layer(processed)

        return self.activation(normed)



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

