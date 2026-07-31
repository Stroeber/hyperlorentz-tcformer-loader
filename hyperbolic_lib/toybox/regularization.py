import math

import torch

from lib.geoopt import Lorentz

torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal

from lib.lorentz.manifold import CustomLorentz
import lib.geoopt as geoopt



class LorentzLearnedNorm(nn.Module):
    """ Implementation of a general Lorentz Activation on space components.
    """
    def __init__(self, manifold: CustomLorentz):
        super(LorentzLearnedNorm, self).__init__()
        self.manifold = manifold
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        sq_norm = torch.abs(self.minkowski_dot(x, x, keepdim=False)).clamp(min=1e-2)
        real_norm = torch.sqrt(torch.abs(sq_norm))
        projected_point = torch.einsum("...i,...->...i", x, self.k * self.scale * real_norm)
        return projected_point


in_features = 32
out_features = 64


a = torch.rand((16, 32, 32, in_features)).to(device)*10 - 5
b = torch.rand((16, 32, 32, in_features)).to(device)*3

manifold = CustomLorentz(k=4).to(device)
x = manifold.projx(a)
y = manifold.projx(b)

sq_norm = torch.abs(manifold.minkowski_dot(x, x, keepdim=False)).clamp(min=1e-2)
real_norm = torch.sqrt(torch.abs(sq_norm))
projected_point = torch.einsum("...i,...->...i", x, 4 / real_norm)

print("break")
