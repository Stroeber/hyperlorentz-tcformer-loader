import math

import torch

from lib.geoopt import Lorentz
from lib.geoopt.manifolds.lorentz.math import expmap0

#torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

from lib.lorentz.manifold import CustomLorentz


def get_max_distances(k):
    d = torch.tensor(1e4).to("cuda")

    if (d/k).min() < 0.99:
        print("smaller")
    return torch.sqrt(k) * torch.arccosh(d / k)

def rescale_to_max(euclid_vector, k):
    max_time = torch.sqrt(k) * torch.arccosh(2e3 / k)
    tanh_factor = torch.atanh(torch.tensor(0.99, device=k.device)) / (max_time * 2)
    x_norm = torch.norm(euclid_vector, dim=-1, keepdim=True)
    x_out = x_norm.clone()
    new_norms = max_time * torch.tanh(tanh_factor * x_norm)
    return new_norms * euclid_vector / x_out


in_features = 32
out_features = 64
k = torch.tensor(4).to("cuda:0")

a = torch.rand((16, 32, 32, in_features)).to(device)*10 - 5
b = torch.rand((16, 32, 32, in_features)).to(device)*3

manifold = CustomLorentz(k=k).to(device)
x = manifold.projx(a)
y = manifold.projx(b)

max_distance = get_max_distances(manifold.k)
print("max_distance")

x_current = manifold.dist0(x)
y_Current = manifold.dist0(y)

manifold_2 = CustomLorentz(torch.tensor(0.5)).to(device)
x_tangent = manifold.logmap0(x)
x_tangent_2 = rescale_to_max(x_tangent, manifold_2.k)

x_2 = manifold_2.expmap0(x_tangent)
x_rescaled_2 = manifold_2.expmap0(x_tangent_2)

x_new = manifold_2.dist0(x_2).max()


print("break")
