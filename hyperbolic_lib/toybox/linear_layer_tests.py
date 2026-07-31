import torch
import math
device = torch.device("cuda")
import torch.nn as nn

import lib.geoopt
from lib.geoopt.manifolds import Stiefel
from lib.lorentz.manifold import CustomLorentz


class LorentzPureBoost(torch.nn.Module):
    def __init__(self, manifold, dim, regularize=True):
        super(LorentzPureBoost, self).__init__()

        self.dim = dim

        self.v = nn.Parameter(torch.rand((dim - 1, 1)).to("cuda:0"))

        self.eye = nn.Parameter(torch.eye(dim - 1).to("cuda:0"), requires_grad=False)
        self.manifold = manifold

    def forward(self, x):

        norm = self.v.norm(2, dim=0, keepdim=False)
        # desired = torch.clamp(norm, max=0.99)
        desired = torch.sigmoid(norm)
        v = self.v * (desired / norm)

        # get boost
        gamma = 1 / torch.sqrt(1 - torch.norm(v) ** 2).reshape(1, -1)
        el_1 = -gamma * v.T
        el_2 = -gamma * v
        el_3 = self.eye + (gamma - 1) * (v * v.T) / (desired ** 2)

        upper = torch.cat([gamma, el_1], dim=1)
        lower = torch.cat([el_2, el_3], dim=1)
        boost = torch.cat([upper, lower], dim=0)

        output = torch.matmul(boost, x.transpose(-1, -2)).transpose(-1, -2)

        return output


in_features = 3
out_features = 64
k = 2

a = torch.rand((16, in_features)).to(device)*10 - 5
b = torch.rand((16, in_features)).to(device)*3

manifold = CustomLorentz(k=0.5).to(device)
x = manifold.projx(a)
y = manifold.projx(b)


weight_manifold = Stiefel()

stdv = 1. / math.sqrt(out_features)

weight = torch.rand((out_features - 1, in_features - 1)).uniform_(-stdv, stdv).to("cuda:0")
weight = lib.geoopt.ManifoldParameter(weight_manifold.projx(weight), manifold=weight_manifold)

output = manifold.add_time(torch.matmul(weight, x[..., 1:].T).T)

print(output[..., 0]-x[..., 0])

boost_layer = LorentzPureBoost(manifold=weight_manifold, dim=out_features)
output_boosted = boost_layer(output)

print("break")

