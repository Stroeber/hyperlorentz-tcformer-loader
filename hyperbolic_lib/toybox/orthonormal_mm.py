import math

import torch

from lib.geoopt import Lorentz

torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal

from lib.lorentz.manifold import CustomLorentz
import lib.geoopt as geoopt


class LorentzBoost(nn.Module):
    """hyperbolic rotation achieved by times A = [cosh\alpha,...,sinh\alpha]
                                                [sinh\alpha,...,cosh\alpha]
    """
    def __init__(self, manifold, init_weight=1):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.FloatTensor([init_weight]))

    def forward(self, x):  # x =[x_0,x_1,...,x_n]
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 2) #x_narrow = [x_1,...,x_n-1]
        x_0 = torch.cosh(self.weight) * x.narrow(-1, 0, 1) + torch.sinh(self.weight) * x.narrow(-1, x.shape[-1] - 1, 1)
        x_n = torch.sinh(self.weight) * x.narrow(-1, 0, 1) + torch.cosh(self.weight) * x.narrow(-1, x.shape[-1] - 1, 1)

        # x_0 = torch.sqrt(self.weight**2 + 1.0) * x_narrow.narrow(-1, 0, 1) + self.weight * x_narrow.narrow(-1, x_narrow.shape[-1] - 1, 1)
        # x_n = self.weight * x_narrow.narrow(-1, 0, 1) + torch.sqrt(self.weight**2 + 1.0) * x_narrow.narrow(-1, x_narrow.shape[-1] - 1, 1)
        x = torch.cat([x_0, x_narrow, x_n], dim=-1)

        return x

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))


in_features = 32
out_features = 64


a = torch.rand((16, 32, 32, in_features)).to(device)*10 - 5
b = torch.rand((16, 32, 32, in_features)).to(device)*3

manifold = CustomLorentz(k=0.2).to(device)
x = manifold.projx(a)
y = manifold.projx(b)

# stiefel test
tester_manifold = geoopt.Stiefel()
y_stiefel = tester_manifold.projx(y[..., 1:].unsqueeze(-2))

x_t = x[..., 0].unsqueeze(-1)
x_s = x[..., 1:]

norm_old = x_s.norm(dim=-1, keepdim=True)

new_x = torch.nn.functional.relu(x_s)
norm_new = new_x.norm(dim=-1, keepdim=True)

mask = norm_new == 0
norm_new[mask] = 1

new_x = new_x * norm_old/norm_new
x_t[mask] = 1

new_x = torch.concat((x_t, new_x), dim=-1)

layer = LorentzBoost(manifold=manifold).to("cuda:0")

# y_stiefel = torch.nn.functional.pad(y_stiefel, (1,0,1,0))
# y_stiefel[0,0] = 1

manifold.check_point_on_manifold(x)

print("break")
