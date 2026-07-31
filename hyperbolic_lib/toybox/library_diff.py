import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

from lib.geoopt.manifolds.lorentz import math
from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzMLR


def expmap0_lorentz(c: torch.Tensor, x: torch.tensor) -> torch.Tensor:
    x_norm = x.norm(dim=1, keepdim=True)
    x0 = torch.cosh(c.sqrt() * x_norm) * c.sqrt()
    xr = torch.sinh(c.sqrt() * x_norm) * x / (c.sqrt() * x_norm + 1e-8)
    mapped_x = torch.cat([x0, xr], dim=-1)
    return mapped_x


def _expmap0(u, k: torch.Tensor, dim: int = -1):
    nomin = math._norm(u, keepdim=True, dim=dim)
    u = u / nomin
    nomin = (nomin / torch.sqrt(k))
    l_v = torch.cosh(nomin) * torch.sqrt(k)
    r_v = torch.sqrt(k) * torch.sinh(nomin) * u
    dn = r_v.size(dim) - 1
    p = torch.cat((l_v + r_v.narrow(dim, 0, 1), r_v.narrow(dim, 1, dn)), dim)
    return p


k = torch.tensor([1]).to(torch.float64).to(device)

a = torch.rand((16, 32)).to(device)*3
a[..., 0] = 0

manifold = CustomLorentz(k=k)

a = manifold.logmap0(manifold.projx(a))

default = _expmap0(a, k)
new = expmap0_lorentz(k, a[..., 1:])

print("break")

#####


class HyperbolicLorentz(nn.Module):
    """ lorentz hyperbolic layer """

    def __init__(self, in_dim, out_dim, manifold, nl) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.manifold = manifold
        self.c = self.manifold.k  # curvature of the lorentz model
        self.nl = nl

        # bn
        self.bn = nn.BatchNorm1d(out_dim)

        # init
        self.z = torch.normal(0, 0.5, (in_dim, out_dim)).to(device)
        self.a = torch.normal(0, 0.5, (1, out_dim)).to(device)

    # aux
    def get_w(self, z, a):
        """ get tangent space vectors from euclidean ones, by a expmap at 0 followed by a parallel transport (similar in PP) """
        z_norm = z.norm(dim=0, keepdim=True)
        w0 = z_norm * torch.sinh(a / self.c.sqrt())
        wr = torch.cosh(a / self.c.sqrt()) * z
        w = torch.cat([w0, wr], dim=0)
        return w

    def lorentz_dist2plane(self, X):
        """
        vectorized lorentz dist2plane

        arcsinh(-<w, x>_L / (||z||_2 * sqrt(c)) * ||z||_2
        adapted from https://proceedings.mlr.press/v89/cho19a.html, and a thorough discussiong with Zhengchao
        """

        z_norm = self.z.norm(dim=0, keepdim=True)
        w0 = z_norm * torch.sinh(self.a / self.manifold.k.sqrt())
        wr = torch.cosh(self.a / self.manifold.k.sqrt()) * self.z
        W = torch.cat([w0, wr], dim=0)

        numerator = - X.narrow(-1, 0, 1) @ W[[0]] + X.narrow(-1, 1, self.in_dim) @ W[1:]
        denom = z_norm * self.manifold.k.sqrt()
        distance = torch.arcsinh(numerator / denom) * z_norm
        logits = torch.sign(numerator) * denom * torch.abs(distance)
        return logits


a = torch.rand((16, 32)).to(device)*3

manifold = CustomLorentz(k=k).to(device)

x = manifold.projx(a)

mlr_default = LorentzMLR(manifold, 31, 10).to(device)
logits, distances = mlr_default(a, return_distance=True)

mlr_new = HyperbolicLorentz(31, 10, manifold, None).to(device)

mlr_new.a = mlr_default.a.unsqueeze(0)
mlr_new.z = mlr_default.z.T

W = mlr_new.get_w(mlr_new.z, mlr_new.a)
distances_new = mlr_new.lorentz_dist2plane(x)

print("break")







