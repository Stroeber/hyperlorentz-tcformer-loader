import torch
device = torch.device("cuda")
import torch.nn as nn

import torch.nn.utils.parametrize as parametrize

from lib.geoopt.manifolds import Stiefel
from lib.lorentz.manifold import CustomLorentz


def cayley_map(X: torch.Tensor) -> torch.Tensor:
    n, k = X.size(-2), X.size(-1)
    transposed = n < k
    if transposed:
        X = X.mT
        n, k = k, n
    # Here n > k and X is a tall matrix

    # We just need n x k - k(k-1)/2 parameters
    X = X.tril()
    if n != k:
        # Embed into a square matrix
        X = torch.cat([X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1)
    A = X - X.mH

    # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
    Id = torch.eye(n, dtype=A.dtype, device=A.device)
    Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
    # Q is now orthogonal (or unitary) of size (..., n, n)
    if n != k:
        Q = Q[..., :k]
    # Q is now the size of the X (albeit perhaps transposed)
    # if hasattr(self, "base"):
    #     Q = self.base @ Q
    if transposed:
        Q = Q.mT
    return Q  # type: ignore[possibly-undefined]


class HyperboleIt(nn.Module):
    rotation_manifold = Stiefel()
    def forward(self, X):
        d_out, d_in, k1, k2 = X.shape
        X = X.reshape(d_out, -1)#.permute(2,3,1,0)
        X = self.rotation_manifold.projx(X)
        X = (X#.permute(1, 0)
             .reshape(d_out, d_in, k1, k2))
             #.permute(0, 3, 1, 2))
        return X


in_features = 3
out_features = 64
k = 2

a = torch.rand((16, 32, 32, in_features)).to(device)*10 - 5
b = torch.rand((16, 32, 32, in_features)).to(device)*3

manifold = CustomLorentz(k=1).to(device)
x = manifold.projx(a)
y = manifold.projx(b)


conv = torch.nn.Conv2d(in_features-1, out_features-1, kernel_size=k, bias=False, padding=0).cuda()
# lorentz_conv = LorentzPureConv_transform(manifold, in_features, out_features, 3).cuda()
#
# lorentz_weight = lorentz_conv.linearized_kernel.rotation.weight
#
# patches_pre = lorentz_conv(x)
# time = patches_pre.narrow(-1, 0, 1)
# space = patches_pre.narrow(-1, 1, patches_pre.shape[-1] - 1)
#
# space_ = torch.matmul(space, lorentz_weight)
# out_lorentz = torch.cat([time, space_], dim=-1).reshape(space_.shape[0], 30, 30, space_.shape[-1]+1)

# with torch.no_grad():
#     conv.weight.copy_(lorentz_weight.permute(1, 0).reshape(conv.weight.shape[0],conv.weight.shape[2],conv.weight.shape[3],conv.weight.shape[1]).permute(0,3,1,2))

parametrize.register_parametrization(conv, "weight", HyperboleIt(), unsafe=True)

padder = nn.ZeroPad2d((1, 0, 1, 0))

out = conv(padder((x[..., 1:]).permute(0,3,1,2))).permute(0, 2, 3, 1)
out_projed = manifold.add_time(out)
print("break")
