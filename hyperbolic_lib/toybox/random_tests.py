import torch

import lib.geoopt
from hier.hyptorch.pmath import expmap0
from lib.lorentz.manifold import CustomLorentz

from math import cosh, sinh, acosh, asinh

torch.set_default_dtype(torch.float64)

x = torch.rand((8, 32, 32, 4)).to(device="cuda:0")*20
y = torch.rand((4)).to(device="cuda:0")*2

manifold = CustomLorentz(k=4).cuda()
origin = manifold.origin(y.shape).to(device="cuda:0")

x_man = manifold.projx(x)
y_man = manifold.projx(y)


y_tangent = manifold.logmap0(y_man)

factor = torch.tensor([4,3,2,1]).cuda()

y = manifold.expmap0(y_tangent)
y_scaled = manifold.expmap0(factor * y_tangent)

# x_test = x_tangent * 3
#std = torch.rand(x_test.shape, device=x_test.device)

#x_test_norm = manifold.norm(x_test, keepdim=True)

#x_Lnorm = manifold.norm(x_tangent, keepdim=True)
#y_Lnorm = manifold.norm(y_tangent, keepdim=True)

#manual_map_x = torch.cosh(x_Lnorm/torch.sqrt(manifold.k))*origin + torch.sqrt(manifold.k) * torch.sinh(x_Lnorm/torch.sqrt(manifold.k))*(x_tangent/x_Lnorm)
norm=y_tangent.norm()
norm=manifold.dist0(y)/2
scaler = (torch.exp(factor*norm)-torch.exp(-factor*norm))/(torch.exp(norm)-torch.exp(-norm))

print("break")

