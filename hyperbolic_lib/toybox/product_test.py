import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

from manifolds import Lorentz

a = torch.rand((2, 3)).to(device)*10
b = torch.rand((2, 3)).to(device)*15

manifold = Lorentz(k=1).to(device)

a_h = manifold.projx(a)
b_h = manifold.projx(b)

c_1 = manifold.inner(None, a[0, :], b[0, :])
c_2 = manifold.inner(None, a[1, :], b[0, :])
c_3 = manifold.inner(None, a[0, :], b[1, :])
c_4 = manifold.inner(None, a[1, :], b[1, :])

print("break")
