import torch

from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz

torch.set_default_dtype(torch.float64)

k1 = 4
k2 = 3

x = torch.rand((8, 32, 32, 4)).to(device="cuda:0")*20
y = torch.rand((4)).to(device="cuda:0")*2

manifold = CustomLorentz(k=4).cuda()
manifold_2 = CustomLorentz(k=3).cuda()

k1 = manifold.k
k2 = manifold_2.k

x_man = manifold.projx(x)
y_man = manifold.projx(y)

homothety_factor = k2.sqrt()/k1.sqrt()

x_man_2 = x_man*homothety_factor
print("break")
