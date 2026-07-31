import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

from lib.lorentz.manifold import CustomLorentz


a = torch.rand((4)).to(device)*5
b = torch.rand((4)).to(device)*5
c = torch.rand((4)).to(device)*5
manifold = CustomLorentz(k=1).to(device)

x = manifold.projx(a)
p = manifold.projx(b)
z = manifold.projx(c)

inner_factored = manifold.inner(None, x, p+z)
inner_not = manifold.inner(None, x, p) + manifold.inner(None, x, z)

origin = manifold.origin(x.shape, device=x.device)

denom = manifold.k-manifold.inner0(x)
nom = origin+x


tangent_z_x = manifold.logmap(x, z)

zt_to_origin = manifold.transp0back(x, tangent_z_x)
scaled = torch.tensor((0, 1, 2, 2), device=x.device)*zt_to_origin
zt_x = manifold.transp0(x, scaled)

alternate = torch.tensor((1, 1, 2, 2),device=x.device) * (tangent_z_x + manifold.inner(None, x, tangent_z_x))*(nom/denom)

d0_p = manifold.dist0(p)

p_h = manifold.logmap0(p)
p_h = manifold.transp0(x, p_h)
p2 = manifold.expmap(x, p_h)

dx_p2 = manifold.dist(x, p2)

print("break")




