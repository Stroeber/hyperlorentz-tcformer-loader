import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

from lib.lorentz.manifold import CustomLorentz


a = torch.rand((10, 4)).to(device)*5
b = torch.rand((4)).to(device)*5
c = torch.rand((4)).to(device)*5
manifold = CustomLorentz(k=1).to(device)

x = manifold.projx(a)
p = manifold.projx(b)
z = manifold.projx(c)

mean = manifold.centroid(x)

# Transport batch to origin (center batch)
x_T = manifold.logmap(mean, x)

################################# original method #########################################################
x_T_1 = manifold.transp0back(mean, x_T)

# Compute Fréchet variance
if len(x.shape) == 3:
    var = torch.mean(torch.norm(x_T_1, dim=-1), dim=(0, 1))
else:
    var = torch.mean(torch.norm(x_T_1, dim=-1), dim=0)

# Rescale batch
x_T_1 = x_T_1 * var

# Transport batch to learned mean
x_T_1 = manifold.transp0(z, x_T_1)
#################################### naive method #########################################################



x_T_2 = manifold.transp(mean, z, x_T)

var = torch.mean(manifold.norm(x_T_2, dim=-1), dim=0)
x_T_2 = x_T_2 * var

output = manifold.expmap(z, x_T)

print("break")




