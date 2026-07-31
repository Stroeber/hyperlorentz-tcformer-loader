import torch
device = torch.device("cuda")

from torch.nn.utils.parametrizations import orthogonal

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers.linear_layers.FF_betas import LorentzRotation_Up


in_features = 8
out_features = 128
k = 1

a = torch.rand((16, 32, 32, in_features)).to(device)*10 - 5
b = torch.rand((16, 32, 32, in_features)).to(device)*3

manifold = CustomLorentz(k=k).to(device)
x = manifold.projx(a)
y = manifold.projx(b)

euclid_x = manifold.logmap0(x)
euclid_y = manifold.logmap0(y)

conv = torch.nn.Conv2d(in_features-1, out_features-1, kernel_size=3, bias=False).cuda()
rotation_layer = LorentzRotation_Up(manifold, in_features, out_features, if_regularize=False, if_projected=False).cuda()
rotation_layer = orthogonal(rotation_layer, "weight", orthogonal_map="cayley")

rotated_x = rotation_layer(x)
rotated_y = rotation_layer(y)

euclid_rotated_x = manifold.logmap0(rotated_x)
euclid_rotated_y = manifold.logmap0(rotated_y)

norm_euclid_rotated_x = torch.norm(euclid_rotated_x)
norm_euclid_rotated_y = torch.norm(euclid_rotated_y)

norm_euclid_x = torch.norm(euclid_x)
norm_euclid_y = torch.norm(euclid_y)

print("break")
