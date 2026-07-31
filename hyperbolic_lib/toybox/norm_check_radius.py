import math

import torch

from lib.geoopt import Lorentz

torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

from lib.lorentz.manifold import CustomLorentz
import lib.geoopt as geoopt


in_features = 1024
out_features = 1024


a = torch.rand((4, in_features)).to(device)*10 - 5
b = torch.rand((4, in_features)).to(device)*3

manifold = CustomLorentz(k=0.2).to(device)
x = manifold.add_time(a)
y = manifold.add_time(b)


print("break")
