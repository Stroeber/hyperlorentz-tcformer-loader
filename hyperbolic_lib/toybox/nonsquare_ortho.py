import math

import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda")

import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal

from lib.lorentz.manifold import CustomLorentz
import geotorch

in_features = 32
out_features = 16


def reset_parameters(linear):
    stdv = 1. / math.sqrt(out_features)
    step = in_features
    nn.init.uniform_(linear.weight, -stdv, stdv)
    with torch.no_grad():
        for idx in range(0, in_features, step):
            linear.weight[:, idx] = 0
    return linear

seed = 453
torch.manual_seed(seed)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
# Rotation

linear = nn.Linear(in_features - 1, out_features - 1, bias=False).to(device)
linear.weight = torch.nn.Parameter(linear.weight*121)
linear = orthogonal(linear, "weight", orthogonal_map="cayley")

#linear = reset_parameters(linear)

a = torch.rand(in_features).to(device)*5

manifold = CustomLorentz(k=1).to(device)
x = manifold.projx(a)

x_0 = x.narrow(-1, 0, 1)
x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)

x_ = linear(x_narrow)
x = torch.cat([x_0, x_], dim=-1)

tester_manifold = geotorch.Stiefel((out_features-1, in_features-1))
sampled = tester_manifold.sample("torus")

x_ = torch.matmul(x_narrow.unsqueeze(0),sampled.to(device).T)
x = torch.cat([x_0, -x_.squeeze()], dim=-1)


print("break")
