import math

import torch
import torch.nn as nn

from torch.nn.utils.parametrizations import orthogonal

from ...manifold import CustomLorentz

class LorentzTransform(torch.nn.Module):
    def __init__(self, manifold, dim, mode="boost", regularize=True):
        super(LorentzTransform, self).__init__()

        self.dim = dim

        self.boost = self.rotate = False

        if mode == "both" or mode == "boost":
            self.v = nn.Parameter(torch.rand((dim - 1, 1)))
            self.boost = True

        if mode == "both" or mode == "rotate":
            self.rotation_weight = nn.Parameter(torch.rand((dim - 1, dim - 1)))
            self.rotate = True

        self.eye = nn.Parameter(torch.eye(dim - 1), requires_grad=False)
        self.manifold = manifold
        self.if_regularize = True
        self.reset_parameters()

    def forward(self, x, stabalize=False):

        if self.boost:
            norm = self.v.norm(2, dim=0, keepdim=False)
            # desired = torch.clamp(norm, max=0.99)
            desired = torch.sigmoid(norm/2)
            v = self.v * (desired / norm)

            # get boost
            gamma = 1 / torch.sqrt(1 - torch.norm(v) ** 2).reshape(1, -1)
            el_1 = -gamma * v.T
            el_2 = -gamma * v
            el_3 = self.eye + (gamma - 1) * (v * v.T) / (v.norm(2, dim=0, keepdim=True) ** 2)

            upper = torch.cat([gamma, el_1], dim=1)
            lower = torch.cat([el_2, el_3], dim=1)
            boost = torch.cat([upper, lower], dim=0)

        # get rotation
        if self.rotate:
            rotation = torch.nn.functional.pad(self.rotation_weight, (1, 0, 1, 0))
            rotation[..., 0, 0] = 1


        if self.rotate and self.boost:
            output = torch.matmul(torch.matmul(rotation, boost), x.transpose(-1, -2)).transpose(-1, -2)
        elif self.rotate:
            output = torch.matmul(rotation, x.transpose(-1, -2)).transpose(-1, -2)
        elif self.boost:
            output = torch.matmul(boost, x.transpose(-1, -2)).transpose(-1, -2)

        if stabalize:
            output = self.manifold.logmap0(output)
            norm = output[..., 1:].norm(2, dim=-1, keepdim=True)
            desired = torch.clamp(norm, max=10)

            output = output[..., 1:] * (desired / norm)
            output = self.manifold.add_time(output)

            output = self.manifold.expmap0(output)

        if self.if_regularize is True:
            output = self.manifold.regularize(x)

        return output

    def reset_parameters(self):
        return
        # nn.init.kaiming_normal_(self.v)
        # nn.init.orthogonal_(self.rotation_weight)


class LorentzFullyConnected_transform(nn.Module):
    """
        Modified Lorentz fully connected layer of Chen et al. (2022).

        Code modified from https://github.com/chenweize1998/fully-hyperbolic-nn

        args:
            manifold: Instance of Lorentz manifold
            in_features, out_features, bias: Same as nn.Linear
            init_scale: Scale parameter for internal normalization
            learn_scale: If scale parameter should be learnable
            normalize: If internal normalization should be applied
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_features,
            out_features,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False
    ):
        super(LorentzFullyConnected_transform, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features - 1, self.out_features - 1, bias=bias)

        self.init_std = 0.02
        self.reset_parameters()

        self.shape_matrix = nn.Parameter(torch.ones((in_features, out_features)), requires_grad=False)

        self.transform = LorentzTransform(self.manifold, out_features)
        self.transform = orthogonal(self.transform, name="rotation_weight")

    def forward(self, x):

        if self.out_features != self.in_features:
            x_space = self.weight(x[..., 1:])
            x = self.manifold.add_time(x_space)

        return self.transform(x)

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)




