import math

import torch
import torch.nn as nn


class LorentzRotationFixedNorm(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 if_regularize=False,
                 if_projected=False
                 ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features-1, self.out_features-1, bias=False)
        self.reset_parameters()
        self.if_regularize = if_regularize
        self.if_projected = if_projected

    def forward(self, x):

        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)

        old_norms = torch.norm(x_narrow, p=2, dim=-1, keepdim=True)

        x_ = self.linear(x_narrow)

        new_norms = torch.norm(x_, p=2, dim=-1, keepdim=True).clone()
        new_norms[new_norms == 0] = 1

        x_ = old_norms * x_/new_norms

        x = self.manifold.add_time(x_)
        if self.if_regularize is True:
            x = self.manifold.rescale_to_max(x)

        if self.if_projected is True:
            x = self.manifold.projx(x)

        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.linear.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.linear.weight[:, idx] = 0


class LorentzRotationUp(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 if_regularize=False,
                 if_projected=False
                 ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features

        stdv = 1. / math.sqrt(self.out_features)

        weight = torch.rand((self.out_features-1, self.in_features-1)).uniform_(-stdv, stdv)
        self.weight = torch.nn.Parameter(weight)

        self.if_regularize = if_regularize
        self.if_projected = if_projected

    def forward(self, x):

        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)

        x_ = torch.matmul(self.weight, x_narrow.transpose(-1, -2)).transpose(-1, -2)
        x = self.manifold.add_time(x_)

        if self.if_regularize is True:
            x = self.manifold.rescale_to_max(x)

        if self.if_projected is True:
            x = self.manifold.projx(x)

        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight[:, idx] = 0

