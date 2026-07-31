import math

import torch
import torch.nn as nn

MAX_BOOST = 3
MIN_BOOST = 0.01


def fix_scale_layer_output(value):
    output = nn.functional.sigmoid(value)*MAX_BOOST + MIN_BOOST
    return output


class LorentzPureBoost(torch.nn.Module):
    """
         Uses the original mathematical formulation of the Lorentz boost

         Very sensitive initialization
    """

    def __init__(self, manifold, dim, regularize=True):
        super(LorentzPureBoost, self).__init__()

        self.dim = dim

        self.v = nn.Parameter(torch.rand((dim - 1, 1))*0.05)
        #self.v = nn.Parameter(torch.ones((dim - 1, 1)))

        self.eye = nn.Parameter(torch.eye(dim - 1), requires_grad=False)
        self.manifold = manifold

        self.if_regularize = True
        self.reset_parameters()

    def forward(self, x, stabalize=False):

        norm = self.v.norm(2, dim=0, keepdim=False)
        # desired = torch.clamp(norm, max=0.99)
        desired = torch.sigmoid(norm)
        v = self.v * (desired / norm)

        # get boost
        gamma = 1 / torch.sqrt(1 - torch.norm(v) ** 2).reshape(1, -1)
        el_1 = -gamma * v.T
        el_2 = -gamma * v
        el_3 = self.eye + (gamma - 1) * (v * v.T) / (desired ** 2)

        upper = torch.cat([gamma, el_1], dim=1)
        lower = torch.cat([el_2, el_3], dim=1)
        boost = torch.cat([upper, lower], dim=0)

        output = torch.matmul(boost, x.transpose(-1, -2)).transpose(-1, -2)

        #output = self.manifold.projx(output)

        return output

    def reset_parameters(self):
        return


class LorentzBoost(nn.Module):
    """hyperbolic rotation achieved by times A = [cosh\alpha,...,sinh\alpha]
                                                [sinh\alpha,...,cosh\alpha]
    """
    def __init__(self, manifold, init_weight=1):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.FloatTensor([init_weight]))

    def forward(self, x):  # x =[x_0,x_1,...,x_n]
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 2) #x_narrow = [x_1,...,x_n-1]
        x_0 = torch.cosh(self.weight) * x.narrow(-1, 0, 1) + torch.sinh(self.weight) * x.narrow(-1, x.shape[-1] - 1, 1)
        x_n = torch.sinh(self.weight) * x.narrow(-1, 0, 1) + torch.cosh(self.weight) * x.narrow(-1, x.shape[-1] - 1, 1)

        # x_0 = torch.sqrt(self.weight**2 + 1.0) * x_narrow.narrow(-1, 0, 1) + self.weight * x_narrow.narrow(-1, x_narrow.shape[-1] - 1, 1)
        # x_n = self.weight * x_narrow.narrow(-1, 0, 1) + torch.sqrt(self.weight**2 + 1.0) * x_narrow.narrow(-1, x_narrow.shape[-1] - 1, 1)
        x = torch.cat([x_0, x_narrow, x_n], dim=-1)

        return x

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))


class LorentzBoostAlternate(nn.Module):
    """hyperbolic rotation achieved by times A = [cosh\alpha,...,sinh\alpha]
                                                [sinh\alpha,...,cosh\alpha]
    """
    def __init__(self, manifold, in_features, init_weight=1):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Linear(in_features - 1, 1, bias=False)

    def forward(self, x):

        weight = self.weight(x[..., 1:])
        weight = fix_scale_layer_output(weight)
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 2)
        x_0 = torch.cosh(weight) * x.narrow(-1, 0, 1) + torch.sinh(weight) * x.narrow(-1, x.shape[-1] - 1, 1)
        x_n = torch.sinh(weight) * x.narrow(-1, 0, 1) + torch.cosh(weight) * x.narrow(-1, x.shape[-1] - 1, 1)

        x = torch.cat([x_0, x_narrow, x_n], dim=-1)

        return x

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight)


class LorentzBoostScale(nn.Module):
    """hyperbolic rotation achieved by times A = [cosh\alpha,...,sinh\alpha]
                                                [sinh\alpha,...,cosh\alpha]
    """
    def __init__(self, manifold, init_weight=1):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Parameter(torch.FloatTensor([init_weight]))

    def forward(self, x):  # x =[x_0,x_1,...,x_n]
        return self.manifold.scale_hyperbolic_vector(x, self.weight)


class LorentzBoostScaleAlternate(nn.Module):
    """hyperbolic rotation achieved by times A = [cosh\alpha,...,sinh\alpha]
                                                [sinh\alpha,...,cosh\alpha]
    """
    def __init__(self, manifold, in_features, init_weight=1):
        super().__init__()
        self.manifold = manifold
        self.weight = nn.Linear(in_features - 1, 1, bias=False)
        self.reset_parameters()

    def forward(self, x):

        weight = self.weight(x[..., 1:])
        weight = fix_scale_layer_output(weight)

        return self.manifold.scale_hyperbolic_vector(x, weight)

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, a=0, b=0.1)


