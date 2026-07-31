# Lorentzian neural network layers
import torch.nn as nn
from ..lorentz.layers.linear_layers.FF_betas import LorentzProjection
from ..lorentz.layers import LorentzAct


class FullLorentzGraphNeuralNetwork(nn.Module):
    def __init__(self, manifold, in_feature, out_features, c_in, act):
        super(FullLorentzGraphNeuralNetwork, self).__init__()
        self.c_in = c_in
        self.linear = LorentzProjection(manifold, in_feature, out_features)
        self.lorentz_act = LorentzAct(act, manifold)

    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        h = self.manifold.centroid(h, adj)
        h = self.lorentz_act(h)
        output = h, adj
        return output

    def reset_parameters(self):
        self.linear.reset_parameters()

