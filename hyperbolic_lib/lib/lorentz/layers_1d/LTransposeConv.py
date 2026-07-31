import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz
from hyperbolic_lib.lib.geoopt.manifolds import Stiefel
from hyperbolic_lib.lib.lorentz.layers_1d.layer_utils import CustomWeightTransposeConv1d
from hyperbolic_lib.lib.lorentz.layers_1d.LConv import LorentzConv1d
from hyperbolic_lib.lib.lorentz.layers.linear_layers import LorentzBoost


class LorentzTransposeConv1d(nn.Module):
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            rescale_before=False,
            rescale_after=False,
            LFC_normalize=False
    ):
        super(LorentzTransposeConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        padding_implicit = kernel_size - self.padding - 1  # Ensure padding > kernel_size

        self.pad_weight = nn.Parameter(F.pad(torch.ones((self.in_channels,1)),(1,1)), requires_grad=False)

        self.conv = LorentzConv1d(
            manifold=manifold,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding_implicit,
            bias=bias,
            LFC_normalize=LFC_normalize
        )

        self.rescale_before = rescale_before
        self.rescale_after = rescale_after

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x len x C """
        if self.stride > 1:
            # Insert hyperbolic origin vectors between features
            x = x.permute(0,2,1)
            # -> Insert zero vectors
            x = F.conv_transpose2d(x, self.pad_weight,stride=self.stride,padding=1, groups=self.in_channels)
            x = x.permute(0,2,1)
            x[..., 0].clamp_(min=self.manifold.k.sqrt())

        x = self.conv(x)

        if self.output_padding > 0:
            x = F.pad(x, pad=(0, self.output_padding))  # Pad one side of each dimension (right) (see PyTorch documentation)
            x[..., 0].clamp_(min=self.manifold.k.sqrt())  # Fix origin padding

        return x


class LorentzPureTransposeConv1d(nn.Module):
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            rescale_before=False,
            rescale_after=False,
    ):
        super(LorentzPureTransposeConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        padding_implicit = kernel_size - self.padding - 1  # Ensure padding > kernel_size

        self.pad_weight = nn.Parameter(F.pad(torch.ones((self.in_channels,1)),(1,1)), requires_grad=False)

        self.conv = LorentzConv1d(
            manifold=manifold,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding_implicit,
            bias=bias,
        )

        self.rescale_before = rescale_before
        self.rescale_after = rescale_after

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x len x C """
        if self.stride > 1:
            # Insert hyperbolic origin vectors between features
            x = x.permute(0,2,1)
            # -> Insert zero vectors
            x = F.conv_transpose2d(x, self.pad_weight,stride=self.stride,padding=1, groups=self.in_channels)
            x = x.permute(0,2,1)
            x[..., 0].clamp_(min=self.manifold.k.sqrt())

        x = self.conv(x)

        if self.output_padding > 0:
            x = F.pad(x, pad=(0, self.output_padding))  # Pad one side of each dimension (right) (see PyTorch documentation)
            x[..., 0].clamp_(min=self.manifold.k.sqrt())  # Fix origin padding

        return x


class HyperbolicStiefelTransposeConv1D(nn.Module):
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            boost_type="lorentzboost"
    ):
        super(HyperbolicStiefelTransposeConv1D, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.kernel_size = kernel_size

        self.rotation_manifold = Stiefel()

        self.rotate = nn.ConvTranspose1d(in_channels-1,
                                out_channels-1,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False)
        d_out, d_in, n = self.rotate.weight.shape

        self.debug_test = False

        if d_out > d_in * n:

            self.rotate = CustomWeightTransposeConv1d(self.rotation_manifold,
                                             (d_out, d_in, n),
                                             in_channels - 1,
                                             out_channels - 1,
                                             kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             bias=False)

            self.debug_test = True

        self.boost = LorentzBoost(manifold)

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        with torch.no_grad():
            self.rotate.weight.copy_(self.rotation_manifold.projx(self.rotate.weight.data.uniform_(-stdv, stdv)))

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x N x C """

        # restore space_last representation
        out = self.rotate(x[..., 1:].permute(0, 2, 1)).permute(0, 2, 1)
        out = self.manifold.add_time(out)

        out = self.boost(out)
        out = self.manifold.rescale_to_max(out)

        return out


class HyperbolicCayleyTransposeConv1D(nn.Module):

    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            boost_type="lorentzboost"
    ):
        super(HyperbolicCayleyTransposeConv1D, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.kernel_size = kernel_size

        self.rotation_manifold = Stiefel()

        self.rotate = nn.ConvTranspose1d(in_channels - 1,
                                out_channels - 1,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False)
        d_out, d_in, n = self.rotate.weight.shape

        if d_out > d_in * n:
            self.rotate = CustomWeightTransposeConv1d(None,
                                             (d_out, d_in, n),
                                             in_channels - 1,
                                             out_channels - 1,
                                             kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             bias=False)

            torch.nn.utils.parametrizations.orthogonal(self.rotate,
                                                       name='weight',
                                                       orthogonal_map="cayley",
                                                       use_trivialization=True)

        self.boost = LorentzBoost(manifold)

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]))
        with torch.no_grad():
            self.rotate.weight.copy_(self.rotation_manifold.projx(self.rotate.weight.data.uniform_(-stdv, stdv)))

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x N x C """

        # restore space_last representation
        out = self.rotate(x[..., 1:].permute(0, 2, 1)).permute(0, 2, 1)
        out = self.manifold.add_time(out)

        out = self.boost(out)
        out = self.manifold.rescale_to_max(out)

        return out
