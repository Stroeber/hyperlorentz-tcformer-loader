import math

import torch
from torch import nn
import torch.nn.functional as F

from hyperbolic_lib.lib.geoopt import Stiefel, ManifoldParameter
from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz
from hyperbolic_lib.lib.lorentz.layers import LorentzBoostScale, LorentzGlobalAvgPool2d
from hyperbolic_lib.lib.lorentz.layers import LorentzReLU

from hyperbolic_lib.lib.lorentz.layers_1d.LConv import (
    LorentzPureConv1d,
    LorentzConv1d,
    HyperbolicCayleyConv1D,
    HyperbolicStiefelConv1D,
    LorentzLoRAConv1d,
)
from hyperbolic_lib.lib.lorentz.layers.LBnorm import (
    LorentzBatchNorm1d,
    LorentzBatchNorm1dLVar,
    LorentzBatchNorm2d,
    LorentzBatchNorm2dLVar
)
from hyperbolic_lib.lib.lorentz.layers.BN_betas import (
    LorentzBatchNorm1d_allVar,
    LorentzBatchNorm2d_allvar,
)
from hyperbolic_lib.lib.lorentz.layers.LModules import (QuickDirtyMaxPool,
                                                        LorentzMaxPool1D,
                                                        LorentzMaxPool2D,
                                                        LorentzAvgPool1d)
from hyperbolic_lib.lib.lorentz.layers import LorentzConv2d, LorentzPureConv
from hyperbolic_lib.lib.lorentz.layers_1d.LTransposeConv import (
    LorentzPureTransposeConv1d,
    LorentzTransposeConv1d,
    HyperbolicCayleyTransposeConv1D,
    HyperbolicStiefelTransposeConv1D,
)


class QuickDirtyLayerNorm(nn.Module):
    def __init__(self, manifold, dim):
        super(QuickDirtyLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(dim-1)
        self.manifold = manifold

    def forward(self, x):
        output = self.layer_norm(x[..., 1:])
        return self.manifold.add_time(output)


CONV1D_TYPES = {
    "original": LorentzConv1d,
    "pure": LorentzPureConv1d,
    "cayley": HyperbolicCayleyConv1D,
    "stiefel": HyperbolicStiefelConv1D,
    "LoRA": LorentzLoRAConv1d,
}
CONV2D_TYPES = {
    "original": LorentzConv2d,
    "pure": LorentzPureConv,
}

TRANSPOSE_CONV_TYPES = {
    "original": LorentzTransposeConv1d,
    "pure": LorentzPureTransposeConv1d,
    "cayley": HyperbolicCayleyTransposeConv1D,
    "stiefel": HyperbolicStiefelTransposeConv1D,
}

BATCH1D_TYPES = {
    "original": LorentzBatchNorm1d,
    "all_var": LorentzBatchNorm1d_allVar,
    "layer": QuickDirtyLayerNorm
}

BATCH2D_TYPES = {
    "original": LorentzBatchNorm2d,
    "all_var": LorentzBatchNorm2d_allvar,
    "layer": QuickDirtyLayerNorm
}

POOL1D_TYPES = {
    "dirty": QuickDirtyMaxPool,
    "clean": LorentzMaxPool1D,
    "average": LorentzAvgPool1d
}

POOL2D_TYPES = {
    "dirty": QuickDirtyMaxPool,
    "clean": LorentzMaxPool2D,
}


class ToHyperbolic(nn.Module):
    def __init__(self, manifold, norm=False, tangent_based=False):
        super(ToHyperbolic, self).__init__()
        self.manifold = manifold

        self.norm = nn.Sequential() if not norm else self.manifold.rescale_to_max
        self.euclid_norm = nn.Sequential() if not norm else self.manifold.rescale_to_max_euclid

        self.to_hyperbolic = self.project_tangents if tangent_based else self.project_basic

    def project_tangents(self, x):
        x = nn.functional.pad(x, (1, 0, 0, 0))
        x = self.euclid_norm(x)
        return self.manifold.expmap0(x)

    def project_basic(self, x):
        x = self.manifold.add_time(x)
        return self.norm(x)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.to_hyperbolic(x)


def patch_len(n, epochs):
    list_len = []
    base = n // epochs
    for i in range(epochs):
        list_len.append(base)
    for i in range(n - base * epochs):
        list_len[i] += 1

    if sum(list_len) == n:
        return list_len
    else:
        return ValueError('check your epochs and axis should be split again')


def matt_covar(x):
    vects = []
    for t in x:
        t = t.squeeze()

        mean = t.mean(dim=-1).unsqueeze(-1).repeat(1, 1, t.shape[-1])
        t = t - mean

        cov = t @ t.permute(0, 2, 1)
        cov = cov / (t.shape[-1] - 1)
        vects.append(cov)

    return torch.stack(vects)


def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    @staticmethod
    def __new__(cls,
                manifold,
                *args,
                **kwargs
                ):
                return super().__new__(cls)

    def __init__(self, manifold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifold = manifold
        self.weight_manifold = Stiefel()
        d_out, d_in, k1 = self.weight.shape
        self.orig_shape = self.weight.shape

        if d_out > d_in * k1:
            temp_weight = self.weight_manifold.projx(
                torch.rand(self.orig_shape).reshape(self.orig_shape[0], -1) * 10)
            self.weight = ManifoldParameter(temp_weight, manifold=self.weight_manifold)

    def forward(self, input):
        return conv1d_same_padding(
            input, self.weight, self.bias, self.stride, self.dilation, self.groups, self.orig_shape
        )


def conv1d_same_padding(input, weight, bias, stride, dilation, groups, orig_shape):
    # stride and dilation are expected to be tuples.

    d_out, d_in, k1 = orig_shape
    if d_out > d_in * k1:
       weight = weight.reshape(d_out, d_in, k1)

    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = ((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])


    return F.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding // 2,
        dilation=dilation,
        groups=groups,
    )


class Conv1dHyper(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    @staticmethod
    def __new__(cls,
                manifold,
                *args,
                **kwargs
                ):
                return super().__new__(cls)

    def __init__(self, manifold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifold = manifold
        self.weight_manifold = Stiefel()
        d_out, d_in, k1 = self.weight.shape

        self.orig_shape = self.weight.shape

        if d_out > d_in * k1:
            temp_weight = self.weight_manifold.projx(
                torch.rand(self.weight.shape).reshape(self.weight.shape[0], -1) * 10)
            self.weight = ManifoldParameter(temp_weight, manifold=self.weight_manifold)

    def forward(self, input):
        return conv1d_Hyper(
            input, self.weight, self.bias, self.stride, self.dilation, self.groups, self.padding, orig_shape=self.orig_shape
        )


def conv1d_Hyper(input, weight, bias, stride, dilation, groups, padding, orig_shape):
    # stride and dilation are expected to be tuples.

    d_out, d_in, k1 = orig_shape

    if d_out > d_in * k1:
       weight = weight.reshape(d_out, d_in, k1)

    return F.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def cayley_map(X: torch.Tensor) -> torch.Tensor:
    n, k = X.size(-2), X.size(-1)
    transposed = n < k
    if transposed:
        X = X.mT
        n, k = k, n
    # Here n > k and X is a tall matrix

    # We just need n x k - k(k-1)/2 parameters
    X = X.tril()
    if n != k:
        # Embed into a square matrix
        X = torch.cat([X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1)
    A = X - X.mH

    # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
    Id = torch.eye(n, dtype=A.dtype, device=A.device)
    Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
    # Q is now orthogonal (or unitary) of size (..., n, n)
    if n != k:
        Q = Q[..., :k]
    # Q is now the size of the X (albeit perhaps transposed)
    if transposed:
        Q = Q.mT
    return Q  # type: ignore[possibly-undefined]


class Hyperbolic1DConv(nn.Module):
    """Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC
    """

    def __init__(
        self,
        manifold: CustomLorentz,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        bias=False,
        LFC_normalize=False,
    ):
        super(Hyperbolic1DConv, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        if padding=="same":
            self.rotate = Conv1dSamePadding(
                manifold=self.manifold,
                in_channels=in_channels-1,
                out_channels=out_channels-1,
                kernel_size=kernel_size,
                stride=stride,
            )
        else:
            self.rotate = Conv1dHyper(
                manifold=self.manifold,
                in_channels=in_channels-1,
                out_channels=out_channels-1,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )

        #self.boost = LorentzBoost(manifold, init_weight=1)
        self.boost = LorentzBoostScale(manifold, init_weight=1)

        # parametrize.register_parametrization(self.rotate, "weight", HyperboleIt())
        # old_shape = self.rotate.weight.shape
        # parametrize.register_parametrization(self.rotate, "weight", shape_to(), unsafe=True)
        # orthogonal(self.rotate, "weight", orthogonal_map="cayley")
        # parametrize.register_parametrization(self.rotate, "weight", shape_back(old_shape), unsafe=True)

        # self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(
            2.0 / ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1])
        )
        # self.linearized_kernel.w = ManifoldParameter(self.manifold.projx(self.linearized_kernel.w.data.uniform_(-stdv, stdv)))
        # if self.bias:
        #    self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """x has to be in channel-last representation -> Shape = bs x H x W x C"""
        out = self.rotate(x[..., 1:].permute(0,2,1)).permute(0,2,1)
        #out = self.manifold.rescale_to_max(out)
        out = self.manifold.add_time(out)

        out = self.boost(out)

        # out = self.manifold.rescale_to_max(out)
        #out = self.manifold.logmap0(out)
        out = self.manifold.rescale_to_max(out)
        #out = self.manifold.add_time(out)

        return out


class Hyperbolic1DConvBlock(nn.Module):
    def __init__(
        self,
        manifold,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding,
        act=True
    ) -> None:
        super().__init__()

        if act:
            self.layers = nn.Sequential(
                Hyperbolic1DConv(
                    manifold=manifold,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                LorentzBatchNorm1d(manifold, num_features=out_channels, norm_moment=0.1),
                LorentzReLU(manifold),
            )
        else:
            self.layers = nn.Sequential(
                Hyperbolic1DConv(
                    manifold=manifold,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                LorentzBatchNorm1d(manifold, num_features=out_channels, norm_moment=0.1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x)


class HyperbolicResNetBlock(nn.Module):
    def __init__(self, manifold, in_channels: int, out_channels: int, stride: int = 1, padding="same") -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.manifold = manifold

        self.conv1 = Hyperbolic1DConvBlock(
                    manifold,
                    in_channels=channels[0],
                    out_channels=channels[1],
                    kernel_size=kernel_sizes[0],
                    stride=stride,
                    padding=padding,
                )
        self.conv2 = Hyperbolic1DConvBlock(
                    manifold,
                    in_channels=channels[1],
                    out_channels=channels[2],
                    kernel_size=kernel_sizes[1],
                    stride=1,
                    padding="same"
                )

        self.conv3 = Hyperbolic1DConvBlock(
                    manifold,
                    in_channels=channels[2],
                    out_channels=channels[3],
                    kernel_size=kernel_sizes[2],
                    stride=1,
                    #act=False,
                    padding="same"
                )


        self.activation = LorentzReLU(manifold)

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True

            self.residual = nn.Sequential(
                *[
                    Hyperbolic1DConv(
                        manifold,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride,
                        padding="same"
                    ),
                    LorentzBatchNorm1d(
                        manifold, num_features=out_channels, norm_moment=0.1
                    ),
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.match_channels:
            out = self.manifold.add_time(out[..., 1:] + self.residual(x)[..., 1:])

        return out #self.activation(out)