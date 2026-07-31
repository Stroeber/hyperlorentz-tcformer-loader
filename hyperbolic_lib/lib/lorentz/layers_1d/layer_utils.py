from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic_lib.lib.geoopt.tensor import ManifoldParameter
from hyperbolic_lib.lib.lorentz.layers.linear_layers import LorentzBoost


BOOST_TYPES = {"lorentz": LorentzBoost}


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


class HyperboleIt(nn.Module):
    def forward(self, X):
        d_out, d_in, k1, k2 = X.shape

        if d_out < d_in*k1*k2:
            return X

        X = X.permute(2,3,1,0).reshape(-1, d_out)
        try:
            X = cayley_map(X)
        except:
            print("break")
        return (X.permute(1, 0)
             .reshape(d_out, k1, k2, d_in)
             .permute(0, 3, 1, 2))


class CustomWeightConv2d(nn.Conv2d):
    @staticmethod
    def __new__(cls,
                weight_manifold,
                weight_shape,
                *args,
                **kwargs
                ):
        return super().__new__(cls)

    def __init__(self, weight_manifold, weight_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_manifold = weight_manifold
        self.weight_shape = weight_shape

    def forward(self, input):

        weight = self.weight.reshape(self.weight_shape)

        return F.conv2d(input,
                        weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)


class CustomWeightConv1d(nn.Conv1d):
    @staticmethod
    def __new__(cls,
                weight_manifold,
                weight_shape,
                *args,
                **kwargs
                ):
        return super().__new__(cls)

    def __init__(self, weight_manifold, weight_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_manifold = weight_manifold
        self.weight_shape = weight_shape

        self.bias = None
        r = 1/(self.weight_shape[-1]*self.weight_shape[-2])

        if weight_manifold is not None:
            temp_weight = self.weight_manifold.projx(torch.rand(self.weight_shape).uniform_(-r, r).permute(1, 2, 0).reshape(-1, self.weight_shape[0]))
            self.weight = ManifoldParameter(temp_weight, manifold=self.weight_manifold)
        else:
            self.weight = torch.nn.Parameter(torch.rand(self.weight_shape).uniform_(-r, r).permute(1, 2, 0).reshape(-1, self.weight_shape[0]))

    def forward(self, input):
        out_channels, in_channels, kernel_size = self.weight_shape
        weight = self.weight.permute(1, 0).reshape(out_channels, in_channels, kernel_size)

        return F.conv1d(input,
                        weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)


class CustomWeightTransposeConv1d(nn.ConvTranspose1d):
    @staticmethod
    def __new__(cls,
                weight_manifold,
                weight_shape,
                *args,
                **kwargs
                ):
        return super().__new__(cls)

    def __init__(self, weight_manifold, weight_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_manifold = weight_manifold
        self.weight_shape = weight_shape

        if weight_manifold is not None:
            temp_weight = self.weight_manifold.projx(torch.rand(self.weight_shape).reshape(self.weight_shape[0], -1))
            self.weight = ManifoldParameter(temp_weight, manifold=self.weight_manifold)
        else:
            self.weight = torch.nn.Parameter(torch.rand(self.weight_shape).reshape(self.weight_shape[0], -1))

    def forward(self, input, output_size: Optional[List[int]] = None):

        weight = self.weight.reshape(self.weight_shape)
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        return F.conv_transpose1d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
