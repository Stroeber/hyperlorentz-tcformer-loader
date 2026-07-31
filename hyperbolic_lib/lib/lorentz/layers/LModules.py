import torch
import torch.nn as nn
import unfoldNd

from ..manifold import CustomLorentz


class ManifoldSwapper1D(nn.Module):

    def __init__(self, manifold=None, manifold_2=None, to_euclidean=False, from_euclidean=False, space_only=False):
        super(ManifoldSwapper1D, self).__init__()

        self.manifold = manifold
        self.manifold_2 = manifold_2
        self.to_euclidean = to_euclidean
        self.from_euclidean = from_euclidean
        self.space_only = space_only

    def forward(self, x):

        if self.to_euclidean:
            if self.space_only:
                return x[..., 1:]
            return self.manifold.logmap0(x)[..., 1:]

        if self.from_euclidean:
            x = self.manifold_2.rescale_to_max_euclid(x)
            return self.manifold_2.add_time(x)

        if self.space_only:
            return self.manifold_2.projx(x)

        x = self.manifold.logmap0(x)
        x = self.manifold_2.rescale_to_max_euclid(x)

        return self.manifold_2.expmap0(x)


class LorentzAct(nn.Module):
    """ Implementation of a general Lorentz Activation on space components. 
    """
    def __init__(self, activation, manifold: CustomLorentz):
        super(LorentzAct, self).__init__()
        self.manifold = manifold
        self.activation = activation  # e.g. torch.relu

    def forward(self, x):
        if type(x) == tuple:
            return self.manifold.lorentz_activation(x[0], self.activation), x[1]
        else:
            return self.manifold.lorentz_activation(x, self.activation)


class LorentzLearnedNorm(nn.Module):
    """ Implementation of a general Lorentz Activation on space components.
    """
    def __init__(self, manifold: CustomLorentz):
        super(LorentzLearnedNorm, self).__init__()
        self.manifold = manifold
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        sq_norm = torch.abs(self.minkowski_dot(x, x, keepdim=False)).clamp(min=1e-2)
        real_norm = torch.sqrt(torch.abs(sq_norm))
        projected_point = torch.einsum("...i,...->...i", x, self.k * self.scale * real_norm)
        return projected_point
    

class LorentzReLU(nn.Module):
    """ Implementation of Lorentz ReLU Activation on space components. 
    """
    def __init__(self, manifold: CustomLorentz):
        super(LorentzReLU, self).__init__()
        self.manifold = manifold

    def forward(self, x):
        return self.manifold.lorentz_relu(x)


class LorentzGlobalAvgPool2d(nn.Module):
    """ Implementation of a Lorentz Global Average Pooling based on Lorentz centroid defintion. 
    """
    def __init__(self, manifold: CustomLorentz, w=None, keep_dim=False, last_dim=None):
        super(LorentzGlobalAvgPool2d, self).__init__()

        self.manifold = manifold
        self.keep_dim = keep_dim
        self.w = nn.Parameter(torch.ones(w)) if w is not None else None

        self.lin = torch.nn.Linear(last_dim, 1) if last_dim is not None else None

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.reshape(bs, -1, c)

        if self.lin is not None:
            self.w = torch.nn.functional.softmax(self.lin(x[..., 1:]).squeeze(), dim=-1)

        if self.w is not None:
            x = self.manifold.centroid(x, self.w.unsqueeze(-2)).squeeze()

        x = self.manifold.centroid(x).squeeze()

        if self.keep_dim:
            x = x.view(bs, 1, 1, c)

        return x


class LorentzAvgPool1d(nn.Module):
    def __init__(self, manifold: CustomLorentz, kernel_size, padding=1, stride=1):
        super(LorentzAvgPool1d, self).__init__()

        self.manifold = manifold

        self.unfold = unfoldNd.UnfoldNd(
            (kernel_size), padding=padding, stride=stride
        )

    def forward(self, x):
        bs, n, c = x.shape
        x = x.permute(0, 2, 1)

        # (B, CxW, N')
        unfolded = self.unfold(x)
        unfolded = unfolded.reshape(bs, c, -1, unfolded.shape[-1]).permute(0, 3, 2, 1)
        out = self.manifold.centroid(unfolded).squeeze()

        return out


class QuickDirtyMaxPool(nn.Module):
    def __init__(self, manifold, pool_layer):
        super(QuickDirtyMaxPool, self).__init__()

        self.manifold = manifold
        self.pool_layer = pool_layer

    def forward(self, x, return_indices=False):
        x_temp = x[..., 1:].permute(0, 2, 1)
        if return_indices:
            z_maxpool, indices = self.pool_layer(x_temp)
        else:
            z_maxpool = self.pool_layer(x_temp)
        z_maxpool = self.manifold.add_time(z_maxpool.permute(0, 2, 1))
        # z_maxpool = self.manifold.rescale_to_max(z_maxpool)

        if return_indices:
            return z_maxpool, indices
        return z_maxpool


class LorentzMaxPool2D(nn.Module):
    def __init__(self,
                 manifold,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1):
        super(LorentzMaxPool2D, self).__init__()

        self.manifold = manifold
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                    return_indices=True)

    def forward(self, x):
        distances = self.manifold.dist0(x, keepdim=True).permute(0, 3, 1, 2)

        pooled, indices = self.maxpool(distances)

        # Get flat indices of norms before pooling
        B, _, H_out, W_out = pooled.shape
        unpooled_shape = distances.shape  # (B, 1, H, W)
        _, _, H, W = unpooled_shape

        # Prepare to gather from x
        x_reshaped = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x_flat = x_reshaped.view(B, x.shape[-1], -1)  # (B, C, H*W)

        # Indices from maxpool are w.r.t. (H*W), so we use them directly
        indices_flat = indices.view(B, -1)  # (B, H'*W')

        # Gather values from original tensor
        gathered = torch.gather(x_flat, dim=2,
                                index=indices_flat.unsqueeze(1).expand(-1, x.shape[-1], -1))  # (B, C, H'*W')

        # Reshape if needed to (B, H', W', C)
        output = gathered.permute(0, 2, 1).view(B, H_out, W_out, -1)  # (B, H', W', C)

        return output


class LorentzMaxPool1D(nn.Module):
    def __init__(self,
                 manifold,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1):
        super(LorentzMaxPool1D, self).__init__()

        self.manifold = manifold
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                    return_indices=True)

    def forward(self, x):
        distances = self.manifold.dist0(x, keepdim=True).permute(0, 2, 1)

        pooled, indices = self.maxpool(distances)

        # Get flat indices of norms before pooling
        B, _, L = pooled.shape
        unpooled_shape = distances.shape  # (B, 1, L)
        _, _, L = unpooled_shape

        # Prepare to gather from x
        x_flat = x.permute(0, 2, 1)

        # Indices from maxpool are w.r.t. (L), so we use them directly
        indices_flat = indices.view(B, -1)  # (B, L')

        # Gather values from original tensor
        gathered = torch.gather(x_flat, dim=2,
                                index=indices_flat.unsqueeze(1).expand(-1, x.shape[-1], -1))  # (B, C, L')

        # Reshape if needed to (B, L', C)
        output = gathered.permute(0, 2, 1)

        return output

class LorentzMaxPool1D(nn.Module):
    def __init__(self,
                 manifold,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1):
        super(LorentzMaxPool1D, self).__init__()

        self.manifold = manifold
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                    return_indices=True)

    def forward(self, x):
        distances = self.manifold.dist0(x, keepdim=True).permute(0, 2, 1)

        pooled, indices = self.maxpool(distances)

        # Get flat indices of norms before pooling
        B, _, L = pooled.shape
        unpooled_shape = distances.shape  # (B, 1, L)
        _, _, L = unpooled_shape

        # Prepare to gather from x
        x_flat = x.permute(0, 2, 1)

        # Indices from maxpool are w.r.t. (L), so we use them directly
        indices_flat = indices.view(B, -1)  # (B, L')

        # Gather values from original tensor
        gathered = torch.gather(x_flat, dim=2,
                                index=indices_flat.unsqueeze(1).expand(-1, x.shape[-1], -1))  # (B, C, L')

        # Reshape if needed to (B, L', C)
        output = gathered.permute(0, 2, 1)

        return output

