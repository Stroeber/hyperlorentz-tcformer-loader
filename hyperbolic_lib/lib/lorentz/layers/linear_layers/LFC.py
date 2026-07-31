import torch
import torch.nn as nn

from ...manifold import CustomLorentz
from ..LMLR import LorentzMLR
from ..cayley_layers import CayleyLinear
from .rotations import LorentzRotationUp, LorentzRotationFixedNorm
from hyperbolic_lib.lib.geoopt import ManifoldParameter
import torch.nn.functional as F

from .boosts import (LorentzBoost,
                     LorentzBoostAlternate,
                     LorentzBoostScale,
                     LorentzBoostScaleAlternate,
                     LorentzPureBoost)


class LorentzFullyConnected(nn.Module):
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
            normalize=False,
            activation=None,
            dropout=0.0,
            nheads=1,
        ):
        super(LorentzFullyConnected, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features - 1, self.out_features - 1, bias=bias)
        self.nheads = nheads
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.init_std = 0.02
        self.reset_parameters()

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

    def forward(self, x):

        # changed the transformation to only include space values
        # need to check if the vastly different scale of the time element negatively affects results

        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        x_space = self.weight(x_space)

        if self.activation is not None:
            x_space = self.activation(x_space)
        x_space = self.dropout(x_space)


        if self.nheads>1:
            # Lorentz direct split
            x_space = x_space.view(x_space.size(0), x_space.size(1), self.nheads, self.out_features//self.nheads).transpose(1,2)

        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

            mask = square_norm <= 1e-10

            square_norm[mask] = 1
            unit_length = x_space/torch.sqrt(square_norm)
            x_space = scale*unit_length

            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

            mask = mask==False
            x_space = x_space * mask

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(x_space)

        return x

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzFullyConnectedLora(nn.Module):
    """
    Lorentz fully connected layer with subject-specific low-rank adapters (LoRA).

    Args:
        manifold: Instance of Lorentz manifold
        in_features, out_features, bias: Same as nn.Linear
        init_scale: Scale parameter for internal normalization
        learn_scale: If scale parameter should be learnable
        normalize: If internal normalization should be applied
        activation: Optional nonlinearity
        dropout: Dropout probability
        nheads: Number of heads for splitting
        num_subjects: Number of subject-specific adapters
        rank: LoRA rank (low-rank dimension)
    """

    def __init__(
        self,
        manifold,
        in_features,
        out_features,
        bias=False,
        init_scale=None,
        learn_scale=False,
        normalize=False,
        activation=None,
        dropout=0.0,
        nheads=1,
        num_subjects=None,
        rank=0,
        lora_lr = 1e-6
    ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize
        self.nheads = nheads
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.rank = rank
        self.lora_lr = lora_lr

        # Shared base weight (frozen)
        self.weight = nn.Linear(self.in_features - 1, self.out_features - 1, bias=bias)

        # LoRA adapters
        self.Q = nn.Parameter(torch.zeros(num_subjects + 1, self.in_features - 1, rank))
        self.R = nn.Parameter(torch.zeros(num_subjects + 1, self.out_features - 1, rank))

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

        # Initialization
        self.init_std = 0.02
        self.reset_parameters()
        nn.init.uniform_(self.Q, -self.init_std, self.init_std)
        nn.init.zeros_(self.R)

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)

    def forward(self, x, subject_ids):
        # Extract subject IDs
        subject_ids = subject_ids[..., 0].squeeze().int()

        # Drop Lorentz time dimension -> (B, in_features-1)
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        # Shared base linear transform
        x_base = F.linear(x_space, self.weight.weight)  # (B, out_features-1)

        # Subject-specific LoRA adapters
        batch_Q = self.Q[subject_ids]  # (B, out_features-1, r)
        batch_R = self.R[subject_ids]  # (B, in_features-1, r)

        # Low-rank LoRA update: (x_space @ R) @ Q^T
        lora_intermediate = torch.einsum("bir,bj->br", batch_Q, x_space)   # (B, r)
        lora_out = torch.einsum("bir,br->bi", batch_R, lora_intermediate)  # (B, out_features-1)
        # Combine base + LoRA
        x_space = x_base + self.lora_lr * lora_out

        # Nonlinearity and dropout
        if self.activation is not None:
            x_space = self.activation(x_space)
        x_space = self.dropout(x_space)

        # Multi-head reshaping
        if self.nheads > 1:
            x_space = x_space.view(
                x_space.size(0),
                x_space.size(1),
                self.nheads,
                self.out_features // self.nheads,
            ).transpose(1, 2)

        # Normalization or Lorentzian time augmentation
        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)
            mask = square_norm <= 1e-10
            square_norm[mask] = 1
            unit_length = x_space / torch.sqrt(square_norm)
            x_space = scale * unit_length
            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())
            mask = ~mask
            x_space = x_space * mask
            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(x_space)

        return x


class LorentzFullyConnectedBoostLora(nn.Module):
    """
    Lorentz fully connected layer with subject-specific low-rank adaptation via Lorentz boosts.
    
    Instead of applying a linear low-rank update in the space component, this layer
    applies a composition of subject-specific Lorentz boosts, which are the natural
    "translations" on the hyperboloid manifold.

    Args:
        manifold: Instance of Lorentz manifold
        in_features, out_features, bias: Same as nn.Linear
        init_scale: Scale parameter for internal normalization
        learn_scale: If scale parameter should be learnable
        normalize: If internal normalization should be applied
        activation: Optional nonlinearity
        dropout: Dropout probability
        num_subjects: Number of subject-specific adapters
        rank: Number of composed boosts per subject (boost rank)
        alpha: Scaling factor for boost magnitudes
    """

    def __init__(
        self,
        manifold,
        in_features,
        out_features,
        bias=False,
        init_scale=None,
        learn_scale=False,
        normalize=False,
        activation=None,
        dropout=0.0,
        num_subjects=None,
        rank=4,
        alpha=0.1
    ):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.rank = rank
        self.alpha = alpha
        self.num_subjects = num_subjects

        self.weight = nn.Linear(self.in_features - 1, self.out_features - 1, bias=bias)

        # Subject-specific boost direction parameters
        # Each subject has `rank` boost directions, each of dimension (out_features - 1)
        self.boost_dirs = nn.Parameter(
            torch.zeros(num_subjects + 1, rank, self.out_features - 1)
        )
        
        # Subject-specific boost magnitude
        self.boost_mags = nn.Parameter(
            torch.zeros(num_subjects + 1, rank)
        )

        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

        self.init_std = 0.02
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)
        nn.init.uniform_(self.boost_dirs, -self.init_std, self.init_std)
        nn.init.zeros_(self.boost_mags)

    def _apply_boost(self, x, direction, magnitude):
        x_t = x[..., 0:1]  # Time component (B, 1)
        x_s = x[..., 1:]   # Space component (B, out_features-1)
        
        # Normalize direction
        dir_norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
        v_unit = direction / dir_norm
        
        # Scale magnitude
        mag = magnitude.unsqueeze(-1) * self.alpha  # (B, 1)
        
        # Compute boost components
        cosh_mag = torch.cosh(mag)
        sinh_mag = torch.sinh(mag)

        vx = (v_unit * x_s).sum(dim=-1, keepdim=True)  # (B, 1)

        #apply
        xt_new = x_t * cosh_mag + vx * sinh_mag
        xs_new = x_s + v_unit * ((cosh_mag - 1) * vx + x_t * sinh_mag)
        
        return torch.cat([xt_new, xs_new], dim=-1)

    def forward(self, x, subject_ids):
        subject_ids = subject_ids[..., 0].squeeze().int()


        x_space = x.narrow(-1, 1, x.shape[-1] - 1)
        x_space = self.weight(x_space)  # (B, out_features-1)

        # Nonlinearity and dropout (applied before adding time)
        if self.activation is not None:
            x_space = self.activation(x_space)
        x_space = self.dropout(x_space)

        x_hyp = self.manifold.add_time(x_space)  # (B, out_features)

        batch_dirs = self.boost_dirs[subject_ids]  # (B, rank, out_features-1)
        batch_mags = self.boost_mags[subject_ids]  # (B, rank)

        for r in range(self.rank):
            x_hyp = self._apply_boost(
                x_hyp,
                batch_dirs[:, r, :],  # (B, out_features-1)
                batch_mags[:, r]       # (B,)
            )

        x_hyp = self.manifold.projx(x_hyp)

        return x_hyp


class LorentzProjection(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, dropout=False):
        super(LorentzProjection, self).__init__()

        self.down = False

        if out_features >= in_features:
            # self.rotation = LorentzRotation_Up(manifold, in_features, out_features, if_regularize=False, if_projected=True)
            # self.rotation = orthogonal(self.rotation, "weight", orthogonal_map="cayley")

            # self.rotation = CayleyLinear(in_features-1, out_features-1, bias=False)
            self.rotation = LorentzRotationFixedNorm(manifold, in_features, out_features, if_regularize=False,
                                                      if_projected=True)
            self.down = True

        else:
            self.rotation = LorentzRotationFixedNorm(manifold, in_features, out_features, if_regularize=False,
                                               if_projected=True)
            self.down = True

        #self.boost = LorentzBoost(manifold, init_weight=1)
        self.boost = LorentzBoostScaleAlternate(manifold, in_features=out_features)
       #self.boost = LorentzBoostAlternate(manifold, dim=out_features)
        self.manifold = manifold

    def forward(self, input):

        if not self.down:
            xt = self.rotation(input[...,1:])
            xt = self.manifold.add_time(xt)
            #print("not down")
        else:
            xt = self.rotation(input)

        h = self.boost(xt)
        h = self.manifold.projx(h)
        # h = self.projection(h)

        return h


class LorentzMLRFF(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, dropout=False):
        super(LorentzMLRFF, self).__init__()

        self.manifold = manifold

        self.mlr = LorentzMLR(manifold, in_features, out_features)
        self.projection = nn.Linear(out_features, out_features-1)

    def forward(self, input):
        x = self.mlr(input)
        x = self.projection(x)

        x = self.manifold.add_time(x)

        return x


class HorosphereFC(nn.Module):
    def __init__(self, manifold, input_dim: int, output_dim: int):
        """
        input_dim: Dimension d of input hyperboloid (H^d)
        output_dim: Dimension m of output hyperboloid (H^m)
        k: Curvature parameter (k > 0)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create Lorentz manifold for input space
        self.manifold = manifold

        # Parameters: p_j ∈ H^d, ξ_j ∈ tangent space at origin (lightlike)
        self.p = ManifoldParameter(
            self.manifold.projx(torch.randn(output_dim-1, input_dim)),
            manifold=self.manifold
        )

        # ξ direction parameters (in tangent space at origin)
        self.xi_dir = nn.Parameter(torch.randn(output_dim-1, input_dim-1))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x ∈ H_k^d (shape: [batch_size, d+1])
        Output: y ∈ H_k^m (shape: [batch_size, m+1])
        """
        # Normalize ξ directions to unit norm (in Euclidean sense)
        z = self.xi_dir / torch.norm(self.xi_dir, dim=1, keepdim=True)  # Unit norm
        xi = torch.cat([
            torch.norm(z, dim=1, keepdim=True),  # ‖v‖
            z
        ], dim=1)   # [num_planes, d+1]

        xi = self.manifold.projx(xi)

        log_inner = torch.log(
            -self.manifold.minkowski_dot(
                x.unsqueeze(-2),  # [batch_size, 1, in_dim+1]
                xi.unsqueeze(0)  # [1, out_dim, in_dim+1]
            )
        )  # [batch_size, out_dim]

        busemann_x = torch.sqrt(self.manifold.k) * log_inner
        busemann_p = torch.sqrt(self.manifold.k) * torch.log(
            -self.manifold.minkowski_dot(self.p, xi)
        )  # [out_dim]

        # [batch_size, out_dim]
        v = busemann_x - busemann_p.unsqueeze(0)

        v = torch.nn.functional.pad(v.squeeze(-1), (1, 0, 0, 0), mode='constant', value=0)
        v = self.manifold.rescale_to_max_euclid(v)
        v = self.manifold.expmap0(v)

        # Apply exponential map
        return v

