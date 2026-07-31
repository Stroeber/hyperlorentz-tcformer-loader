import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import unfoldNd

from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz
from hyperbolic_lib.lib.lorentz.layers import LorentzFullyConnected, LorentzBoostScaleAlternate, HorosphereFC
from hyperbolic_lib.lib.geoopt.manifolds import Stiefel
from hyperbolic_lib.lib.lorentz.layers_1d.layer_utils import CustomWeightConv1d
from hyperbolic_lib.lib.lorentz.layers.linear_layers import (LorentzProjection,
                                                           LorentzBoost,
                                                           LorentzBoostScale)


def unfold1d(input, kernel_size: int, stride: int, padding: int):
    *shape, length = input.shape
    n_frames = (max(length, kernel_size) - kernel_size) // stride + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    input = input[..., :tgt_length].contiguous()
    strides = list(input.stride())
    strides = strides[:-1] + [stride, 1]
    out = input.as_strided(shape + [n_frames, kernel_size], strides)
    return out.transpose(-1, -2)

class LorentzLoRAConv1dDebug(nn.Module):
    def __init__(self,
                 manifold: CustomLorentz,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 rescale_before=False,
                 rescale_after=False,
                 LFC_normalize=False,
                 rank=3,
                 num_subjects=None):
        super(LorentzLoRAConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rescale_before = rescale_before
        self.rescale_after = rescale_after

        # linearized features (like original)
        lin_features = (self.in_channels - 1) * self.kernel_size + 1

        # Shared kernel (keep LorentzFullyConnected as-is)
        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )

        # LoRA adapters (subject-specific), shapes derived from actual Linear weight
        self.rank = rank
        self.num_subjects = int(num_subjects)

        # internal linear weight shape used by LorentzFullyConnected:
        # self.linearized_kernel.weight is nn.Linear(in_features-1, out_features-1)
        W_shared_shape = tuple(self.linearized_kernel.weight.weight.shape)  # (W_rows, W_cols)
        W_rows, W_cols = W_shared_shape

        # one Q/R per subject, matching the actual linear weight dims
        self.Q = nn.Parameter(torch.zeros(self.num_subjects + 1, W_rows, rank))
        self.R = nn.Parameter(torch.zeros(self.num_subjects + 1, W_cols, rank))
        nn.init.zeros_(self.Q)
        nn.init.zeros_(self.R)

        self.rescale_before = rescale_before
        self.rescale_after = rescale_after
        # unfolding for patches (same as original conv)
        self.unfold = unfoldNd.UnfoldNd(kernel_size, padding=0, stride=stride)

    def forward(self, x, subject_ids):

        """
        x: (B, L, C) channel-last
        subject_ids: tensor with one id per example (shape (B,) or (B,1) etc.)
        """
        # normalize subject ids into (B,)
        subject_ids = subject_ids[..., 0].squeeze().int()

        bsz = x.shape[0]

        # safety: allow scalar subject id (apply same adapter to full batch)
        if subject_ids.numel() == 1 and bsz > 1:
            subject_ids = subject_ids.expand(bsz)
        if subject_ids.numel() != bsz:
            raise ValueError(f"subject_ids must have length 1 or batch size ({bsz}); got {subject_ids.numel()}")

        if subject_ids.max().item() > self.num_subjects:
            raise ValueError(
                f"Subject id {subject_ids.max().item()} exceeds LoRA capacity {self.num_subjects}"
            )

        # --- padding & clamp (same as original conv) ---
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt())

        # --- unfold into patches ---
        x_perm = x.permute(0, 2, 1)  # (B, C, L)
        patches = self.unfold(x_perm)  # (B, C*k, L')
        patches = patches.permute(0, 2, 1)  # (B, L', C*k)

        # Lorentz-specific preprocessing (same ops as original conv)
        patches_space = patches.narrow(-1, self.kernel_size, patches.shape[-1] - self.kernel_size)
        patches_space = patches_space.reshape(
            patches_space.shape[0],
            patches_space.shape[1],
            self.in_channels - 1,
            -1
        ).transpose(-1, -2).reshape(patches_space.shape)
        patches_pre_kernel = self.manifold.add_time(patches_space)  # expected (B, L', lin_features)

        if self.rescale_before:
            patches_pre_kernel = self.manifold.rescale_to_max(patches_pre_kernel)

        # --- prepare space-only input expected by internal Linear ---
        # patches_pre_kernel: (B, L', lin_features)
        B, Lp, linf = patches_pre_kernel.shape
        # the internal linear receives the "space" part of size (lin_features - 1)
        space_in = patches_pre_kernel.narrow(-1, 1, linf - 1)  # ideally (B, L', W_cols)

        # If space_in was accidentally flattened to (B*L', J), detect and reshape:
        if space_in.ndim == 2:
            # common flatten occurrence: shape == (B*L', J)
            if space_in.shape[0] == B * Lp:
                space_in = space_in.view(B, Lp, space_in.shape[-1])
            else:
                raise RuntimeError(
                    f"Unexpected space_in shape {tuple(space_in.shape)}; cannot interpret as (B, L', J)."
                )

        # --- shared weight tensor and per-subject LoRA deltas ---
        W_shared = self.linearized_kernel.weight.weight  # (W_rows, W_cols)

        batch_Q = self.Q[subject_ids]   # (B, W_rows, r)
        batch_R = self.R[subject_ids]   # (B, W_cols, r)

        # sub_W: (B, W_rows, W_cols)
        W_sub = torch.einsum("bir,bjr->bij", batch_Q, batch_R)

        # broadcast-add to shared weights:
        W = W_shared.unsqueeze(0) + W_sub  # (B, W_rows, W_cols)

        # --- apply kernel to space_in:
        # space_in: (B, L', W_cols)
        # W:    (B, W_rows, W_cols)
        # result:   (B, L', W_rows)
        out_space = torch.einsum("bkj,bij->bki", space_in, W)

        # add bias if present (internal Linear bias)
        bias = self.linearized_kernel.weight.bias
        if bias is not None:
            out_space = out_space + bias.view(1, 1, -1)

        # same post-processing as LorentzFullyConnected.forward
        if self.linearized_kernel.activation is not None:
            out_space = self.linearized_kernel.activation(out_space)
        out_space = self.linearized_kernel.dropout(out_space)

        if self.linearized_kernel.nheads > 1:
            out_space = out_space.view(
                out_space.size(0),
                out_space.size(1),
                self.linearized_kernel.nheads,
                self.linearized_kernel.out_features // self.linearized_kernel.nheads
            ).transpose(1, 2)

        if self.linearized_kernel.normalize:
            # compute scale from time coordinate of patches_pre_kernel
            scale = patches_pre_kernel.narrow(-1, 0, 1).sigmoid() * self.linearized_kernel.scale.exp()
            square_norm = (out_space * out_space).sum(dim=-1, keepdim=True)

            mask = square_norm <= 1e-10

            square_norm[mask] = 1
            unit_length = out_space / torch.sqrt(square_norm)
            x_space = scale * unit_length

            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

            mask_neg = mask == False
            x_space = x_space * mask_neg

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(out_space)


        return x
class LorentzLoRAConv1d(nn.Module):
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
        LFC_normalize=False,
        rank=3,
        num_subjects=None
    ):
        super(LorentzLoRAConv1d, self).__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rescale_before = rescale_before
        self.rescale_after = rescale_after

        lin_features = (self.in_channels - 1) * self.kernel_size + 1
        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )

        # LoRA adapters (subject-specific)
        self.rank = rank
        self.num_subjects = num_subjects
        Ip, Jp = out_channels, lin_features
        self.Q = nn.Parameter(torch.zeros(num_subjects + 1, Ip - 1, rank))
        self.R = nn.Parameter(torch.zeros(num_subjects + 1, Jp - 1, rank))
        nn.init.normal_(self.Q)
        nn.init.zeros_(self.R)

        # unfolding for patches
        self.unfold = unfoldNd.UnfoldNd(kernel_size, padding=0, stride=stride)

    def forward(self, x, subject_ids):
        subject_ids = subject_ids[..., 0].squeeze().int()
        """
        x: (B, L, C) with channel-last representation
        """
        bsz = x.shape[0]

        # --- padding ---
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt())

        # --- unfold into patches ---
        x = x.permute(0, 2, 1)  # (B, C, L)
        patches = self.unfold(x)  # (B, C*k, L')
        patches = patches.permute(0, 2, 1)  # (B, L', C*k)

        patches_space = patches.narrow(-1, self.kernel_size, patches.shape[-1] - self.kernel_size)
        patches_space = patches_space.reshape(
            patches_space.shape[0],
            patches_space.shape[1],
            self.in_channels - 1,
            -1
        ).transpose(-1, -2).reshape(patches_space.shape)

        patches_pre_kernel = self.manifold.add_time(patches_space)

        if self.rescale_before:
            patches_pre_kernel = self.manifold.rescale_to_max(patches_pre_kernel)

        # --- shared weight ---
        W_shared = self.linearized_kernel.weight.weight  # (Ip, Jp)

        # --- LoRA subject adapters ---
        batch_Q = self.Q[subject_ids]  # (B, Ip, r)
        batch_R = self.R[subject_ids]  # (B, Jp, r)
        W_sub = torch.einsum("bir,bjr->bij", batch_Q, batch_R)  # (B, Ip, Jp)

        # Add LoRA to shared
        W_full = W_shared.unsqueeze(0) + W_sub  # (B, Ip, Jp)

        # --- apply kernel ---
        out_space = torch.einsum("bij,bkj->bki", W_full, patches_pre_kernel[..., 1:])
        out = self.manifold.add_time(out_space)

        if self.rescale_after:
            out = self.manifold.rescale_to_max(out)

        return out


class LorentzConv1d(nn.Module):
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
        super(LorentzConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        lin_features = (self.in_channels - 1) * self.kernel_size + 1

        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )

        # self.linearized_kernel = HorosphereFC(
        #     manifold,
        #     lin_features,
        #     self.out_channels,
        # )
        self.rescale_before = rescale_before
        self.rescale_after = rescale_after

        self.unfold = unfoldNd.UnfoldNd(
            kernel_size, padding=0, stride=stride
        )

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x len x C """
        bsz = x.shape[0]

        # origin padding
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt())

        x = x.permute(0, 2, 1)
        #  patches = unfold1d(x, self.kernel_size, self.stride, self.padding)
        #  patches = patches.reshape(bsz, self.kernel_size * self.in_channels, -1)
        patches = self.unfold(x)
        patches = patches.permute(0, 2, 1)

        patches_space = patches.narrow(-1, self.kernel_size, patches.shape[-1] - self.kernel_size)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape)  # No need, but seems to improve runtime??
        patches_pre_kernel = self.manifold.add_time(patches_space)

        if self.rescale_before:
            patches_pre_kernel = self.manifold.rescale_to_max(patches_pre_kernel)

        out = self.linearized_kernel(patches_pre_kernel)

        if torch.any(torch.isnan(out)):
            print("issues conv")

        if self.rescale_after:
            out = self.manifold.rescale_to_max(out)

        return out


class LorentzPureConv1d(nn.Module):
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
            rescale_after=True,
    ):
        super(LorentzPureConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        lin_features = (self.in_channels - 1) * self.kernel_size + 1

        self.linearized_kernel = LorentzProjection(self.manifold,
                                                   lin_features,
                                                   out_channels,)

        #self.unfold = torch.nn.Unfold(kernel_size=self.kernel_size, padding=padding, stride=stride)

        self.rescale_before = rescale_before
        self.rescale_after = rescale_after

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x len x C """
        bsz = x.shape[0]

        # origin padding
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt())

        x = x.permute(0, 2, 1)
        patches = unfold1d(x, self.kernel_size, self.stride, padding=0)
        patches = patches.reshape(bsz, self.kernel_size*self.in_channels, -1).permute(0, 2, 1)

        patches_space = patches.narrow(-1, self.kernel_size, patches.shape[-1] - self.kernel_size)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape)  # No need, but seems to improve runtime??

        patches_pre_kernel = self.manifold.add_time(patches_space)

        if self.rescale_before:
            patches_pre_kernel = self.manifold.rescale_to_max(patches_pre_kernel)

        out = self.linearized_kernel(patches_pre_kernel)
        out = self.manifold.projx(out)
        if self.rescale_after:
            out = self.manifold.rescale_to_max(out)

        return out


class HyperbolicStiefelConv1D(nn.Module):
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
        super(HyperbolicStiefelConv1D, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.kernel_size = kernel_size

        self.rotation_manifold = Stiefel()

        self.rotate = nn.Conv1d(in_channels-1,
                                out_channels-1,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False)
        d_out, d_in, n = self.rotate.weight.shape

        self.debug_test = False

        if d_out >= d_in * n:

            self.rotate = CustomWeightConv1d(self.rotation_manifold,
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


class HyperbolicCayleyConv1D(nn.Module):

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
        super(HyperbolicCayleyConv1D, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.kernel_size = kernel_size

        self.rotation_manifold = Stiefel()

        self.rotate = nn.Conv1d(in_channels - 1,
                                out_channels - 1,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False)
        d_out, d_in, n = self.rotate.weight.shape

        if d_out >= d_in * n:
            self.rotate = CustomWeightConv1d(None,
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
                                                       use_trivialization=False)

        #self.boost = LorentzBoost(manifold)
        self.boost = LorentzBoostScaleAlternate(manifold, in_features=out_channels)

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


if __name__ == '__main__':

    x = torch.rand((128, 4, 32))
    manifold = CustomLorentz(k=1, learnable=True)

    project_x = manifold.projx(x)

    layer = LorentzConv1d(manifold, 32,
                          64,
                          6)

    output = layer(project_x)

    print("break")
