import torch
import math


class LorentzConv(torch.nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold:
        in_channels, ...
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            manifold,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias: bool = False,
            head_num: int = 1,
            head_input: bool = False
    ):
        super(LorentzConv, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        input_channels = ((self.in_channels - head_num) * self.kernel_size[0] * self.kernel_size[1]) + head_num if (head_input) \
            else ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.linearized_kernel = torch.nn.parameter(torch.rand(input_channels, self.out_channels))

        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation,
                                      padding=padding, stride=stride)

        self.num_heads = head_num
        self.head_input = head_input

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        if self.head_input:
            x = x.reshape(bsz, h, w, -1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, channels * elements/window, windows
        patches = patches.permute(0, 2, 1)

        if self.head_input:
            patches = patches.reshape(patches.shape[0], patches.shape[1], self.num_heads, -1)

        # Now we have flattened patches with multiple time elements -> fix the concatenation
        # Lorentz direct concatenation by Qu et al. (2022)
        patches_time = torch.clamp(patches[..., :self.kernel_len], min=self.manifold.k.sqrt())  # Zero padding -> 1
        patches_space = patches[..., self.kernel_len:]

        if self.head_input:
            patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.num_heads, int(self.in_channels/self.num_heads) - 1,
                                                  -1).permute(0, 1, 4,  2, 3).reshape(patches_space.shape)
        else:
            patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1,
                                                  -1).transpose(-1, -2).reshape(patches_space.shape)

        patches_time_rescaled = torch.sqrt(
            torch.sum(patches_time ** 2, dim=-1, keepdim=True) + ((self.kernel_len - 1) * -self.manifold.k))

        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        patches_dist = self.manifold.sqdist(patches_pre_kernel, self.linearized_kernel)
        patches_weights = torch.nn.functional.softmax(patches_dist, dim=-1)


        # reshape again
        if self.num_heads <= 1:
            patches = patches.view(bsz, h_out, w_out, self.out_channels)
        else:
            patches = patches.view(bsz, h_out, w_out, self.num_heads, int(self.out_channels / self.num_heads))

        return patches
