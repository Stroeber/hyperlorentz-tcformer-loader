import torch
import torch.nn as nn

from hyperbolic_lib.lib.geoopt import ManifoldParameter
from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz
from hyperbolic_lib.lib.lorentz.blocks.transformer_blocks import LorentzMultiHeadAttention
from utils.utils_h import (BATCH1D_TYPES,
                                       patch_len,
                                       matt_covar)

from models.input_processors import EuclideanConvDenoiser
from models.default_configs import DEFAULT_DENOISER_CONFIGS
from models.blocks import LorentzInceptionBlock
from utils.utils_h import (Conv1dSamePadding,
                                       CONV1D_TYPES,
                                       BATCH1D_TYPES,
                                       POOL1D_TYPES)



class BaselineInceptionEncoder(nn.Module):
    def __init__(self,
                 manifold,
                 in_channel=1,
                 features=18 * 18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0):
        super().__init__()

        self.manifold = manifold
        self.baseline_block = LorentzInceptionBlock(self.manifold,
                                                             in_channels=in_channel,
                                                             n_filters=int(features / 4),
                                                             kernel_sizes=kernel_sizes,
                                                             bottleneck_channels=inception_channels,
                                                             activation=None,
                                                             return_indices=False,
                                                             conv_type=conv_type,
                                                             batch_type=None,
                                                             pool_type="average",
                                                             dropout=dropout)


        self.layer_norm = BATCH1D_TYPES["original"](self.manifold, features + 1) if batch_type is not None else nn.Sequential()
        self.activation = nn.ReLU(inplace=True)


    def forward(self, x):

        processed = self.baseline_block(x)
        normed = self.layer(processed)

        return self.activation(normed)


class BaselineInceptionEncoderSeparateNorm(nn.Module):
    def __init__(self,
                 manifold,
                 in_channel=1,
                 features=18 * 18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0):
        super().__init__()

        self.manifold = manifold
        self.baseline_block = LorentzInceptionBlock(self.manifold,
                                                    in_channels=in_channel,
                                                    n_filters=int(features / 4),
                                                    kernel_sizes=kernel_sizes,
                                                    bottleneck_channels=inception_channels,
                                                    activation=None,
                                                    return_indices=False,
                                                    conv_type=conv_type,
                                                    batch_type=None,
                                                    pool_type="average",
                                                    dropout=dropout)

        self.layer_norm = BATCH1D_TYPES["original"](self.manifold,
                                                    features + 1) if batch_type is not None else nn.Sequential()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, x_sub=None):
        processed = self.baseline_block(x)
        normed = self.layer(processed)

        return self.activation(normed)


class BaselineFCEncoder(nn.Module):
    def __init__(self,
                 manifold,
                 in_channel=1,
                 features=18 * 18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0):

        self.manifold = manifold

        conv_layer = CONV1D_TYPES[conv_type]

        self.conv_from_bottleneck_1 = conv_layer(self.manifold,
                                                 in_channels=in_channel,
                                                 out_channels=features,
                                                 kernel_size=kernel_sizes[0],
                                                 stride=3,
                                                 padding=kernel_sizes[0] // 2,
                                                 bias=False
                                                 )


        self.layer_norm = BATCH1D_TYPES[batch_type](self.manifold, features + 1) if batch_type is not None else nn.Sequential()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):

        processed = self.baseline_block(x)
        normed = self.layer(processed)

        return self.activation(normed)




