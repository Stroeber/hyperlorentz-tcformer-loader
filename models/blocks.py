import torch
from torch import nn


from hyperbolic_lib.lib.lorentz.layers.LModules import LorentzAct, QuickDirtyMaxPool

from utils.utils_h import (Conv1dSamePadding,
                           CONV1D_TYPES,
                           BATCH1D_TYPES,
                           POOL1D_TYPES)


def pass_through(x):
    return x


class LorentzInceptionBlock(nn.Module):
    def __init__(self,
                 manifold,
                 in_channels,
                 n_filters=32,
                 kernel_sizes=(9, 19, 39),
                 bottleneck_channels=8,
                 activation=nn.ReLU(),
                 return_indices=False,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0):

        super(LorentzInceptionBlock, self).__init__()
        self.return_indices = return_indices

        self.manifold = manifold

        conv_layer = CONV1D_TYPES[conv_type]
        batch_layer = BATCH1D_TYPES[batch_type] if batch_type is not None else None

        n_filters = n_filters + 1
        bottleneck_channels = bottleneck_channels + 1
        in_channels = in_channels + 1

        if in_channels > 1:
            self.bottleneck = conv_layer(self.manifold,
                                         in_channels=in_channels,
                                         out_channels=bottleneck_channels,
                                         kernel_size=1,
                                         stride=1,
                                         bias=False
                                         )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = conv_layer(self.manifold,
                                                 in_channels=bottleneck_channels,
                                                 out_channels=n_filters,
                                                 kernel_size=kernel_sizes[0],
                                                 stride=1,
                                                 padding=kernel_sizes[0] // 2,
                                                 bias=False
                                                 )
        self.conv_from_bottleneck_2 = conv_layer(self.manifold,
                                                 in_channels=bottleneck_channels,
                                                 out_channels=n_filters,
                                                 kernel_size=kernel_sizes[1],
                                                 stride=1,
                                                 padding=kernel_sizes[1] // 2,
                                                 bias=False
                                                 )
        self.conv_from_bottleneck_3 = conv_layer(self.manifold,
                                                 in_channels=bottleneck_channels,
                                                 out_channels=n_filters,
                                                 kernel_size=kernel_sizes[2],
                                                 stride=1,
                                                 padding=kernel_sizes[2] // 2,
                                                 bias=False
                                                 )
        if pool_type == "dirty":
            self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
            self.max_pool = QuickDirtyMaxPool(self.manifold, self.max_pool)
        elif pool_type == "dirty_average":
            self.max_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
            self.max_pool = QuickDirtyMaxPool(self.manifold, self.max_pool)
        else:
            self.max_pool = POOL1D_TYPES[pool_type](self.manifold, kernel_size=3, padding=1, stride=1)

        self.conv_from_maxpool = conv_layer(
            self.manifold,
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batch_norm = batch_layer(manifold, num_features=4 * (n_filters - 1) + 1) if batch_type is not None else nn.Sequential()

        self.activation = activation
        self.activation = LorentzAct(self.activation, manifold) if activation is not None else nn.Sequential()

        self.drop = nn.Dropout(p=dropout)
        self.drop = LorentzAct(self.drop, manifold) if activation is not None else nn.Sequential()

    def forward(self, x):
        # step 1
        z_bottleneck = self.bottleneck(x)
        z_maxpool = self.max_pool(x)
        z_maxpool = self.manifold.rescale_to_max(z_maxpool)

        # step 2
        z1 = self.conv_from_bottleneck_1(z_bottleneck)
        z2 = self.conv_from_bottleneck_2(z_bottleneck)
        z3 = self.conv_from_bottleneck_3(z_bottleneck)
        z4 = self.conv_from_maxpool(z_maxpool)

        # step 3
        z = torch.cat([z1, z2[..., 1:], z3[..., 1:], z4[..., 1:]], dim=-1)
        z = self.manifold.projx(z)
        z = self.activation(self.batch_norm(z))
        z = self.drop(z)

        z = self.manifold.rescale_to_max(z)

        return z

class LorentzInceptionBlockJoint(nn.Module):
    def __init__(self,
                 manifold,
                 in_channels,
                 n_filters=32,
                 kernel_sizes=(9, 19, 39),
                 bottleneck_channels=8,
                 activation=nn.ReLU(),
                 return_indices=False,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 num_subjects=16,
                 subject_embed_dim=3):

        super(LorentzInceptionBlockJoint, self).__init__()
        self.return_indices = return_indices

        self.manifold = manifold
        self.subject_dim = subject_embed_dim
        self.subject_embeddings = nn.Embedding(num_subjects + 1, self.subject_dim, padding_idx=0)

        conv_layer = CONV1D_TYPES[conv_type]
        batch_layer = BATCH1D_TYPES[batch_type] if batch_type is not None else None

        n_filters = n_filters + 1
        bottleneck_channels = bottleneck_channels + 1
        in_channels = in_channels + 1

        if in_channels > 1:
            self.bottleneck = conv_layer(self.manifold,
                                         in_channels=in_channels,
                                         out_channels=bottleneck_channels,
                                         kernel_size=1,
                                         stride=1,
                                         bias=False
                                         )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = conv_layer(self.manifold,
                                                 in_channels=bottleneck_channels + self.subject_dim,
                                                 out_channels=n_filters,
                                                 kernel_size=kernel_sizes[0],
                                                 stride=1,
                                                 padding=kernel_sizes[0] // 2,
                                                 bias=False
                                                 )
        self.conv_from_bottleneck_2 = conv_layer(self.manifold,
                                                 in_channels=bottleneck_channels + self.subject_dim,
                                                 out_channels=n_filters,
                                                 kernel_size=kernel_sizes[1],
                                                 stride=1,
                                                 padding=kernel_sizes[1] // 2,
                                                 bias=False
                                                 )
        self.conv_from_bottleneck_3 = conv_layer(self.manifold,
                                                 in_channels=bottleneck_channels + self.subject_dim,
                                                 out_channels=n_filters,
                                                 kernel_size=kernel_sizes[2],
                                                 stride=1,
                                                 padding=kernel_sizes[2] // 2,
                                                 bias=False
                                                 )
        if pool_type == "dirty":
            self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
            self.max_pool = QuickDirtyMaxPool(self.manifold, self.max_pool)
        elif pool_type == "dirty_average":
            self.max_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
            self.max_pool = QuickDirtyMaxPool(self.manifold, self.max_pool)
        else:
            self.max_pool = POOL1D_TYPES[pool_type](self.manifold, kernel_size=3, padding=1, stride=1)

        self.conv_from_maxpool = conv_layer(
            self.manifold,
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batch_norm = batch_layer(manifold, num_features=4 * (n_filters - 1) + 1) if batch_type is not None else nn.Sequential()

        self.activation = activation
        self.activation = LorentzAct(self.activation, manifold) if activation is not None else nn.Sequential()

        self.drop = nn.Dropout(p=dropout)
        self.drop = LorentzAct(self.drop, manifold) if activation is not None else nn.Sequential()

    def forward(self, x, x_sub):

        # step 1
        z_bottleneck = self.bottleneck(x)
        z_maxpool = self.max_pool(x)
        z_maxpool = self.manifold.rescale_to_max(z_maxpool)

        # step 2
        x_sub = self.subject_embeddings(x_sub.int()).squeeze()
        z_bottleneck = torch.cat([z_bottleneck, x_sub], dim=-1)

        z1 = self.conv_from_bottleneck_1(z_bottleneck)
        z2 = self.conv_from_bottleneck_2(z_bottleneck)
        z3 = self.conv_from_bottleneck_3(z_bottleneck)
        z4 = self.conv_from_maxpool(z_maxpool)

        # step 3
        z = torch.cat([z1, z2[..., 1:], z3[..., 1:], z4[..., 1:]], dim=-1)
        z = self.manifold.projx(z)
        z = self.activation(self.batch_norm(z))
        z = self.drop(z)

        z = self.manifold.rescale_to_max(z)

        return z


class LorentzInceptionBlockJointLora(nn.Module):
    def __init__(self,
                 manifold,
                 in_channels,
                 n_filters=32,
                 kernel_sizes=(9, 19, 39),
                 bottleneck_channels=8,
                 activation=nn.ReLU(),
                 return_indices=False,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 num_subjects=None,
                 sub_rank = 3):

        super(LorentzInceptionBlockJointLora, self).__init__()
        self.return_indices = return_indices

        self.manifold = manifold

        conv_layer = CONV1D_TYPES["LoRA"]
        batch_layer = BATCH1D_TYPES[batch_type] if batch_type is not None else None

        n_filters = n_filters + 1
        bottleneck_channels = bottleneck_channels + 1
        in_channels = in_channels + 1

        if in_channels > 1:
            self.bottleneck = conv_layer(self.manifold,
                                        in_channels = in_channels,
                                        out_channels = bottleneck_channels,
                                        kernel_size = 1,
                                        stride = 1,
                                        bias = False,
                                        rank=sub_rank ,
                                        num_subjects=num_subjects
                                         )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = conv_layer(self.manifold,
                                                in_channels=bottleneck_channels,
                                                out_channels=n_filters,
                                                kernel_size=kernel_sizes[0],
                                                stride=1,
                                                padding=kernel_sizes[0] // 2,
                                                bias=False,
                                                rank=sub_rank,
                                                num_subjects=num_subjects
                                                 )
        self.conv_from_bottleneck_2 = conv_layer(self.manifold,
                                                in_channels=bottleneck_channels,
                                                out_channels=n_filters,
                                                kernel_size=kernel_sizes[1],
                                                stride=1,
                                                padding=kernel_sizes[1] // 2,
                                                bias=False,
                                                rank=sub_rank,
                                                num_subjects=num_subjects
                                                 )
        self.conv_from_bottleneck_3 = conv_layer(self.manifold,
                                                in_channels=bottleneck_channels,
                                                out_channels=n_filters,
                                                kernel_size=kernel_sizes[2],
                                                stride=1,
                                                padding=kernel_sizes[2] // 2,
                                                bias=False,
                                                rank=sub_rank,
                                                num_subjects=num_subjects
                                                 )
        if pool_type == "dirty":
            self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
            self.max_pool = QuickDirtyMaxPool(self.manifold, self.max_pool)
        elif pool_type == "dirty_average":
            self.max_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
            self.max_pool = QuickDirtyMaxPool(self.manifold, self.max_pool)
        else:
            self.max_pool = POOL1D_TYPES[pool_type](self.manifold, kernel_size=3, padding=1, stride=1)

        self.conv_from_maxpool = conv_layer(
            self.manifold,
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            num_subjects=num_subjects,
            rank=sub_rank
        )
        self.batch_norm = batch_layer(manifold, num_features=4 * (n_filters - 1) + 1) if batch_type is not None else nn.Sequential()

        self.activation = activation
        self.activation = LorentzAct(self.activation, manifold) if activation is not None else nn.Sequential()

        self.drop = nn.Dropout(p=dropout)
        self.drop = LorentzAct(self.drop, manifold) if activation is not None else nn.Sequential()

    def forward(self, x, x_sub):

        # step 1
        z_bottleneck = self.bottleneck(x, x_sub)
        z_maxpool = self.max_pool(x)
        z_maxpool = self.manifold.rescale_to_max(z_maxpool)

        # step 2
        z1 = self.conv_from_bottleneck_1(z_bottleneck, x_sub)
        z2 = self.conv_from_bottleneck_2(z_bottleneck, x_sub)
        z3 = self.conv_from_bottleneck_3(z_bottleneck, x_sub)
        z4 = self.conv_from_maxpool(z_maxpool, x_sub)

        # step 3
        z = torch.cat([z1, z2[..., 1:], z3[..., 1:], z4[..., 1:]], dim=-1)
        z = self.manifold.projx(z)
        z = self.activation(self.batch_norm(z))
        z = self.drop(z)

        z = self.manifold.rescale_to_max(z)

        return z


class EuclidInceptionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 n_filters=32,
                 kernel_sizes=(9, 19, 39),
                 bottleneck_channels=32,
                 activation=nn.ReLU(),
                 return_indices=False):

        super(EuclidInceptionBlock, self).__init__()
        self.return_indices = return_indices
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False
        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False
        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)
        self.activation = activation if activation is not None else nn.Sequential()

    def forward(self, x):
        # step 1
        z_bottleneck = self.bottleneck(x)
        z_maxpool = self.max_pool(x)

        # step 2
        z1 = self.conv_from_bottleneck_1(z_bottleneck)
        z2 = self.conv_from_bottleneck_2(z_bottleneck)
        z3 = self.conv_from_bottleneck_3(z_bottleneck)
        z4 = self.conv_from_maxpool(z_maxpool)

        # step 3
        z = torch.cat([z1, z2, z3, z4], dim=1)
        z = self.activation(self.batch_norm(z))

        return z

