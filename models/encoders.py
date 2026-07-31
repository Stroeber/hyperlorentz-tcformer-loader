import torch
import torch.nn as nn

from hyperbolic_lib.lib.geoopt import ManifoldParameter
from hyperbolic_lib.lib.lorentz.layers import LorentzFullyConnected, LorentzAct
from hyperbolic_lib.lib.Euclidean.blocks.transformer_blocks import MultiHeadAttention
from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz
from hyperbolic_lib.lib.lorentz.blocks.transformer_blocks import LorentzMultiHeadAttention
from utils.utils_h import (BATCH1D_TYPES,
                                       patch_len,
                                       matt_covar)

from models.input_processors import EuclideanConvDenoiser
from models.default_configs import DEFAULT_DENOISER_CONFIGS
from models.blocks import LorentzInceptionBlock, EuclidInceptionBlock, LorentzInceptionBlockJoint, LorentzInceptionBlockJointLora
from hyperbolic_lib.lib.lorentz.layers_1d.LConv import (
    LorentzPureConv1d,
    LorentzConv1d,
    HyperbolicCayleyConv1D,
    HyperbolicStiefelConv1D,
)
from models.random_components import LorentzMovingAverageFast, LorentzCrossAttention, EEGBaselineExtractor


class SimpleBandEncoder(nn.Module):
    def __init__(self,
                 epochs,
                 manifold=None,
                 in_channel=1,
                 features=18*18,
                 batch_type="original",
                 dataset="bci"):
        super().__init__()

        self.epochs = epochs
        self.manifold = CustomLorentz(k=1, learnable=False) if manifold is None else manifold

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        n_features = denoiser_configs[1]

        self.att = LorentzMultiHeadAttention(self.manifold,
                                             n_features + 1,
                                             1,
                                             1,
                                             out_features=features + 1)
        self.layer_norm = BATCH1D_TYPES[batch_type](self.manifold, features + 1)

        self.contribution_weights = nn.Parameter(torch.ones(in_channel), requires_grad=True)
        self.activation = nn.ReLU(inplace=True)

        self.flat = nn.Flatten()

    def forward(self, x):

        list_patch = patch_len(x.shape[-1], int(self.epochs))
        covar = torch.split(x, list_patch, dim=-1)
        covar = matt_covar(covar).permute(1, 0, 2, 3)

        covar = self.manifold.add_time(covar.reshape(covar.shape[0], covar.shape[1], -1))

        z = self.att(covar)

        z = self.manifold.add_time(self.activation(z[..., 1:]))
        z1 = self.manifold.add_time(self.flat(z[..., 1:]))

        z1 = self.manifold.rescale_to_max(z1)

        z1 = z1.reshape(x.shape[0], -1)

        return z1

class IABandEncoder(nn.Module):
    def __init__(self,
                 manifold,
                 in_channel=1,
                 features=18*18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty"):
        super().__init__()

        self.manifold = manifold

        self.inception_block = LorentzInceptionBlock(self.manifold,
                                                     in_channels=in_channel,
                                                     n_filters=int(features / 4),
                                                     kernel_sizes=kernel_sizes,
                                                     bottleneck_channels=inception_channels,
                                                     activation=None,
                                                     return_indices=False,
                                                     conv_type=conv_type,
                                                     batch_type=None,
                                                     pool_type=pool_type,
                                                     )

        self.att = LorentzMultiHeadAttention(self.manifold,
                                             features + 1,
                                             1,
                                             1,
                                             out_features=features + 1)
        self.layer_norm = BATCH1D_TYPES[batch_type](self.manifold, features + 1) if batch_type is not None else nn.Sequential()

        self.contribution_weights = nn.Parameter(torch.ones(in_channel), requires_grad=True)
        self.activation = nn.ReLU(inplace=True)

        self.flat = nn.Flatten()

    def forward(self, x):

        incepted = self.inception_block(x)
        incepted = self.manifold.add_time(self.activation(incepted[..., 1:]))
        incepted = self.layer_norm(incepted)

        atted = self.att(incepted)

        return atted

class BaselineDeviationEncoder(nn.Module):
    def __init__(self,
                 manifold,
                 in_channel=1,
                 features=18*18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0):
        super().__init__()

        self.manifold = manifold

        self.inception_block = LorentzInceptionBlock(self.manifold,
                                                     in_channels=in_channel,
                                                     n_filters=int(features / 4),
                                                     kernel_sizes=kernel_sizes,
                                                     bottleneck_channels=inception_channels,
                                                     activation=None,
                                                     return_indices=False,
                                                     conv_type=conv_type,
                                                     batch_type=None,
                                                     pool_type=pool_type,
                                                     dropout=dropout
                                                     )

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

        # init_baseline = self.manifold.projx(self.manifold.random_normal((1, 113)))
        # self.learned_baseline = ManifoldParameter(init_baseline, manifold)

        self.att = LorentzMultiHeadAttention(self.manifold,
                                             features + 1,
                                             1,
                                             1,
                                             out_features=features + 1)

        # self.att = LorentzCrossAttention(self.manifold,
        #                                  features + 1,
        #                                  num_heads=1)
        self.attention_prototypes = nn.Parameter(torch.rand(features+ 1), requires_grad=True)

        self.layer_norm = BATCH1D_TYPES["layer"](self.manifold, features + 1) #if batch_type is not None else nn.Sequential()
        self.layer_norm_2 = BATCH1D_TYPES[batch_type](self.manifold,
                                                    features + 1) if batch_type is not None else nn.Sequential()


        self.weight = nn.Parameter(torch.nn.functional.softmax(torch.randn(2)))

        self.activation = nn.ReLU(inplace=True)

        self.flat = nn.Flatten()

    def forward(self, x):


        incepted = self.inception_block(x)
        baseline = self.baseline_block(x)

        # incepted = self.manifold.rescale_to_max(incepted)
        # baseline = self.manifold.rescale_to_max(baseline)
        # baseline = self.learned_baseline

        # incepted = self.manifold.logmap0(incepted)
        # incepted = self.manifold.transp0(baseline, incepted)
        # incepted = self.manifold.expmap(baseline, incepted)


        # incepted = self.layer_norm_2(incepted)
        # baseline = self.layer_norm(baseline)
        # incepted = self.manifold.add_time(self.activation(incepted[..., 1:]))

        incepted = self.manifold.add_time(baseline[..., 1:] - incepted[..., 1:])
        #
        incepted = self.manifold.add_time(self.activation(incepted[..., 1:]))

        # incepted = self.manifold.centroid(torch.concat([baseline.unsqueeze(-2), incepted.unsqueeze(-2)], dim=-2),w=self.weight)
        # incepted = self.manifold.add_time(self.activation(incepted[..., 1:]))

        # incepted = self.layer_norm(incepted)

        # atted = self.att(incepted, self.manifold.projx(self.attention_prototypes))
        atted = self.att(incepted)
        # atted = self.manifold.add_time(self.activation(atted[..., 1:]))

        return atted

class BaselineDeviationEncoderCrossPatch(nn.Module):
    def __init__(self,
                 manifold,
                 n_windows,
                 in_channel=1,
                 features=18*18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0):
        super().__init__()

        self.manifold = manifold
        self.features = features
        self.n_windows = n_windows

        self.inception_block = LorentzInceptionBlock(
            manifold,
            in_channels=in_channel,
            n_filters=int(features / 4),
            kernel_sizes=kernel_sizes,
            bottleneck_channels=inception_channels,
            activation=None,
            return_indices=False,
            conv_type=conv_type,
            batch_type=None,
            pool_type="average",
            dropout=dropout
        )

        self.baseline_block = LorentzInceptionBlock(
            manifold,
            in_channels=in_channel,
            n_filters=int(features / 4),
            kernel_sizes=kernel_sizes,
            bottleneck_channels=inception_channels,
            activation=None,
            return_indices=False,
            conv_type=conv_type,
            batch_type=None,
            pool_type=pool_type,
            dropout=dropout
        )

        # Cross-patch attention
        self.cross_att = LorentzCrossAttention(
            manifold,
            embed_dim=features + 1,
            num_heads=1
        )

        # Layer normalization
        self.layer_norm = BATCH1D_TYPES["original"](manifold, features + 1)
        self.layer_norm_2 = BATCH1D_TYPES[batch_type](manifold, features + 1) if batch_type is not None else nn.Sequential()

        # Activation
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def split_windows(x: torch.Tensor, n_windows: int) -> torch.Tensor:
        """
        Split [B, T, C] into [B, n_windows, window_len, C] with minimal overlap
        """
        B, T, C = x.shape
        window_len = (T + n_windows - 1) // n_windows  # ceil(T / n_windows)
        stride = (T - window_len) // (n_windows - 1) if n_windows > 1 else 0

        windows = []
        for i in range(n_windows):
            start = i * stride
            end = start + window_len
            if end > T:
                start = T - window_len
                end = T
            windows.append(x[:, start:end, :])
        return torch.stack(windows, dim=1)  # [B, n_windows, window_len, C]

    def forward(self, x):
        """
        x: [B, T, C]
        Returns:
            atted: [B, embed_dim, n_windows]
        """
        B, T, C = x.shape

        # split into windows
        x_windows = self.split_windows(x, self.n_windows)  # [B, n_windows, window_len, C]
        n_windows, window_len = x_windows.shape[1], x_windows.shape[2]

        # reshape for per-patch processing
        patches = x_windows.reshape(B * n_windows, window_len, C)  # [B*n_windows, L, C]

        # apply inception and baseline blocks per patch
        incepted = self.inception_block(patches)  # [B*n_windows, window_len, features]
        baseline = self.baseline_block(patches)  # [B*n_windows, window_len, features]

        # compute deviation
        # Subtract incepted from baseline over space dimensions
        deviation = baseline[..., 1:] - incepted[..., 1:]
        deviation = self.manifold.add_time(deviation)
        deviation = self.layer_norm(deviation)
        deviation = self.manifold.add_time(self.activation(deviation[..., 1:]))

        # pool across patch length to get a single embedding per patch
        patch_embedding = deviation.mean(dim=1)  # [B*n_windows, embed_dim]

        # reshape for cross-patch attention
        seq = patch_embedding.reshape(B, n_windows, -1)  # [B, n_windows, embed_dim]

        # cross-patch self-attention
        atted = self.cross_att(seq, seq, seq)  # [B, n_windows, embed_dim]

        # permute to [B, embed_dim, n_windows] for consistency
        atted = atted.permute(0, 2, 1)

        return atted

class BaselineDeviationEncoderJoint(nn.Module):
    def __init__(self,
                 manifold,
                 in_channel=1,
                 features=18*18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 num_subjects=16):
        super().__init__()

        self.manifold = manifold

        self.subject_dim = 3
        self.subject_embeddings = nn.Embedding(num_subjects + 1, self.subject_dim, padding_idx=0)

        self.inception_block = LorentzInceptionBlock(self.manifold,
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


        self.baseline_block = LorentzInceptionBlockJoint(self.manifold,
                                                     in_channels=in_channel,
                                                     n_filters=int(features / 4),
                                                     kernel_sizes=kernel_sizes,
                                                     bottleneck_channels=inception_channels,
                                                     activation=None,
                                                     return_indices=False,
                                                     conv_type=conv_type,
                                                     batch_type=None,
                                                     pool_type=pool_type,
                                                     dropout=dropout,
                                                     num_subjects=num_subjects,
                                                     subject_embed_dim=self.subject_dim)

        self.inception_combination = LorentzInceptionBlock(self.manifold,
                                                     in_channels=(features*2)+1,
                                                     n_filters=int(features / 4),
                                                     kernel_sizes=kernel_sizes,
                                                     bottleneck_channels=inception_channels,
                                                     activation=None,
                                                     return_indices=False,
                                                     conv_type=conv_type,
                                                     batch_type=None,
                                                     pool_type="average",
                                                     dropout=dropout)
        self.inception_block2 = LorentzInceptionBlock(self.manifold,
                                                           in_channels=features,
                                                           n_filters=int(features / 4),
                                                           kernel_sizes=kernel_sizes,
                                                           bottleneck_channels=inception_channels,
                                                           activation=None,
                                                           return_indices=False,
                                                           conv_type=conv_type,
                                                           batch_type=None,
                                                           pool_type=pool_type,
                                                           dropout=dropout)
        self.inception2 = LorentzInceptionBlock(self.manifold,
                                                           in_channels=features,
                                                           n_filters=int(features / 4),
                                                           kernel_sizes=kernel_sizes,
                                                           bottleneck_channels=inception_channels,
                                                           activation=None,
                                                           return_indices=False,
                                                           conv_type=conv_type,
                                                           batch_type=None,
                                                           pool_type=pool_type,
                                                           dropout=dropout)

        # init_baseline = self.manifold.projx(self.manifold.random_normal((1, 113)))
        # self.learned_baseline = ManifoldParameter(init_baseline, manifold)

        self.att = LorentzMultiHeadAttention(self.manifold,
                                             features + 1,
                                             1,
                                             1,
                                             out_features=features + 1)

        # self.att = LorentzCrossAttention(self.manifold,
        #                                  features + 1,
        #                                  num_heads=1)
        self.attention_prototypes = nn.Parameter(torch.rand(features+ 1), requires_grad=True)

        self.layer_norm = BATCH1D_TYPES["original"](self.manifold, features + 1) #if batch_type is not None else nn.Sequential()
        self.layer_norm_2 = BATCH1D_TYPES[batch_type](self.manifold,
                                                    features + 1) if batch_type is not None else nn.Sequential()


        self.weight = nn.Parameter(torch.nn.functional.softmax(torch.randn(2)))

        self.activation = nn.ReLU(inplace=True)

        self.flat = nn.Flatten()

    def forward(self, x, x_sub):

        incepted = self.inception_block(x)
        baseline = self.baseline_block(x, x_sub)

        incepted = self.manifold.add_time(baseline[..., 1:] - incepted[..., 1:])
        incepted = self.manifold.add_time(self.activation(incepted[..., 1:]))

        atted = self.att(incepted)

        return atted

class BaselineDeviationEncoderJointLora(nn.Module):
    def __init__(self,
                 manifold,
                 in_channel=1,
                 features=18*18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 num_subjects=None,
                 sub_rank=None):
        super().__init__()

        self.manifold = manifold

        self.inception_block = LorentzInceptionBlockJointLora(self.manifold,
                                                     in_channels=in_channel,
                                                     n_filters=int(features / 4),
                                                     kernel_sizes=kernel_sizes,
                                                     bottleneck_channels=inception_channels,
                                                     activation=None,
                                                     return_indices=False,
                                                     conv_type=conv_type,
                                                     batch_type=None,
                                                     pool_type="average",
                                                     dropout=dropout,
                                                     num_subjects=num_subjects,
                                                     sub_rank=sub_rank)

        self.baseline_block = LorentzInceptionBlockJointLora(self.manifold,
                                                             in_channels=in_channel,
                                                             n_filters=int(features / 4),
                                                             kernel_sizes=kernel_sizes,
                                                             bottleneck_channels=inception_channels,
                                                             activation=None,
                                                             return_indices=False,
                                                             conv_type=conv_type,
                                                             batch_type=None,
                                                             pool_type=pool_type,
                                                            dropout=dropout,
                                                            num_subjects=num_subjects,
                                                            sub_rank=sub_rank)

        self.inception_combination = LorentzInceptionBlock(self.manifold,
                                                     in_channels=features*1,
                                                     n_filters=int(features / 4),
                                                     kernel_sizes=kernel_sizes,
                                                     bottleneck_channels=inception_channels,
                                                     activation=None,
                                                     return_indices=False,
                                                     conv_type=conv_type,
                                                     batch_type=None,
                                                     pool_type="average",
                                                     dropout=dropout)
        self.inception_combination2 = LorentzInceptionBlock(self.manifold,
                                                           in_channels=features,
                                                           n_filters=int(features / 4),
                                                           kernel_sizes=kernel_sizes,
                                                           bottleneck_channels=inception_channels,
                                                           activation=None,
                                                           return_indices=False,
                                                           conv_type=conv_type,
                                                           batch_type=None,
                                                           pool_type=pool_type,
                                                           dropout=dropout)
        self.inception2 = LorentzInceptionBlock(self.manifold,
                                                           in_channels=features,
                                                           n_filters=int(features / 4),
                                                           kernel_sizes=kernel_sizes,
                                                           bottleneck_channels=inception_channels,
                                                           activation=None,
                                                           return_indices=False,
                                                           conv_type=conv_type,
                                                           batch_type=None,
                                                           pool_type=pool_type,
                                                           dropout=dropout)

        # init_baseline = self.manifold.projx(self.manifold.random_normal((1, 113)))
        # self.learned_baseline = ManifoldParameter(init_baseline, manifold)

        self.att = LorentzMultiHeadAttention(self.manifold,
                                             features + 1,
                                             1,
                                             1,
                                             out_features=features + 1)

        self.attention_prototypes = nn.Parameter(torch.rand(features+ 1), requires_grad=True)

        self.layer_norm = BATCH1D_TYPES["original"](self.manifold, features + 1) #if batch_type is not None else nn.Sequential()
        self.layer_norm_2 = BATCH1D_TYPES[batch_type](self.manifold,
                                                    features + 1) if batch_type is not None else nn.Sequential()

        self.activation = nn.ReLU(inplace=True)

        self.flat = nn.Flatten()

    def forward(self, x, x_sub):

        # x_sub = self.subject_embeddings(x_sub.int()).squeeze()  # .permute(0, 1, 3, 2)
        # x = torch.cat([x, x_sub], dim=-1)
        incepted = self.inception_block(x, x_sub)
        baseline = self.baseline_block(x, x_sub)

        # incepted = self.manifold.rescale_to_max(incepted)
        # baseline = self.manifold.rescale_to_max(baseline)
        # baseline = self.learned_baseline

        # incepted = self.manifold.logmap0(incepted)
        # incepted = self.manifold.transp0(baseline, incepted)
        # incepted = self.manifold.expmap(baseline, incepted)



        # incepted = self.layer_norm_2(incepted)
        # z = self.manifold.add_time(baseline[..., 1:] - incepted[..., 1:])
        # combined = self.inception_combination2(z)
        #
        # z = torch.cat([incepted[..., 1:], baseline], dim=-1)
        # z = self.manifold.projx(z)
        # combined = self.inception_combination(z)
        # incepted = self.manifold.add_time(combined)

        incepted = self.manifold.add_time(baseline[..., 1:] - incepted[..., 1:])
        incepted = self.layer_norm(incepted)
        incepted = self.manifold.add_time(self.activation(incepted[..., 1:]))

        # print(baseline.shape)
        # print(incepted.shape)
        # incepted = self.manifold.centroid(torch.concat([baseline.unsqueeze(-2), incepted.unsqueeze(-2)], dim=-2),w=self.weight)
        # incepted = self.manifold.add_time(self.activation(incepted[..., 1:]))

        # incepted = self.layer_norm(incepted)

        # atted = self.att(incepted, self.manifold.projx(self.attention_prototypes))
        atted = self.att(incepted)
        # atted = self.manifold.add_time(self.activation(atted[..., 1:]))

        return atted

class BaselineDeviationEncoderEuclid(nn.Module):
    def __init__(self,
                 in_channel=1,
                 features=18*18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0):
        super().__init__()

        self.inception_block = EuclidInceptionBlock(in_channels=in_channel,
                                                     n_filters=int(features / 4),
                                                     kernel_sizes=kernel_sizes,
                                                     bottleneck_channels=inception_channels,
                                                     activation=None,
                                                     return_indices=False)

        self.baseline_block = EuclidInceptionBlock(in_channels=in_channel,
                                                     n_filters=int(features / 4),
                                                     kernel_sizes=kernel_sizes,
                                                     bottleneck_channels=inception_channels,
                                                     activation=None,
                                                     return_indices=False)

        self.att = MultiHeadAttention(features, 1, 1)


    def forward(self, x):


        incepted = self.inception_block(x).permute(0,2,1)
        baseline = self.baseline_block(x).permute(0,2,1)

        atted = self.att(incepted + baseline)

        return atted

class TestEncoder(nn.Module):
    def __init__(self, manifold,
                 in_channel=1,
                 features=18*18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0):
        super().__init__()
        self.manifold = manifold

        # Signal processing pathway
        self.signal_path = nn.Sequential(
            LorentzInceptionBlock(self.manifold,
                                  in_channels=in_channel,
                                  n_filters=int(features / 4),
                                  kernel_sizes=kernel_sizes,
                                  bottleneck_channels=inception_channels,
                                  activation=None,
                                  return_indices=False,
                                  conv_type=conv_type,
                                  batch_type=None,
                                  pool_type=pool_type,
                                  dropout=dropout
                                  ),
            # LorentzInceptionBlock(self.manifold,
            #                       in_channels=int(features),
            #                       n_filters=int(features / 4),
            #                       kernel_sizes=kernel_sizes,
            #                       bottleneck_channels=inception_channels,
            #                       activation=None,
            #                       return_indices=False,
            #                       conv_type=conv_type,
            #                       batch_type=None,
            #                       pool_type=pool_type,
            #                       dropout=dropout
            #                       )
        )

        # Baseline pathway
        self.baseline_path = nn.Sequential(
            # LorentzMovingAverageFast(manifold, window_size=31),
            LorentzInceptionBlock(self.manifold,
                                  in_channels=in_channel,
                                  n_filters=int(features / 4),
                                  kernel_sizes=(9, 19, 39),
                                  bottleneck_channels=inception_channels,
                                  activation=nn.ReLU(inplace=True),
                                  return_indices=False,
                                  conv_type=conv_type,
                                  batch_type=None,
                                  pool_type="average",
                                  dropout=dropout
                                  )
            # LorentzAct(nn.ReLU(inplace=True), manifold)
        )

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            LorentzCrossAttention(manifold, features+1, 1)
            for _ in range(1)
        ])

        self.att = LorentzMultiHeadAttention(self.manifold,
                                             features + 1,
                                             1,
                                             1,
                                             out_features=features + 1)

        # # Temporal aggregation
        # self.temporal_pool = LorentzAdaptivePool(manifold, 'max')
        #
        # # Deviation metric learning
        # self.deviation_metric = nn.Sequential(
        #     LorentzLinear(manifold, embed_dim, embed_dim // 2),
        #     LorentzAct(nn.ReLU(), manifold),
        #     LorentzLinear(manifold, embed_dim // 2, 1)
        # )

        self.windows = 3
        # self.manifold_parameter = None
        self.register_buffer("manifold_parameter", tensor=None)

    def forward(self, x):
        # Process both pathways
        signal_emb = self.signal_path(x)
        baseline_emb = self.baseline_path(x)


        # if self.manifold_parameter is None:
        #     manifold_parameter = self.manifold.centroid(baseline_emb.permute(1,0,2)).unsqueeze(-2)
        #     with torch.no_grad():
        #         self.manifold_parameter = manifold_parameter
        # else:
        #     temp_param = self.manifold.centroid(baseline_emb.permute(1,0,2)).unsqueeze(-2)
        #     manifold_parameter = self.manifold.centroid(torch.cat([temp_param, self.manifold_parameter], dim=-2)).unsqueeze(-2)
        #     with torch.no_grad():
        #         self.manifold_parameter = manifold_parameter

        # manifold_parameter = self.manifold.centroid(baseline_emb.permute(1, 0, 2)).unsqueeze(-2)
        # baseline_emb = self.manifold.centroid()
        # signal_emb = self.att(signal_emb)

        out = self.manifold.add_time(signal_emb[..., 1:] + baseline_emb[..., 1:])
        # Multi-layer cross-attention
        # for attn_layer in self.cross_attn_layers:
        #     signal_emb = attn_layer( self.manifold_parameter.permute(1,0,2),signal_emb,  signal_emb)

        out = self.att(out)
        # # Aggregate temporal features
        # pooled = self.temporal_pool(signal_emb)
        #
        # # Compute baseline deviation
        # deviation = self.manifold.dist(pooled, self.temporal_pool(baseline_emb))
        return out

from .cbramod import CBraMod
import torch.nn.functional as F
class BaselineDeviationEncoderCbramod(nn.Module):
    def __init__(self,
                 manifold,
                 in_channel=1,
                 features=18*18,
                 kernel_sizes=(9, 19, 39),
                 inception_channels=8,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 num_patches=1,
                 freeze_FM=True,
                 dataset='mamem'
                 ):
        super().__init__()

        self.manifold = manifold
        self.patch_length = 200

        # Frozen CBraMod as baseline
        self.baseline_encoder = CBraMod()
        self.baseline_encoder.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location='cuda:0'))
        self.baseline_encoder.proj_out = nn.Identity()
        if freeze_FM:
            for param in self.baseline_encoder.parameters():
                param.requires_grad = False

        # Project CBraMod output from (batch, in_channels_eeg, num_patches*200)
        # to match inception output shape (batch, features, features+1)

        if dataset == 'mamem':
            in_channel_eeg = 8
            proj_dim = 126
        elif dataset == 'bci':
            in_channel_eeg = 22
            proj_dim = 439
        elif dataset == 'bcicha':
            in_channel_eeg = 56
            proj_dim = 161
        self.baseline_proj = nn.Sequential(
            nn.Conv1d(in_channel_eeg, proj_dim, kernel_size=3, padding=1),  # (batch, 126, 200)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(features + 1),  # (batch, 126, 113)
        )

        self.inception_block = LorentzInceptionBlock(self.manifold,
                                                     in_channels=in_channel,
                                                     n_filters=int(features / 4),
                                                     kernel_sizes=kernel_sizes,
                                                     bottleneck_channels=inception_channels,
                                                     activation=None,
                                                     return_indices=False,
                                                     conv_type=conv_type,
                                                     batch_type=None,
                                                     pool_type=pool_type,
                                                     dropout=dropout)

        self.att = LorentzMultiHeadAttention(self.manifold,
                                             features + 1,
                                             1,
                                             1,
                                             out_features=features + 1)

        self.attention_prototypes = nn.Parameter(torch.rand(features + 1), requires_grad=True)

        self.layer_norm = BATCH1D_TYPES["layer"](self.manifold, features + 1)
        self.layer_norm_2 = BATCH1D_TYPES[batch_type](self.manifold,
                                                       features + 1) if batch_type is not None else nn.Sequential()

        self.weight = nn.Parameter(torch.nn.functional.softmax(torch.randn(2)))
        self.activation = nn.ReLU(inplace=True)
        self.flat = nn.Flatten()

    def _prepare_for_cbramod(self, x):
        """Pad and patch input for CBraMod if needed."""
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() == 3:
            bz, ch, time = x.shape
            remainder = time % self.patch_length
            if remainder != 0:
                x = F.pad(x, (0, self.patch_length - remainder), value=0)
            num_patches = x.shape[-1] // self.patch_length
            x = x.reshape(bz, ch, num_patches, self.patch_length)
        return x

    def forward(self, z, x):
        # Get CBraMod baseline
        x_cbra = self._prepare_for_cbramod(x)
        with torch.no_grad():
            baseline_feats = self.baseline_encoder(x_cbra)  # (batch, 8, 1, 200)

        # Project to match inception output shape (batch, 126, features+1)
        baseline = baseline_feats.squeeze(2)  # (batch, 8, 200)
        baseline = torch.flatten(baseline_feats, start_dim=-2)
        baseline = self.baseline_proj(baseline)  # (batch, 126, 113)
        baseline = self.manifold.projx(baseline)

        # Lorentz inception
        incepted = self.inception_block(z)  # (batch, 126, 113)
        # Deviation
        incepted = self.manifold.add_time(baseline[..., 1:] - incepted[..., 1:])
        incepted = self.manifold.add_time(self.activation(incepted[..., 1:]))

        atted = self.att(incepted)
        return atted