import torch
import torch.nn as nn

from hyperbolic_lib.lib.lorentz.layers import LorentzFullyConnected
from hyperbolic_lib.lib.lorentz.blocks.transformer_blocks import LorentzMultiHeadAttention
from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz
from models.blocks import LorentzInceptionBlock, LorentzInceptionBlockJoint, LorentzInceptionBlockJointLora

from models.input_processors import (EuclideanConvDenoiser,
                                     EuclideanConvDenoiserJoint,
                                     EuclideanConvDenoiserJointLora,
                                     EuclideanConvDenoiserJointLoraTest,
                                     HyperbolicConvDenoiser,
                                     InputEmbedding,
                                     InputEmbeddingIDEmbd)
import models.encoders as encoders
import models.decoders as decoders
from utils.utils_h import ToHyperbolic, BATCH1D_TYPES
from utils.helpers import slice_time_series, slice_and_split_with_sub

from models.default_configs import DEFAULT_DENOISER_CONFIGS, DEFAULT_MODEL_CONFIGS


class BaselineBlockModel(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 decoder_type="prototype",
                 subject_embed_loc="pre",
                 subject_embed=None,
                 subject_dim=0,
                 slice_type=None,
                 slice_window=None,
                 slice_stride=None,
                 dropout=0):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        if slice_window is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = slice_window

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiser(**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = LorentzInceptionBlock(self.manifold,
                                             in_channels=processor_output,
                                             n_filters=int(features / 4),
                                             kernel_sizes=(9, 19, 39),
                                             bottleneck_channels=inception_channels,
                                             activation=None,
                                             return_indices=False,
                                             conv_type=conv_type,
                                             batch_type=None,
                                             pool_type="average",
                                             dropout=dropout)

        self.decoder_intermediate = 32
        self.decoder = decoders.SimpleDecoder(123, 123, 123, self.decoder_intermediate, 0.1)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True
        self.seq_length = 0

    def get_decoder(self, x, x_sub=None):

        self.first_run = False
        processed = self.processor(x)
        processed = self.to_hyperbolic(processed.squeeze())

        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.decoder = decoders.SimpleDecoder(flattened.shape[-1], x.shape[-2], x.shape[-1], self.decoder_intermediate, 0.0)
        # self.decoder = decoders.EEGDecoderTransformerSimple(
        #                 n_in=flattened.shape[-1], C=x.shape[-2], T=x.shape[-1],
        #                 d_model=128, t_down=1,   # set t_down=2 if you want a smaller MLP and upsample once
        #                 n_heads=4, p_drop=0
        #             )

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x)

        processed = self.processor(x)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        output = self.decoder(flattened)

        return output

class BaselineBlockModelIdEmb(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 decoder_type="prototype",
                 subject_embed_loc="pre",
                 subject_embed=None,
                 subject_dim=0,
                 slice_type=None,
                 slice_window=None,
                 slice_stride=None,
                 dropout=0):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        if slice_window is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = slice_window

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJoint(**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = LorentzInceptionBlock(self.manifold,
                                             in_channels=processor_output,
                                             n_filters=int(features / 4),
                                             kernel_sizes=(9, 19, 39),
                                             bottleneck_channels=inception_channels,
                                             activation=None,
                                             return_indices=False,
                                             conv_type=conv_type,
                                             batch_type=None,
                                             pool_type="average",
                                             dropout=dropout)

        self.decoder_intermediate = 32
        self.decoder = decoders.SimpleDecoder(123, 123, 123, self.decoder_intermediate, 0.1)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True
        self.seq_length = 0

    def get_decoder(self, x, x_sub=None):

        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())

        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.decoder = decoders.SimpleDecoder(flattened.shape[-1], x.shape[-2], x.shape[-1], self.decoder_intermediate,
                                              0.0)
        # self.decoder = decoders.EEGDecoderTransformerSimple(
        #                 n_in=flattened.shape[-1], C=x.shape[-2], T=x.shape[-1],
        #                 d_model=128, t_down=1,   # set t_down=2 if you want a smaller MLP and upsample once
        #                 n_heads=4, p_drop=0
        #             )

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())

        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        output = self.decoder(flattened)

        return output


class BaselineBlockModelIdEmbedLora(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 decoder_type="prototype",
                 subject_embed_loc="pre",
                 subject_embed=None,
                 subject_dim=0,
                 slice_type=None,
                 slice_window=None,
                 slice_stride=None,
                 dropout=0,
                 recon=None,
                 curvature=None,
                 ):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        if curvature is None:
            curvature = model_configs[0]['curvature']
        else:
            curvature = curvature
        learnable_k = model_configs[0]['learnable']
        if slice_window is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = slice_window

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJointLora(recon=recon, rank=model_configs[0]['sub_rank'], **denoiser_configs[0] )
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = LorentzInceptionBlock(self.manifold,
                                             in_channels=processor_output,
                                             n_filters=int(features / 4),
                                             kernel_sizes=(9, 19, 39),
                                             bottleneck_channels=inception_channels,
                                             activation=None,
                                             return_indices=False,
                                             conv_type=conv_type,
                                             batch_type=None,
                                             pool_type="average",
                                             dropout=dropout)
        self.decoder_intermediate = 32
        self.decoder = decoders.SimpleDecoder(123, 123, 123, self.decoder_intermediate, 0.1)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True
        self.seq_length = 0
        self.layer_norm = BATCH1D_TYPES["layer"](self.manifold, features + 1)

    def get_decoder(self, x, x_sub=None):
        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.decoder = decoders.SimpleDecoder(flattened.shape[-1], x.shape[-2], x.shape[-1], self.decoder_intermediate,
                                              0.0)

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)
        # embeddings = self.layer_norm(embeddings)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        # embeddings = self.manifold.rescale_to_max(embeddings)

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        output = self.decoder(flattened)

        return output

class BaselineBlockFullModelIdEmbedLora(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 decoder_type="prototype",
                 subject_embed_loc="pre",
                 subject_embed=None,
                 subject_dim=0,
                 slice_type=None,
                 slice_window=None,
                 slice_stride=None,
                 dropout=0,
                 recon=None,
                 curvature=None,
                 ):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        if curvature is None:
            curvature = model_configs[0]['curvature']
        else:
            curvature = curvature
        learnable_k = model_configs[0]['learnable']
        if slice_window is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = slice_window

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJointLora(recon=recon, rank=model_configs[0]['sub_rank'], **denoiser_configs[0] )
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = encoders.BaselineDeviationEncoder(self.manifold,
                                                         in_channel=processor_output,
                                                         features=features,
                                                         kernel_sizes=(9, 19, 39),
                                                         inception_channels=inception_channels,
                                                         conv_type=conv_type,
                                                         batch_type=batch_type,
                                                         pool_type=pool_type,
                                                         dropout=dropout
                                                         )
        self.decoder_intermediate = 32
        self.decoder = decoders.SimpleDecoder(123, 123, 123, self.decoder_intermediate, 0.1)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True
        self.seq_length = 0
        self.layer_norm = BATCH1D_TYPES["layer"](self.manifold, features + 1)

    def get_decoder(self, x, x_sub=None):
        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.decoder = decoders.SimpleDecoder(flattened.shape[-1], x.shape[-2], x.shape[-1], self.decoder_intermediate,
                                              0.0)

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)
        embeddings = self.layer_norm(embeddings)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        #
        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)
        flattened = self.manifold.add_time(flattened)
        flattened = self.manifold.rescale_to_max(flattened)[...,1:]
        #flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        output = self.decoder(flattened)

        return output

class BaselineBlockHalfModelIdEmbedLora(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 decoder_type="prototype",
                 subject_embed_loc="pre",
                 subject_embed=None,
                 subject_dim=0,
                 slice_type=None,
                 slice_window=None,
                 slice_stride=None,
                 dropout=0,
                 recon=None,
                 curvature=None,
                 ):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        if curvature is None:
            curvature = model_configs[0]['curvature']
        else:
            curvature = curvature
        learnable_k = model_configs[0]['learnable']
        if slice_window is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = slice_window

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJointLora(recon=recon, rank=model_configs[0]['sub_rank'], **denoiser_configs[0] )
        processor_output = denoiser_configs[0]['out_channels']

        in_channel = processor_output
        kernel_sizes = (9, 19, 39)

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

        self.decoder_intermediate = 32
        self.decoder = decoders.SimpleDecoder(123, 123, 123, self.decoder_intermediate, 0.1)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True
        self.seq_length = 0
        self.layer_norm = BATCH1D_TYPES["layer"](self.manifold, features + 1)

        self.activation = nn.ReLU(inplace=True)
    def get_decoder(self, x, x_sub=None):
        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        incepted = self.inception_block(processed)
        baseline = self.baseline_block(processed)

        embeddings = self.manifold.add_time(baseline[..., 1:] - incepted[..., 1:])
        #
        embeddings = self.manifold.add_time(self.activation(embeddings[..., 1:]))

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.decoder = decoders.SimpleDecoder(flattened.shape[-1], x.shape[-2], x.shape[-1], self.decoder_intermediate,
                                              0.0)

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        incepted = self.inception_block(processed)
        baseline = self.baseline_block(processed)

        embeddings = self.manifold.add_time(baseline[..., 1:] - incepted[..., 1:])
        #
        embeddings = self.manifold.add_time(self.activation(embeddings[..., 1:]))

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()


        #flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)
        #flattened = self.manifold.add_time(flattened)
        #flattened = self.manifold.rescale_to_max(flattened)[...,1:]
        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        output = self.decoder(flattened)

        return output


class BaselineBlockModelFullLora(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 decoder_type="prototype",
                 subject_embed_loc="pre",
                 subject_embed=None,
                 subject_dim=0,
                 slice_type=None,
                 slice_window=None,
                 slice_stride=None,
                 dropout=0,
                 recon=None
                 ):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        if slice_window is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = slice_window

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJointLora(recon=recon, rank=5, **denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = LorentzInceptionBlockJointLora(self.manifold,
                                                      in_channels=processor_output,
                                                      n_filters=int(features / 4),
                                                      kernel_sizes=(9, 19, 39),
                                                      bottleneck_channels=inception_channels,
                                                      activation=None,
                                                      return_indices=False,
                                                      conv_type=conv_type,
                                                      batch_type=None,
                                                      pool_type=pool_type,
                                                      dropout=dropout,
                                                      num_subjects=denoiser_configs[0]['num_subjects'],
                                                      sub_rank=6
                                                      )

        self.decoder_intermediate = 32
        self.decoder = decoders.SimpleDecoder(123, 123, 123, self.decoder_intermediate, 0.1)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True
        self.seq_length = 0

    def get_decoder(self, x, x_sub=None):
        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed,x_sub = slice_and_split_with_sub(processed, x_sub, self.windows)

        embeddings = self.encoder(processed,x_sub)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.decoder = decoders.SimpleDecoder(flattened.shape[-1], x.shape[-2], x.shape[-1], self.decoder_intermediate,
                                              0.0)

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed, x_sub = slice_and_split_with_sub(processed, x_sub, self.windows)

        embeddings = self.encoder(processed, x_sub)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        output = self.decoder(flattened)

        return output

class BaselineBlockModelSubMidLora(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 decoder_type="prototype",
                 subject_embed_loc="pre",
                 subject_embed=None,
                 subject_dim=0,
                 slice_type=None,
                 slice_window=None,
                 slice_stride=None,
                 dropout=0,
                 ):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        if slice_window is not None:
            self.windows = slice_window
        else:
            self.windows = model_configs[0]['windows']
        self.sub_rank = model_configs[0]['sub_rank']

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold
        self.n_classes = n_classes
        self.decoder_type = decoder_type

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiser(**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = LorentzInceptionBlockJointLora(self.manifold,
                                                      in_channels=processor_output,
                                                      n_filters=int(features / 4),
                                                      kernel_sizes=(9, 19, 39),
                                                      bottleneck_channels=inception_channels,
                                                      activation=None,
                                                      return_indices=False,
                                                      conv_type=conv_type,
                                                      batch_type=None,
                                                      pool_type=pool_type,
                                                      dropout=dropout,
                                                      num_subjects=denoiser_configs[0]['num_subjects'],
                                                      sub_rank=6
                                                      )

        self.decoder_intermediate = 32
        self.decoder = decoders.SimpleDecoder(123, 123, 123, self.decoder_intermediate, 0.1)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True
        self.seq_length = 0

    def get_decoder(self, x, x_sub=None):

        self.first_run = False
        processed = self.processor(x)
        processed = self.to_hyperbolic(processed.squeeze())
        processed, x_sub = slice_and_split_with_sub(processed, x_sub, self.windows)

        embeddings = self.encoder(processed, x_sub)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.decoder = decoders.SimpleDecoder(flattened.shape[-1], x.shape[-2], x.shape[-1], self.decoder_intermediate,
                                              0.0)

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x)
        processed = self.to_hyperbolic(processed.squeeze())
        processed, x_sub = slice_and_split_with_sub(processed, x_sub, self.windows)

        embeddings = self.encoder(processed, x_sub)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        output = self.decoder(flattened)

        return output


class BaselineBlockModelSubMid(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 decoder_type="prototype",
                 subject_embed_loc="pre",
                 subject_embed=None,
                 subject_dim=0,
                 slice_type=None,
                 slice_window=None,
                 slice_stride=None,
                 dropout=0):
        super().__init__()

        features = int(features)
        self.manifold = CustomLorentz(k=1, learnable=learn_k) if manifold is None else manifold
        self.n_classes = n_classes
        self.decoder_type = decoder_type
        self.subject_dim = subject_dim

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiser(**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = LorentzInceptionBlockJoint(self.manifold,
                                                  in_channels=processor_output,
                                                  n_filters=int(features / 4),
                                                  kernel_sizes=(9, 19, 39),
                                                  bottleneck_channels=10,
                                                  activation=None,
                                                  return_indices=False,
                                                  conv_type=conv_type,
                                                  batch_type=None,
                                                  pool_type="average",
                                                  dropout=dropout,
                                                  num_subjects=denoiser_configs[0]['num_subjects'],
                                                  subject_embed_dim=self.subject_dim
                                                  )

        self.decoder_intermediate = 32
        self.decoder = decoders.SimpleDecoder(123, 123, 123, self.decoder_intermediate, 0.1)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True
        self.seq_length = 0

    def get_decoder(self, x, x_sub=None):
        self.first_run = False
        processed = self.processor(x)
        processed = self.to_hyperbolic(processed.squeeze())

        processed, x_sub = slice_and_split_with_sub(processed, x_sub, self.windows)

        embeddings = self.encoder(processed, x_sub)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.decoder = decoders.SimpleDecoder(flattened.shape[-1], x.shape[-2], x.shape[-1], self.decoder_intermediate,
                                              0.0)
        # self.decoder = decoders.EEGDecoderTransformerSimple(
        #                 n_in=flattened.shape[-1], C=x.shape[-2], T=x.shape[-1],
        #                 d_model=128, t_down=1,   # set t_down=2 if you want a smaller MLP and upsample once
        #                 n_heads=4, p_drop=0
        #             )

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x)
        processed = self.to_hyperbolic(processed.squeeze())

        processed, x_sub = slice_and_split_with_sub(processed, x_sub, self.windows)

        embeddings = self.encoder(processed, x_sub)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)
        output = self.decoder(flattened)

        return output


class BaselineBlockModelCrossAttIdEmbedLora(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 decoder_type="prototype",
                 subject_embed_loc="pre",
                 subject_embed=None,
                 subject_dim=0,
                 slice_type=None,
                 slice_window=None,
                 slice_stride=None,
                 dropout=0):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        sub_rank = model_configs[0]['sub_rank']
        if slice_window is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = slice_window

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold
        self.n_classes = n_classes

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJointLora(**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = encoders.BaselineDeviationEncoderCrossPatch(self.manifold,
                                                                   in_channel=processor_output,
                                                                   features=features,
                                                                   kernel_sizes=(9, 19, 39),
                                                                   inception_channels=inception_channels,
                                                                   conv_type=conv_type,
                                                                   batch_type=batch_type,
                                                                   pool_type=pool_type,
                                                                   dropout=dropout,
                                                                   n_windows=self.windows,
                                                                   )

        self.decoder_intermediate = 32
        self.decoder = decoders.SimpleDecoder(123, 123, 123, self.decoder_intermediate, 0.1)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True
        self.seq_length = 0

    def get_decoder(self, x, x_sub=None):
        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        # embeddings: [B, embed_dim, n_windows]
        embeddings = embeddings.unsqueeze(2)  # [B, embed_dim, 1, n_windows]
        embeddings = self.manifold.centroid(embeddings, dim=-1).squeeze(2)  # [B, embed_dim, n_windows]

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)  # [B, (embed_dim-1)*n_windows]

        self.decoder = decoders.SimpleDecoder(flattened.shape[-1]+1, x.shape[-2], x.shape[-1], self.decoder_intermediate,
                                              0.0)

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings_centroid = embeddings.unsqueeze(2)  # [B, embed_dim, 1, n_windows]
        embeddings_centroid = self.manifold.centroid(embeddings_centroid, dim=-1).squeeze(2)  # [B, embed_dim, n_windows]

        flattened = embeddings_centroid[..., 1:].reshape(embeddings.shape[0], -1)  # [B, (embed_dim-1)*n_windows]
        flattened = self.manifold.add_time(flattened)  # add time component back

        output = self.decoder(flattened)

        return output