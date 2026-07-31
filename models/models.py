import math

import torch
import torch.nn as nn

from hyperbolic_lib.lib.lorentz.layers import LorentzFullyConnected, LorentzFullyConnectedLora, LorentzFullyConnectedBoostLora
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
from utils.utils_h import ToHyperbolic
from utils.helpers import slice_time_series, slice_and_split_with_sub

from models.default_configs import DEFAULT_DENOISER_CONFIGS, DEFAULT_MODEL_CONFIGS




def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def fwht_inplace(x: torch.Tensor) -> torch.Tensor:
    """
    In-place FWHT (unnormalized). O(d log d) with minimal temporaries.
    x: (..., d) with d power of two. Returns same view (and also returns it).
    """
    d = x.shape[-1]
    h = 1
    orig_shape = x.shape
    while h < d:
        v = x.view(*orig_shape[:-1], -1, 2, h)
        a = v[..., 0, :]
        b = v[..., 1, :]
        tmp = a.clone()
        a.add_(b)             # a = a + b
        b.neg_().add_(tmp)    # b = tmp - b
        h <<= 1
    return x

class FastFJLT(nn.Module):
    """
    Faster FJLT: y = sqrt(d/k) * S_k * H * P * D * x
      - P: one random permutation (applied BEFORE H)
      - S_k: take first k coords (cheap slice) AFTER H
      - In-place FWHT to minimize memory traffic
      - Pads to next power-of-two but avoids work on the tail if you want (see block mode below)
    """
    def __init__(self, d: int, k: int, resample: bool = False, device=None, dtype=None,
                 block_mode: bool = False):
        super().__init__()
        assert 1 <= k <= d
        self.d = d
        self.k = k
        self.resample = resample
        self.block_mode = block_mode

        self.d_pad = _next_pow2(d)
        # random signs and permute
        D = torch.empty(self.d_pad, device=device, dtype=dtype).bernoulli_(0.5).mul_(2).add_(-1)
        perm = torch.randperm(self.d_pad, device=device)
        self.register_buffer("D", D)
        self.register_buffer("perm", perm)

        # Precompute slice for S_k
        self.k_slice = slice(0, k)

        if block_mode:
            p2 = 1 << (d.bit_length() - 1)     # <= d
            self.register_buffer("block_mask", torch.arange(self.d_pad, device=device) < p2)
            self.p2 = p2
        else:
            self.p2 = self.d_pad

    @torch.no_grad()
    def resample_params(self):
        self.D.copy_(torch.empty_like(self.D).bernoulli_(0.5).mul_(2).add_(-1))
        self.perm.copy_(torch.randperm(self.d_pad, device=self.D.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d) -> (..., k)
        """
        assert x.shape[-1] == self.d
        if self.resample:
            self.resample_params()

        if self.d_pad != self.d:
            x = torch.nn.functional.pad(x, (0, self.d_pad - self.d))

        x = x * self.D                      # D * x
        x = x.index_select(-1, self.perm)   # P * (D * x)

        if self.p2 < self.d_pad:
            xb = x[..., :self.p2]
            fwht_inplace(xb)
            xb.div_(math.sqrt(self.p2))
        else:
            fwht_inplace(x)
            x.div_(math.sqrt(self.d_pad))

        y = x[..., self.k_slice] * math.sqrt(self.d / self.k)
        return y


class RandomProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RandomProjector, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.transform = torch.randn((self.in_dim, self.out_dim))

    def forward(self, x):

        return x * self.transform


class LorentzJB(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256/4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty"):
        super().__init__()

        features = int(features)
        self.manifold = CustomLorentz(k=1, learnable=learn_k) if manifold is None else manifold

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiser(**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']

        # self.random_projector = RandomProjector(processor_output, processor_output/2)

        self.encoder = encoders.IABandEncoder(self.manifold,
                                              in_channel=processor_output/2,
                                              features=features,
                                              kernel_sizes=(9, 19, 39),
                                              inception_channels=8,
                                              conv_type=conv_type,
                                              batch_type=batch_type,
                                              pool_type=pool_type,
                                              )
        self.pre_decoder = LorentzFullyConnected(self.manifold, 14113, 2048)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 2047, n_classes)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

    def forward(self, x):

        with torch.no_grad():
            self.manifold.update_limits()

        processed = self.processor(x)
        # RANDOM PROJECTOR STILL IN PLACE
        processed = self.random_projector(processed)
        processed = self.to_hyperbolic(processed.squeeze())

        embeddings = self.encoder(processed)

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)
        flattened = self.manifold.add_time(flattened)
        # flattened = self.manifold.centroid(embeddings)

        flattened = self.pre_decoder(flattened)
        flattened = self.manifold.rescale_to_max(flattened)

        output = self.decoder(flattened)

        return output


class BaselineDeviationModel(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256/4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 num_sub=None,
                 windows=None,):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        if windows is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = windows
        self.n_classes = n_classes

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiser(**denoiser_configs[0])
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

        self.pre_decoder = LorentzFullyConnected(self.manifold, 32256, 2048)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 2047, n_classes)
        # self.decoder = decoders.LorentzMLR(self.manifold, 2047, n_classes)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True

    def get_decoder(self, x, x_sub):

        self.first_run = False

        processed = self.processor(x)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)
        flattened = self.manifold.add_time(flattened)

        self.pre_decoder = LorentzFullyConnected(self.manifold, flattened.shape[-1], 3001).to(device=x.device)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, self.n_classes)
        # self.pre_decoder = nn.Sequential()
        # self.decoder = decoders.LorentzMLR(self.manifold, 4096, self.n_classes)
        self.init_weights()

    def forward(self, x, x_sub):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)
        flattened = self.manifold.add_time(flattened)
        # flattened = self.manifold.centroid(embeddings)

        flattened = self.pre_decoder(flattened)
        flattened = self.manifold.rescale_to_max(flattened)

        output = self.decoder(flattened)

        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class BaselineDeviationModelIdEmbed(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 windows=None):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        if windows is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = windows
        self.n_classes = n_classes

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJoint(**denoiser_configs[0])
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

        self.pre_decoder = LorentzFullyConnected(self.manifold, 18033, 3001)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, n_classes)

        # self.pre_decoder.weight.weight.requires_grad = False
        # self.decoder.prototypes.requires_grad = False
        # self.decoder = decoders.LorentzMLRDecoder(self.manifold, 3000, n_classes)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True

    def get_decoder(self, x, x_sub=None):

        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.pre_decoder = LorentzFullyConnected(self.manifold, flattened.shape[-1] + 1, 3001).to(device=x.device)
        # self.pre_pre_decoder = LorentzFullyConnected(self.manifold, 3000, 512).to(device=x.device)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, self.n_classes).to(device=x.device)
        # self.pre_decoder.weight.weight.requires_grad = False
        # self.decoder = decoders.LorentzMLRDecoder(self.manifold, 3000, self.n_classes).to(device=x.device)

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
        flattened = self.manifold.add_time(flattened)
        # flattened = self.manifold.centroid(embeddings)

        flattened = self.pre_decoder(flattened)
        flattened = self.manifold.rescale_to_max(flattened)
        # flattened = self.pre_pre_decoder(flattened)

        output = self.decoder(flattened)

        return output

class BaselineDeviationModelIdEmbedLora(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 windows=None,
                 recon=None,
                 proc=None):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        rank = model_configs[0]['sub_rank']

        if windows is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = windows
        self.n_classes = n_classes

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold
        self.manifold.k.requires_grad = learnable_k
        # dirty fix line to see if the init is preventing k learning
        self.manifold = CustomLorentz(k=self.manifold.k.item(), learnable=learnable_k)
        self.n_classes = n_classes

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJointLora(recon=recon,proc=proc,rank=rank,**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']
        self.num_subs = denoiser_configs[0]['num_subjects']

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

        # self.pre_decoder = LorentzFullyConnected(self.manifold, 18033, 2001)
        # self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, n_classes)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True

    def get_decoder(self, x, x_sub=None):

        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        intermediate = 3000#2048
        # self.pre_decoder = FastFJLT(flattened.shape[-1], intermediate).to(device=x.device)
        self.pre_decoder = LorentzFullyConnected(self.manifold, flattened.shape[-1] + 1, intermediate + 1).to(device=x.device)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, intermediate, self.n_classes).to(device=x.device)

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
        flattened = self.manifold.add_time(flattened)
        flattened = self.manifold.rescale_to_max(flattened)

        flattened = self.pre_decoder(flattened)
        # flattened = self.pre_decoder(self.manifold.logmap0(flattened)[...,1:])
        # flattened = self.manifold.expmap0(nn.functional.pad(flattened, (1, 0, 0, 0)))

        output = self.decoder(flattened)

        return output


class BaselineDeviationModelIdEmbedHeadLora(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 windows=None,
                 recon=None,
                 proc=None,
                 lora_lr=None,
                 lora_type="linear"):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        rank = model_configs[0]['sub_rank']
        self.intermediate = model_configs[0]['intermediate']
        self.decoder_rank = model_configs[0]['decoder_rank']
        self.lora_lr = lora_lr
        self.lora_type = lora_type  # 'linear' or 'boost'

        if windows is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = windows
        self.n_classes = n_classes

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold
        self.manifold.k.requires_grad = learnable_k
        self.n_classes = n_classes

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJointLora(recon=recon,proc=proc,rank=rank,**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']
        self.num_subs = denoiser_configs[0]['num_subjects']

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

        # self.pre_decoder = LorentzFullyConnected(self.manifold, 18033, 2001)
        # self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, n_classes)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True

    def get_decoder(self, x, x_sub=None):

        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.intermediate
        if self.lora_type == "boost":
            self.pre_decoder = LorentzFullyConnectedBoostLora(
                self.manifold, flattened.shape[-1] + 1, self.intermediate + 1,
                num_subjects=self.num_subs, rank=self.decoder_rank, alpha=self.lora_lr
            ).to(device=x.device)
        else:  # default to "linear"
            self.pre_decoder = LorentzFullyConnectedLora(
                self.manifold, flattened.shape[-1] + 1, self.intermediate + 1,
                num_subjects=self.num_subs, rank=self.decoder_rank, lora_lr=self.lora_lr
            ).to(device=x.device)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, self.intermediate, self.n_classes).to(device=x.device)

        # self.pre_decoder = LorentzFullyConnected(self.manifold, flattened.shape[-1] + 1, self.intermediate + 1).to(device=x.device)
        # self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, self.intermediate, self.n_classes).to(device=x.device)

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
        flattened = self.manifold.add_time(flattened)
        flattened = self.manifold.rescale_to_max(flattened)

        flattened = self.pre_decoder(flattened, x_sub)

        output = self.decoder(flattened)

        return output

class BaselineDeviationModelFullLora(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 windows=None,
                 recon=None,
                 proc=None):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        if windows is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = windows
        self.n_classes = n_classes

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold
        self.n_classes = n_classes

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJointLora(proc=proc,recon=recon, rank=5, **denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = encoders.BaselineDeviationEncoderJointLora(self.manifold,
                                                                  in_channel=processor_output,
                                                                  features=features,
                                                                  kernel_sizes=(9, 19, 39),
                                                                  inception_channels=inception_channels,
                                                                  conv_type=conv_type,
                                                                  batch_type=batch_type,
                                                                  pool_type=pool_type,
                                                                  dropout=dropout,
                                                                  num_subjects=denoiser_configs[0]['num_subjects'],
                                                                  sub_rank=3,
                                                                  )

        self.pre_decoder = LorentzFullyConnected(self.manifold, 18033, 1001)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 1000, n_classes)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True

    def get_decoder(self, x, x_sub=None):

        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed, x_sub = slice_and_split_with_sub(processed, x_sub, self.windows)

        embeddings = self.encoder(processed,x_sub)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.pre_decoder = LorentzFullyConnected(self.manifold, flattened.shape[-1] + 1, 2001).to(device=x.device)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 2000, self.n_classes).to(device=x.device)

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed, x_sub = slice_and_split_with_sub(processed, x_sub, self.windows)

        embeddings = self.encoder(processed,x_sub)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)
        flattened = self.manifold.add_time(flattened)

        flattened = self.pre_decoder(flattened)
        flattened = self.manifold.rescale_to_max(flattened)

        output = self.decoder(flattened)

        return output

class BaselineDeviationModelCrossAttIdEmbedLora(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 windows=None):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        sub_rank = model_configs[0]['sub_rank']
        if windows is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = windows
        self.n_classes = n_classes

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

        self.pre_decoder = LorentzFullyConnected(self.manifold, 18033, 3001)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, n_classes)

        # self.pre_decoder.weight.weight.requires_grad = False
        # self.decoder.prototypes.requires_grad = False
        # self.decoder = decoders.LorentzMLRDecoder(self.manifold, 3000, n_classes)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True

    def get_decoder(self, x, x_sub=None):

        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())

        embeddings = self.encoder(processed)

        # embeddings: [B, embed_dim, n_windows]
        embeddings = embeddings.unsqueeze(2)  # [B, embed_dim, 1, n_windows]
        embeddings = self.manifold.centroid(embeddings, dim=-1).squeeze(2)  # [B, embed_dim, n_windows]

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)  # [B, (embed_dim-1)*n_windows]

        self.pre_decoder = LorentzFullyConnected(self.manifold, flattened.shape[-1] + 1, 3001).to(device=x.device)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, self.n_classes).to(device=x.device)

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())


        embeddings = self.encoder(processed)

        embeddings_centroid = embeddings.unsqueeze(2)  # [B, embed_dim, 1, n_windows]
        embeddings_centroid = self.manifold.centroid(embeddings_centroid, dim=-1).squeeze(2)  # [B, embed_dim, n_windows]

        flattened = embeddings_centroid[..., 1:].reshape(embeddings.shape[0], -1)  # [B, (embed_dim-1)*n_windows]
        flattened = self.manifold.add_time(flattened)  # add time component back

        flattened = self.pre_decoder(flattened)
        flattened = self.manifold.rescale_to_max(flattened)

        output = self.decoder(flattened)

        return output


class BaselineDeviationModelIdEmbedMid(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 windows=None):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        if windows is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = windows

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold
        self.n_classes = n_classes

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiser(**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = encoders.BaselineDeviationEncoderJoint(self.manifold,
                                                         in_channel=processor_output,
                                                         features=features,
                                                         kernel_sizes=(9, 19, 39),
                                                         inception_channels=inception_channels,
                                                         conv_type=conv_type,
                                                         batch_type=batch_type,
                                                         pool_type=pool_type,
                                                         dropout=dropout,
                                                         num_subjects=denoiser_configs[0]['num_subjects'],
                                                         )

        self.pre_decoder = LorentzFullyConnected(self.manifold, 18033, 3001)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, n_classes)

        # self.pre_decoder.weight.weight.requires_grad = False
        # self.decoder.prototypes.requires_grad = False
        # self.decoder = decoders.LorentzMLRDecoder(self.manifold, 3000, n_classes)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True

    def get_decoder(self, x, x_sub=None):

        self.first_run = False
        processed = self.processor(x)
        processed = self.to_hyperbolic(processed.squeeze())
        processed, x_sub = slice_and_split_with_sub(processed, x_sub, self.windows)

        embeddings = self.encoder(processed, x_sub)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.pre_decoder = LorentzFullyConnected(self.manifold, flattened.shape[-1] + 1, 3001).to(device=x.device)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, self.n_classes).to(device=x.device)
        # self.pre_decoder.weight.weight.requires_grad = False
        # self.decoder = decoders.LorentzMLRDecoder(self.manifold, 3000, self.n_classes).to(device=x.device)

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
        flattened = self.manifold.add_time(flattened)
        # flattened = self.manifold.centroid(embeddings)

        flattened = self.pre_decoder(flattened)
        flattened = self.manifold.rescale_to_max(flattened)

        output = self.decoder(flattened)

        return output


class BaselineDeviationModelIdEmbedMidLora(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 windows=None):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        sub_rank = model_configs[0]['sub_rank']
        if windows is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = windows

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold
        self.n_classes = n_classes

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiser(**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = encoders.BaselineDeviationEncoderJointLora(self.manifold,
                                                         in_channel=processor_output,
                                                         features=features,
                                                         kernel_sizes=(9, 19, 39),
                                                         inception_channels=inception_channels,
                                                         conv_type=conv_type,
                                                         batch_type=batch_type,
                                                         pool_type=pool_type,
                                                         dropout=dropout,
                                                         num_subjects=denoiser_configs[0]['num_subjects'],
                                                         sub_rank=6,
                                                         )

        self.pre_decoder = LorentzFullyConnected(self.manifold, 18033, 3001)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, n_classes)


        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True

    def get_decoder(self, x, x_sub=None):

        self.first_run = False
        processed = self.processor(x)
        processed = self.to_hyperbolic(processed.squeeze())
        processed, x_sub = slice_and_split_with_sub(processed, x_sub, self.windows)


        embeddings = self.encoder(processed, x_sub)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.pre_decoder = LorentzFullyConnected(self.manifold, flattened.shape[-1] + 1, 3001).to(device=x.device)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, self.n_classes).to(device=x.device)

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
        flattened = self.manifold.add_time(flattened)
        flattened = self.pre_decoder(flattened)
        flattened = self.manifold.rescale_to_max(flattened)

        output = self.decoder(flattened)

        return output

class BaselineDeviationModelEuclidIdEmbed(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0):
        super().__init__()

        features = int(features)
        self.manifold = CustomLorentz(k=1, learnable=False) if manifold is None else manifold
        self.n_classes = n_classes

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJoint(**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']

        self.encoder = encoders.BaselineDeviationEncoderEuclid(in_channel=processor_output,
                                                                 features=features,
                                                                 kernel_sizes=(9, 19, 39),
                                                                 inception_channels=12,
                                                                 conv_type=conv_type,
                                                                 batch_type=batch_type,
                                                                 pool_type=pool_type,
                                                                 dropout=dropout
                                                         )

        self.pre_decoder = nn.Linear(18033, 3001)
        self.decoder = nn.Linear(3000, n_classes)

        self.windows = 7
        self.first_run = True

    def get_decoder(self, x, x_sub=None):

        windows = self.windows

        self.first_run = False
        processed = self.processor(x, x_sub).squeeze().permute(0, 2, 1)

        seq_length = 50
        stride = seq_length - 5
        processed = processed.unfold(dimension=-2, size=seq_length, step=stride).squeeze().permute(0,1,3,2)
        windows = processed.shape[1]
        processed = processed.reshape(processed.shape[0]*windows, seq_length, processed.shape[-1])

        embeddings = self.encoder(processed.permute(0,2,1))

        embeddings = embeddings.reshape(x.shape[0], windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = embeddings.mean(dim=-2)

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.pre_decoder = nn.Linear(flattened.shape[-1], 3000).to(device=x.device)
        self.decoder = nn.Linear(3000, self.n_classes).to(device=x.device)


    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x, x_sub).squeeze().permute(0, 2, 1)
        # processed = processed[:,1:,:].reshape(processed.shape[0] * self.windows, -1, processed.shape[-1])

        seq_length = 50
        stride = seq_length - 5
        processed = processed.unfold(dimension=-2, size=seq_length, step=stride).squeeze().permute(0,1,3,2)
        windows = processed.shape[1]
        processed = processed.reshape(processed.shape[0]*windows, seq_length, processed.shape[-1])

        embeddings = self.encoder(processed.permute(0,2,1))

        embeddings = embeddings.reshape(x.shape[0], windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = embeddings.mean(dim=-2)

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        flattened = self.pre_decoder(flattened)
        output = self.decoder(flattened)
        return output


class BaselineDeviationModelIdLoraCbramod(nn.Module):
    def __init__(self,
                 manifold=None,
                 n_classes=3,
                 dataset="bci",
                 learn_k=False,
                 features=256 / 4,
                 conv_type="original",
                 batch_type="original",
                 pool_type="dirty",
                 dropout=0,
                 windows=None,
                 recon=None,
                 proc=None,
                 lora_lr=None,
                 lora_type="linear",
                 freeze_FM=True):
        super().__init__()

        model_configs = DEFAULT_MODEL_CONFIGS[dataset]
        inception_channels = model_configs[0]['inception_channels']
        curvature = model_configs[0]['curvature']
        learnable_k = model_configs[0]['learnable']
        rank = model_configs[0]['sub_rank']
        self.intermediate = model_configs[0]['intermediate']
        self.decoder_rank = model_configs[0]['decoder_rank']
        self.lora_lr = lora_lr
        self.lora_type = lora_type  # 'linear' or 'boost'

        if windows is None:
            self.windows = model_configs[0]['windows']
        else:
            self.windows = windows
        self.n_classes = n_classes

        features = int(features)
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_k) if manifold is None else manifold
        self.manifold.k.requires_grad = learnable_k
        self.n_classes = n_classes

        denoiser_configs = DEFAULT_DENOISER_CONFIGS[dataset]
        self.processor = EuclideanConvDenoiserJointLora(recon=recon,proc=proc,rank=rank,**denoiser_configs[0])
        processor_output = denoiser_configs[0]['out_channels']
        self.num_subs = denoiser_configs[0]['num_subjects']

        self.encoder = encoders.BaselineDeviationEncoderCbramod(self.manifold,
                                                         in_channel=processor_output,
                                                         features=features,
                                                         kernel_sizes=(9, 19, 39),
                                                         inception_channels=inception_channels,
                                                         conv_type=conv_type,
                                                         batch_type=batch_type,
                                                         pool_type=pool_type,
                                                         dropout=dropout,
                                                         freeze_FM=freeze_FM,
                                                         dataset=dataset
                                                         )

        # self.pre_decoder = LorentzFullyConnected(self.manifold, 18033, 2001)
        # self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, 3000, n_classes)

        self.to_hyperbolic = ToHyperbolic(self.manifold,
                                          norm=False,
                                          tangent_based=False)

        self.first_run = True

    def get_decoder(self, x, x_sub=None):

        self.first_run = False
        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed, x)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)

        self.intermediate
        if self.lora_type == "boost":
            self.pre_decoder = LorentzFullyConnectedBoostLora(
                self.manifold, flattened.shape[-1] + 1, self.intermediate + 1,
                num_subjects=self.num_subs, rank=self.decoder_rank, alpha=self.lora_lr
            ).to(device=x.device)
        else:  # default to "linear"
            self.pre_decoder = LorentzFullyConnectedLora(
                self.manifold, flattened.shape[-1] + 1, self.intermediate + 1,
                num_subjects=self.num_subs, rank=self.decoder_rank, lora_lr=self.lora_lr
            ).to(device=x.device)
        self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, self.intermediate, self.n_classes).to(device=x.device)

        # self.pre_decoder = LorentzFullyConnected(self.manifold, flattened.shape[-1] + 1, self.intermediate + 1).to(device=x.device)
        # self.decoder = decoders.LorentzPrototypeDecoder(self.manifold, self.intermediate, self.n_classes).to(device=x.device)

    def forward(self, x, x_sub=None):
        with torch.no_grad():
            self.manifold.update_limits()
            if self.first_run:
                self.get_decoder(x, x_sub)

        processed = self.processor(x, x_sub)
        processed = self.to_hyperbolic(processed.squeeze())
        processed = slice_time_series(processed, self.windows)

        embeddings = self.encoder(processed, x)

        embeddings = embeddings.reshape(x.shape[0], self.windows, -1, embeddings.shape[-1]).permute(0, 2, 1, 3)
        embeddings = self.manifold.centroid(embeddings).squeeze()

        flattened = embeddings[..., 1:].reshape(embeddings.shape[0], -1)
        flattened = self.manifold.add_time(flattened)
        flattened = self.manifold.rescale_to_max(flattened)

        flattened = self.pre_decoder(flattened, x_sub)

        output = self.decoder(flattened)
        return output
