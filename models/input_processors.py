import torch
import torch.nn as nn
import numpy as np

from utils.utils_h import CONV2D_TYPES, BATCH2D_TYPES


class signal2spd(nn.Module):

    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        cov = x @ x.permute(0, 2, 1)
        cov = cov / (x.shape[-1] - 1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov /= tra
        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=x.device()).repeat(x.shape[0], 1, 1)
        cov = cov + (1e-5 * identity)
        return cov


class E2R(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()

    def patch_len(self, n, epochs):
        list_len = []
        base = n // epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base * epochs):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')

    def forward(self, x):
        # x with shape[bs, ch, time]
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-1))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dLora(nn.Module):
    def __init__(self, conv_layer, rank=3, num_subjects=1, recon=None, proc=None):
        """
        Wraps a Conv2d layer with LoRA adapters.
        Args:
            conv_layer: existing nn.Conv2d layer
            rank: low-rank dimension K
            num_subjects: number of subjects
        """
        super().__init__()
        self.shared_conv = conv_layer
        self.rank = rank
        self.num_subjects = num_subjects

        # Create subject-specific low-rank factors Q, R for each subject
        # Q: (out_channels, rank, 1, 1)   → expands to filters
        # R: (rank, in_channels, kH, kW)
        self.Q = nn.Parameter(torch.zeros(num_subjects+1,
                                          conv_layer.out_channels,
                                          rank, 1, 1))
        self.R = nn.Parameter(torch.zeros(num_subjects+1,
                                          rank,
                                          conv_layer.in_channels,
                                          *conv_layer.kernel_size))

        # LoRA adapters: initialized to zero
        if recon:
            nn.init.normal_(self.Q)
            nn.init.zeros_(self.R)
        elif proc == False:
            nn.init.normal_(self.Q)
            nn.init.zeros_(self.R)

    def forward(self, x, subject_idx):
        """
        x: [B, C, H, W]
        subject_idx: [B] tensor of subject indices
        """
        out = self.shared_conv(x)

        # Subject-specific adapters
        Qs = self.Q[subject_idx]  # [B, out_c, rank, 1, 1]
        Rs = self.R[subject_idx]  # [B, rank, in_c, kH, kW]
        lora_weight = torch.einsum("borhw,bricd->boicd", Qs, Rs)  # [B, out_c, in_c, kH, kW]

        B, out_c, in_c, kH, kW = lora_weight.shape

        # Reshape for grouped conv
        x_reshaped = x.view(1, B * in_c, *x.shape[2:])  # [1, B*in_c, H, W]
        w_reshaped = lora_weight.view(B * out_c, in_c, kH, kW)  # [B*out_c, in_c, kH, kW]

        lora_out = F.conv2d(x_reshaped,
                            w_reshaped,
                            bias=None,
                            stride=self.shared_conv.stride,
                            padding=self.shared_conv.padding,
                            dilation=self.shared_conv.dilation,
                            groups=B)

        # Reshape back: [1, B*out_c, H', W'] → [B, out_c, H', W']
        lora_out = lora_out.view(B, out_c, *lora_out.shape[2:])

        out = out + lora_out
        return out


class EuclideanConvDenoiserJointLora(nn.Module):
    def __init__(self,
                 in_channels,
                 intermediate_channels,
                 out_channels,
                 kernel_timewise,
                 kernel_channelwise,
                 padding,
                 num_subjects=1,
                 rank=3,
                 recon=None,
                 proc=None
                 ):
        super(EuclideanConvDenoiserJointLora, self).__init__()

        base_conv1 = nn.Conv2d(in_channels,
                               intermediate_channels,
                               kernel_size=(kernel_timewise, 1))
        self.conv1 = Conv2dLora(base_conv1, rank=rank, num_subjects=num_subjects, recon=recon, proc=proc)
        self.Bn1 = nn.BatchNorm2d(intermediate_channels)

        base_conv2 = nn.Conv2d(intermediate_channels,
                               out_channels,
                               kernel_size=(1, kernel_channelwise),
                               padding=(0, padding))
        self.conv2 = Conv2dLora(base_conv2, rank=rank, num_subjects=num_subjects, recon=recon, proc=proc)
        self.Bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, x_sub):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        subj_idx = x_sub.int() if len(x_sub.shape) == 1 else x_sub[:, 0, 0, 0].int()

        x = self.conv1(x, subj_idx)
        x = self.Bn1(x)

        x = self.conv2(x, subj_idx)
        x = self.Bn2(x)
        return x

class EuclideanConvDenoiser(nn.Module):
    def __init__(self,
                 in_channels,
                 intermediate_channels,
                 out_channels,
                 kernel_timewise,
                 kernel_channelwise,
                 padding,
                 num_subjects=1,
                 subject_embed=None,
                 subject_dim=16):
        super(EuclideanConvDenoiser, self).__init__()

        if subject_embed is not None:
            self.subject_embeddings, kernel_timewise = subject_embed(num_subjects, subject_dim)

        # we could merge these into a seq. list but this is easier for debugging
        self.conv1 = nn.Conv2d(in_channels,
                               intermediate_channels,
                               kernel_size=(kernel_timewise, 1))
        self.Bn1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(intermediate_channels,
                               out_channels,
                               kernel_size=(1, kernel_channelwise),
                               padding=(0, padding))
        self.Bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.Bn1(x)

        x = self.conv2(x)
        x = self.Bn2(x)
        return x


class HyperbolicConvDenoiser(nn.Module):
    def __init__(self,
                 manifold,
                 in_channels,
                 intermediate_channels,
                 out_channels,
                 kernel_timewise,
                 kernel_channelwise,
                 padding,
                 num_subjects=1,
                 conv_type="original",
                 batch_type="original",
                 subject_embed=None,
                 subject_dim=16):
        super(HyperbolicConvDenoiser, self).__init__()

        if subject_embed is not None:
            self.subject_embeddings, kernel_timewise = subject_embed(num_subjects, subject_dim)


        self.manifold = manifold
        # we could merge these into a seq. list but this is easier for debugging
        self.conv1 = CONV2D_TYPES[conv_type](manifold,
                                             in_channels,
                                             intermediate_channels+1,
                                             kernel_size=(kernel_timewise, 1))

        self.Bn1 = BATCH2D_TYPES[batch_type](manifold,
                                             intermediate_channels+1)

        self.conv2 = CONV2D_TYPES[conv_type](manifold,
                                             intermediate_channels+1,
                                             out_channels+1,
                                             kernel_size=(1, kernel_channelwise),
                                             padding=(0, padding))
        self.Bn2 = BATCH2D_TYPES[batch_type](manifold,
                                             out_channels+1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = x.permute(0, 2, 3, 1)
        x = self.manifold.add_time(x)

        x = self.conv1(x)
        x = self.Bn1(x)

        x = self.conv2(x)
        x = self.Bn2(x)

        return x


class FiLM(nn.Module):
    def __init__(self, num_subjects, feature_dim):
        super().__init__()
        self.gamma = nn.Embedding(num_subjects + 1, feature_dim)
        self.beta = nn.Embedding(num_subjects + 1, feature_dim)

    def forward(self, x, subject_ids):
        """
        x: (B, C, 1, T)
        subject_ids: (B, 1, 1, T) → e.g., one subject ID per time step
        """
        B, C, _, T = x.shape
        subject_ids = subject_ids.squeeze(1).squeeze(1)  # (B, T)

        gamma = self.gamma(subject_ids)  # (B, T, C)
        # beta = self.beta(subject_ids)  # (B, T, C)

        # Reshape to (B, C, 1, T)
        gamma = gamma.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, T)
        # beta = beta.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, T)

        return gamma * x[...,:-1]

class EuclideanConvDenoiserJoint(nn.Module):
    def __init__(self,
                 in_channels,
                 intermediate_channels,
                 out_channels,
                 kernel_timewise,
                 kernel_channelwise,
                 padding,
                 num_subjects=1,
                 subject_embed=None,
                 subject_dim=3):
        super(EuclideanConvDenoiserJoint, self).__init__()

        if subject_embed is not None:
            self.subject_embeddings, kernel_timewise = subject_embed(num_subjects, subject_dim)

        # we could merge these into a seq. list but this is easier for debugging
        self.conv1 = nn.Conv2d(in_channels,
                               intermediate_channels,
                               # kernel_size=(kernel_timewise  + subject_dim, 1))
                                kernel_size = (kernel_timewise, 1))
        self.Bn1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(intermediate_channels,
                               out_channels,
                               kernel_size=(1, kernel_channelwise),
                               padding=(0, padding))
        self.Bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(0.5)

        self.sub = FiLM(num_subjects, out_channels)

        self.subject_embeddings = nn.Embedding(num_subjects + 1, subject_dim, padding_idx=0)

    def forward(self, x, x_sub=None):

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # x_sub = self.subject_embeddings(x_sub.int()).squeeze(1).permute(0, 1, 3,2)
        # x = torch.cat([x, x_sub], dim=-2)

        x = self.conv1(x)
        x = self.Bn1(x)

        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.sub(x, x_sub.int())

        return x

class EuclideanConvDenoiserJointLoraTest(nn.Module):
    def __init__(self,
                 in_channels,
                 intermediate_channels,
                 out_channels,
                 kernel_timewise,
                 kernel_channelwise,
                 padding,
                 num_subjects=1,
                 rank=3):
        super(EuclideanConvDenoiserJointLoraTest, self).__init__()

        self.base_conv1 = nn.Conv2d(in_channels,
                               intermediate_channels,
                               kernel_size=(kernel_timewise, 1))
        self.conv1 = Conv2dLora(self.base_conv1, rank=rank, num_subjects=num_subjects)
        self.Bn1 = nn.BatchNorm2d(intermediate_channels)

        self.base_conv2 = nn.Conv2d(intermediate_channels,
                               out_channels,
                               kernel_size=(1, kernel_channelwise),
                               padding=(0, padding))
        self.conv2 = Conv2dLora(self.base_conv2, rank=out_channels, num_subjects=num_subjects)
        self.Bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, x_sub):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        subj_idx = x_sub[:, 0, 0, 0].int()

        x = self.base_conv1(x)
        x = self.Bn1(x)

        x = self.conv2(x, subj_idx)
        x = self.Bn2(x)
        return x

class EuclideanMattPreJoint(nn.Module):
    def __init__(self,
                 in_channels,
                 intermediate_channels,
                 out_channels,
                 kernel_timewise,
                 kernel_channelwise,
                 padding,
                 num_subjects=1,
                 subject_embed=None,
                 subject_dim=3,
                 segments=3):
        super(EuclideanMattPreJoint, self).__init__()

        if subject_embed is not None:
            self.subject_embeddings, kernel_timewise = subject_embed(num_subjects, subject_dim)

        self.subject_embeddings = nn.Embedding(num_subjects + 1, subject_dim, padding_idx=0)

        # we could merge these into a seq. list but this is easier for debugging
        self.conv1 = nn.Conv2d(in_channels,
                               intermediate_channels,
                               kernel_size=(kernel_timewise + subject_dim, 1))
        self.Bn1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(intermediate_channels,
                               out_channels,
                               kernel_size=(1, kernel_channelwise),
                               padding=(0, padding))
        self.Bn2 = nn.BatchNorm2d(out_channels)

        self.ract1 = E2R(segments)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, x_sub=None):

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x_sub = self.subject_embeddings(x_sub.int()).squeeze(1).permute(0, 1, 3,2)
        x = torch.cat([x, x_sub], dim=-2)

        x = self.conv1(x)
        x = self.Bn1(x)

        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.ract1(x)

        return x


class HyperInceptionDenoiser(nn.Module):
    def __init__(self,
                 in_channels,
                 intermediate_channels,
                 out_channels,
                 kernel_timewise,
                 kernel_channelwise,
                 padding,
                 num_subjects=1,
                 subject_embed=None,
                 subject_dim=16):
        super(HyperInceptionDenoiser, self).__init__()

        if subject_embed is not None:
            self.subject_embeddings, kernel_timewise = subject_embed(num_subjects, subject_dim)

        # we could merge these into a seq. list but this is easier for debugging
        self.conv1 = nn.Conv2d(in_channels,
                               intermediate_channels,
                               kernel_size=(kernel_timewise, 1))
        self.Bn1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(intermediate_channels,
                               out_channels,
                               kernel_size=(1, kernel_channelwise),
                               padding=(0, padding))
        self.Bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.conv1(x)
        x = self.Bn1(x)

        x = self.conv2(x)
        x = self.Bn2(x)
        return x


class NoneDenoiser(nn.Module):
    def __init__(self):
        super(NoneDenoiser, self).__init__()

    def forward(self, x):
        return x.squeeze()



class InputEmbedding(nn.Module):
    def __init__(self,
                 out_channels,
                 kernel_timewise,
                 kernel_channelwise,
                 average_pool=False,
                 num_subjects=1,
                 subject_embed=None,
                 subject_dim=16,
                 padding=0,
                 **kwargs):
        super().__init__()
        k = 36

        # Embedding Layer -----------------------------------------------------------
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_timewise, 1))
        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_channelwise), padding=(0, padding))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_channelwise//4), padding=(0, padding//4))
        self.SiLU = nn.SiLU(inplace=True)

        self.norm_1 = nn.LayerNorm(125)
        self.norm_2 = nn.BatchNorm2d(1)
        self.norm_3 = nn.BatchNorm2d(1)

        if not average_pool:
            self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        else:
            self.maxpool = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3))

    def forward(self, x):
        x = x.unsqueeze(1)

        out = self.depthwise_conv(x)  # (bs, embedding, 1 , T)
        out = self.norm_1(out)

        out = out.transpose(1, 2)  # (bs, 1, embedding, T)

        # out = self.spatial_padding(out)
        out = self.spatialwise_conv1(out)  # (bs, 1, embedding, T)
        # out = self.norm_2(out)

        out = self.SiLU(out)
        out = self.maxpool(out)  # (bs, 1, embedding, T // m)


        out = self.spatial_padding(out)
        out = self.spatialwise_conv2(out)

        # out = self.norm_3(out)

        out = out.squeeze(1)  # (bs, embedding, T // m)
        out = out.transpose(1, 2)  # (bs, T // m, embedding)
        patches = self.SiLU(out)
        return patches


class InputEmbeddingIDEmbd(nn.Module):
    def __init__(self,
                 out_channels,
                 kernel_timewise,
                 kernel_channelwise,
                 average_pool=False,
                 num_subjects=1,
                 subject_embed=None,
                 subject_dim=16,
                 padding=0,
                 **kwargs):
        super().__init__()
        k = 36

        if subject_embed is not None:
            self.subject_embeddings, kernel_timewise = subject_embed(num_subjects, subject_dim)

        self.subject_embeddings = nn.Embedding(num_subjects + 1, subject_dim, padding_idx=0)

        # Embedding Layer -----------------------------------------------------------
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_timewise + subject_dim, 1))
        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_channelwise), padding=(0, padding))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_channelwise), padding=(0, padding))
        self.SiLU = nn.SiLU(inplace=True)

        self.norm_1 = nn.BatchNorm2d(15)
        self.norm_2 = nn.BatchNorm2d(1)
        self.norm_3 = nn.BatchNorm2d(1)

        if not average_pool:
            self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        else:
            self.maxpool = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3))

    def forward(self, x, x_sub):
        x = x.unsqueeze(1)

        x_sub = self.subject_embeddings(x_sub.int()).squeeze(1).permute(0, 1, 3,2)
        out = torch.cat([x, x_sub], dim=-2)

        out = self.depthwise_conv(out)  # (bs, embedding, 1 , T)
        # out = self.norm_1(out)

        out = out.transpose(1, 2)  # (bs, 1, embedding, T)

        # out = self.spatial_padding(out)
        out = self.spatialwise_conv1(out)  # (bs, 1, embedding, T)
        # out = self.norm_2(out)

        out = self.SiLU(out)
        out = self.maxpool(out)  # (bs, 1, embedding, T // m)


        out = self.spatial_padding(out)
        out = self.spatialwise_conv2(out)

        # out = self.norm_3(out)

        out = out.squeeze(1)  # (bs, embedding, T // m)
        out = out.transpose(1, 2)  # (bs, T // m, embedding)
        patches = self.SiLU(out)
        return patches

