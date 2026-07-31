import sys
sys.path.append("..")

import torcheeg.transforms as transforms
import hyperbolic_lib.lib.geoopt as geoopt
import argparse
import torch
import warnings
import os
import random
import numpy as np
import torch.nn.init as init
import torch.nn as nn
from collections import defaultdict
import csv
from utils.helpers import slice_time_series, slice_and_split_with_sub
from models.input_processors import Conv2dLora

from utils.bciDataLoaderFull import BCIDataLoaderFull
from utils.bciDataLoader import BCIDataLoader
from utils.mamemDataloader import MAMEMDataLoader, MAMEMDataLoader_fixed
from utils.mamemDataloaderFull import MAMEMDataLoaderFull, MAMEMDataLoaderFull_fixed
from utils.bcichaDataLoader import BCIchaDataLoader, BCIchaDataLoader_fixed
from utils.bcichaDataLoaderFull import BCIchaDataLoaderFull, BCIchaDataLoaderFull_fixed

from models.models import (LorentzJB, BaselineDeviationModel, BaselineDeviationModelIdEmbed, BaselineDeviationModelIdEmbedMid,
                           BaselineDeviationModelIdEmbedMidLora, BaselineDeviationModelIdEmbedLora, BaselineDeviationModelCrossAttIdEmbedLora,
                            BaselineDeviationModelFullLora, BaselineDeviationModelIdEmbedHeadLora
                           )
from models.model_recon import (BaselineBlockModel, BaselineBlockModelIdEmb, BaselineBlockModelIdEmbedLora,
                                BaselineBlockModelSubMidLora,
                                BaselineBlockModelCrossAttIdEmbedLora, BaselineBlockModelFullLora,
                                BaselineBlockHalfModelIdEmbedLora, BaselineBlockFullModelIdEmbedLora)



def get_model_cross_subject(num_pred_classes, manifold=None, **kwargs):
    learn_k = True if kwargs["learnable_k"] != 0 else False
    match kwargs['model']:
        case 'BaselineDeviationModel':
            net = BaselineDeviationModel(manifold=manifold,
                                         n_classes=num_pred_classes,
                                        features=450/4,
                                        dataset=kwargs["dataset"],
                                        learn_k=learn_k,
                                        pool_type=kwargs["pool_type"],
                                        conv_type=kwargs["conv_type"],
                                        batch_type=kwargs["batch_type"],
                                        dropout=kwargs["dropout"],
                                        windows=kwargs["windows"],
                                        ).to(kwargs["device"])

            checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModel_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
        case 'BaselineDeviationModelIdEmbed':
            net = BaselineDeviationModelIdEmbed(manifold=manifold,
                                         n_classes=num_pred_classes,
                                            features=450/4,
                                            dataset=kwargs["dataset"],
                                            learn_k=learn_k,
                                            pool_type=kwargs["pool_type"],
                                            conv_type=kwargs["conv_type"],
                                            batch_type=kwargs["batch_type"],
                                            dropout=kwargs["dropout"],
                                            windows=kwargs["windows"],
                                            ).to(kwargs["device"])
            checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelIdEmb_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
        case 'BaselineDeviationModelIdEmbedLora':
            net = BaselineDeviationModelIdEmbedLora(manifold=manifold,
                                         n_classes=num_pred_classes,
                                            features=450/4,
                                            dataset=kwargs["dataset"],
                                            learn_k=learn_k,
                                            pool_type=kwargs["pool_type"],
                                            conv_type=kwargs["conv_type"],
                                            batch_type=kwargs["batch_type"],
                                            dropout=kwargs["dropout"],
                                            windows=kwargs["windows"],
                                            recon=False,
                                            proc=kwargs["pre_processor"]
                                            ).to(kwargs["device"])
            match kwargs["type"]:
                case "half":
                    checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockHalfModelIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
                case "full":
                    checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockFullModelIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
                case "default":
                    checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"


        case 'BaselineDeviationModelIdEmbedHeadLora':
            net = BaselineDeviationModelIdEmbedHeadLora(manifold=manifold,
                                         n_classes=num_pred_classes,
                                                    features=450 / 4,
                                                    dataset=kwargs["dataset"],
                                                    learn_k=learn_k,
                                                    pool_type=kwargs["pool_type"],
                                                    conv_type=kwargs["conv_type"],
                                                    batch_type=kwargs["batch_type"],
                                                    dropout=kwargs["dropout"],
                                                    windows=kwargs["windows"],
                                                    recon=False,
                                                    proc=kwargs["pre_processor"],
                                                    lora_lr=kwargs["lora_lr"],
                                                    lora_type=kwargs["lora_type"],
                                                    ).to(kwargs["device"])
            checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
        case 'BaselineDeviationModelFullLora':
            net = BaselineDeviationModelFullLora(manifold=manifold,
                                         n_classes=num_pred_classes,
                                                    features=450 / 4,
                                                    dataset=kwargs["dataset"],
                                                    learn_k=learn_k,
                                                    pool_type=kwargs["pool_type"],
                                                    conv_type=kwargs["conv_type"],
                                                    batch_type=kwargs["batch_type"],
                                                    dropout=kwargs["dropout"],
                                                    windows=kwargs["windows"],
                                                 recon=False
                                                    ).to(kwargs["device"])
            checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelFullLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
        case 'BaselineDeviationModelIdEmbedMidLora':
            net = BaselineDeviationModelIdEmbedMidLora(manifold=manifold,
                                         n_classes=num_pred_classes,
                                               features=450 / 4,
                                               dataset=kwargs["dataset"],
                                               learn_k=learn_k,
                                               pool_type=kwargs["pool_type"],
                                               conv_type=kwargs["conv_type"],
                                               batch_type=kwargs["batch_type"],
                                               dropout=kwargs["dropout"],
                                               windows=kwargs["windows"],
                                               ).to(kwargs["device"])
            checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelSubMidLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
        case 'BaselineDeviationModelCrossAttIdEmbedLora':
            net = BaselineDeviationModelCrossAttIdEmbedLora(manifold=manifold,
                                               n_classes=num_pred_classes,
                                               features=450 / 4,
                                               dataset=kwargs["dataset"],
                                               learn_k=learn_k,
                                               pool_type=kwargs["pool_type"],
                                               conv_type=kwargs["conv_type"],
                                               batch_type=kwargs["batch_type"],
                                               dropout=kwargs["dropout"],
                                               windows=kwargs["windows"],
                                               ).to(kwargs["device"])
            checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelCrossAttIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"


    return net, checkpoint


def get_model_checkpoint_name(**kwargs):

    checkpoint = None

    match kwargs['model']:
        case 'BaselineDeviationModel':
            checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModel_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
        case 'BaselineDeviationModelIdEmbed':
            checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelIdEmb_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
        case 'BaselineDeviationModelIdEmbedLora':
            match kwargs["type"]:
                case "half":
                    checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockHalfModelIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
                case "full":
                    checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockFullModelIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
                case "default":
                    checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
        case 'BaselineDeviationModelIdEmbedHeadLora':
            # checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
            match kwargs["type"]:
                case "half":
                    checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockHalfModelIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
                case "full":
                    checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockFullModelIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
                case "default":
                    checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
        case 'BaselineDeviationModelFullLora':
            checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelFullLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
        case 'BaselineDeviationModelIdEmbedMidLora':
            checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelSubMidLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"
        case 'BaselineDeviationModelCrossAttIdEmbedLora':
            checkpoint = f"./reconstruction/checkpoints/{kwargs['dataset']}_BaselineBlockModelCrossAttIdEmbedLora_w{kwargs['windows']}_pretrained_encoder_decoder_{kwargs['tag']}.pt"

    return checkpoint


def get_model_recon(**kwargs):
    s_embed_type = None if kwargs["sub"] != 'all' and not kwargs["finetune"] else kwargs["subject_embed_type"]
    learn_k = True if kwargs["learnable_k"] != 0 else False

    match kwargs['model']:
        case 'BaselineBlockModel':
            net = BaselineBlockModel(features=450 / 4,
                                         dataset=kwargs["dataset"],
                                         learn_k=learn_k,
                                         pool_type=kwargs["pool_type"],
                                         conv_type=kwargs["conv_type"],
                                         batch_type=kwargs["batch_type"],
                                         dropout=kwargs["dropout"],
                                         subject_embed=s_embed_type,
                                         subject_embed_loc=kwargs["subject_embed_loc"],
                                         subject_dim=kwargs["subject_embed_dim"],
                                         slice_type=kwargs["slice_type"],
                                         slice_window=kwargs["slice_window"],
                                         slice_stride=kwargs["slice_stride"],
                                         ).to(kwargs["device"])
        case 'BaselineBlockModelIdEmb':
            net = BaselineBlockModelIdEmb(features=450 / 4,
                                     dataset=kwargs["dataset"],
                                     learn_k=learn_k,
                                     pool_type=kwargs["pool_type"],
                                     conv_type=kwargs["conv_type"],
                                     batch_type=kwargs["batch_type"],
                                     dropout=kwargs["dropout"],
                                     subject_embed=s_embed_type,
                                     subject_embed_loc=kwargs["subject_embed_loc"],
                                     subject_dim=kwargs["subject_embed_dim"],
                                     slice_type=kwargs["slice_type"],
                                     slice_window=kwargs["slice_window"],
                                     slice_stride=kwargs["slice_stride"],
                                     ).to(kwargs["device"])
        case 'BaselineBlockModelIdEmbedLora':
            net = BaselineBlockModelIdEmbedLora(features=450 / 4,
                                                dataset=kwargs["dataset"],
                                                learn_k=learn_k,
                                                pool_type=kwargs["pool_type"],
                                                conv_type=kwargs["conv_type"],
                                                batch_type=kwargs["batch_type"],
                                                dropout=kwargs["dropout"],
                                                subject_embed=s_embed_type,
                                                subject_embed_loc=kwargs["subject_embed_loc"],
                                                subject_dim=kwargs["subject_embed_dim"],
                                                slice_type=kwargs["slice_type"],
                                                slice_window=kwargs["slice_window"],
                                                slice_stride=kwargs["slice_stride"],
                                                recon=True
                                                ).to(kwargs["device"])
        case 'BaselineBlockFullModelIdEmbedLora':
            net = BaselineBlockFullModelIdEmbedLora(features=450 / 4,
                                                dataset=kwargs["dataset"],
                                                learn_k=learn_k,
                                                pool_type=kwargs["pool_type"],
                                                conv_type=kwargs["conv_type"],
                                                batch_type=kwargs["batch_type"],
                                                dropout=kwargs["dropout"],
                                                subject_embed=s_embed_type,
                                                subject_embed_loc=kwargs["subject_embed_loc"],
                                                subject_dim=kwargs["subject_embed_dim"],
                                                slice_type=kwargs["slice_type"],
                                                slice_window=kwargs["slice_window"],
                                                slice_stride=kwargs["slice_stride"],
                                                recon=True
                                                ).to(kwargs["device"])
        case 'BaselineBlockHalfModelIdEmbedLora':
            net = BaselineBlockHalfModelIdEmbedLora(features=450 / 4,
                                                dataset=kwargs["dataset"],
                                                learn_k=learn_k,
                                                pool_type=kwargs["pool_type"],
                                                conv_type=kwargs["conv_type"],
                                                batch_type=kwargs["batch_type"],
                                                dropout=kwargs["dropout"],
                                                subject_embed=s_embed_type,
                                                subject_embed_loc=kwargs["subject_embed_loc"],
                                                subject_dim=kwargs["subject_embed_dim"],
                                                slice_type=kwargs["slice_type"],
                                                slice_window=kwargs["slice_window"],
                                                slice_stride=kwargs["slice_stride"],
                                                recon=True
                                                ).to(kwargs["device"])
        case 'BaselineBlockModelFullLora':
            net = BaselineBlockModelFullLora(features=450 / 4,
                                                    dataset=kwargs["dataset"],
                                                    learn_k=learn_k,
                                                    pool_type=kwargs["pool_type"],
                                                    conv_type=kwargs["conv_type"],
                                                    batch_type=kwargs["batch_type"],
                                                    dropout=kwargs["dropout"],
                                                    subject_embed=s_embed_type,
                                                    subject_embed_loc=kwargs["subject_embed_loc"],
                                                    subject_dim=kwargs["subject_embed_dim"],
                                                    slice_type=kwargs["slice_type"],
                                                    slice_window=kwargs["slice_window"],
                                                    slice_stride=kwargs["slice_stride"],
                                                    ).to(kwargs["device"])
        case 'BaselineBlockModelSubMidLora':
            net = BaselineBlockModelSubMidLora(features=450 / 4,
                                         dataset=kwargs["dataset"],
                                         learn_k=learn_k,
                                         pool_type=kwargs["pool_type"],
                                         conv_type=kwargs["conv_type"],
                                         batch_type=kwargs["batch_type"],
                                         dropout=kwargs["dropout"],
                                         subject_embed=s_embed_type,
                                         subject_embed_loc=kwargs["subject_embed_loc"],
                                         subject_dim=kwargs["subject_embed_dim"],
                                         slice_type=kwargs["slice_type"],
                                         slice_window=kwargs["slice_window"],
                                         slice_stride=kwargs["slice_stride"],
                                         curvature=kwargs["curvature"]
                                         ).to(kwargs["device"])
        case 'BaselineBlockModelCrossAttIdEmbedLora':
            net = BaselineBlockModelCrossAttIdEmbedLora(features=450 / 4,
                                         dataset=kwargs["dataset"],
                                         learn_k=learn_k,
                                         pool_type=kwargs["pool_type"],
                                         conv_type=kwargs["conv_type"],
                                         batch_type=kwargs["batch_type"],
                                         dropout=kwargs["dropout"],
                                         subject_embed=s_embed_type,
                                         subject_embed_loc=kwargs["subject_embed_loc"],
                                         subject_dim=kwargs["subject_embed_dim"],
                                         slice_type=kwargs["slice_type"],
                                         slice_window=kwargs["slice_window"],
                                         slice_stride=kwargs["slice_stride"],
                                         ).to(kwargs["device"])
    return net




def get_model_single_subject(**kwargs):
    pass


def fix_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

SINGLE_SUBJECT_MAP = {
    'bci': BCIDataLoader,
    'mamem': MAMEMDataLoader,
    'mamem_fixed': MAMEMDataLoader_fixed,
    'bcicha': BCIchaDataLoader,
    'bcicha_fixed': BCIchaDataLoader_fixed
}

CROSS_SUBJECT_MAP = {
    'bci': BCIDataLoaderFull,
    'mamem': MAMEMDataLoaderFull,
    'mamemfixed': MAMEMDataLoaderFull_fixed,
    'bcicha': BCIchaDataLoaderFull,
    'bcicha_fixed': BCIchaDataLoaderFull_fixed
}

import warnings




def get_dataloaders(dataset, subject, batch_size, data_path=None, interaug=False, **kwargs):
    """
    Returns train/valid/test loaders based on dataset and subject type,
    along with in_channels, num_pred_classes, and num_subjects.

    Logic:
        - subject=int -> single-subject loader (can take finetune=True/False via kwargs)
        - subject='all' -> full/full_fixed loader (finetune is ignored with a warning)
        - dataset_name ending with 'fixed' -> use the fixed version of the loader
    """
    dataset = dataset.lower()
    use_fixed = dataset.endswith('fixed')
    base_name = dataset.replace('_fixed', '')

    # Assign default dataset parameters
    if base_name in ['bci']:
        in_channels = 22
        num_pred_classes = 4
        num_subjects = 9
        default_data_path = './data/BCICIV_2a_mat/'
    elif base_name in ['mamem']:
        in_channels = 8
        num_pred_classes = 5
        num_subjects = 11
        default_data_path = './data/MAMEM/'
    elif base_name in ['bcicha']:
        in_channels = 56
        num_pred_classes = 2
        num_subjects = 16
        default_data_path = './data/BCIcha/'
    else:
        raise ValueError(f"No dataset info for: {dataset}")

    # Assign default data paths if none provided
    if data_path is None:
        data_path = default_data_path

    # Determine loader class
    if subject == 'all':
        if 'finetune' in kwargs:
            finetune_val = kwargs.pop('finetune')
            if finetune_val:
                warnings.warn(f"'finetune={finetune_val}' is ignored for full/full_fixed loaders")
        if base_name not in CROSS_SUBJECT_MAP:
            raise ValueError(f"Unknown full loader for dataset: {dataset}")
        loader_class = CROSS_SUBJECT_MAP[base_name + ('_fixed' if use_fixed else '')]
    else:
        if base_name not in SINGLE_SUBJECT_MAP:
            raise ValueError(f"Unknown base loader for dataset: {dataset}")
        loader_class = SINGLE_SUBJECT_MAP[base_name + ('_fixed' if use_fixed else '')]

    # Instantiate loader
    loader = loader_class(subject=subject, data_path=data_path, bs=batch_size, **kwargs)
    trainloader, validloader, testloader = loader.get_dataloader(interaug)

    return trainloader, validloader, testloader, in_channels, num_pred_classes, num_subjects


import os
import torch
from results.analyse import AnalyseResults  # or import get_max_val_acc

import numpy as np

def select_best_checkpoint(dataset, model, window, results_path="results"):
    """
    Select the best checkpoint based on all runs (run_*.csv) per hyperparameter config.
    Checkpoints are on the same level as `results_path`.
    """
    model_path = os.path.join(results_path, dataset, model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    best_ckpt = None
    best_acc = -np.inf

    # Walk through all nested folders (each folder = one hyperparameter config)
    for root, dirs, files in os.walk(model_path):
        # Only consider folders that match the window
        folder_name = os.path.basename(root)
        if f"win{window}_" not in folder_name:
            continue

        # Collect all run CSVs
        run_csvs = [f for f in files if f.startswith("run_") and f.endswith(".csv")]
        if not run_csvs:
            continue

        # Best validation acc among all runs in this folder
        folder_best_val = -np.inf
        for csv_file in run_csvs:
            csv_path = os.path.join(root, csv_file)
            val_acc, _, _ = AnalyseResults.get_max_val_acc(csv_path)
            if val_acc > folder_best_val:
                folder_best_val = val_acc

        # Corresponding checkpoint on the top level
        expected_ckpt_name = f"{dataset}_{model}_{folder_name}.pt"
        ckpt_path = os.path.join(results_path, expected_ckpt_name)
        if not os.path.exists(ckpt_path):
            continue

        # Update overall best
        if folder_best_val > best_acc:
            best_acc = folder_best_val
            best_ckpt = ckpt_path

    if best_ckpt is None:
        raise FileNotFoundError(
            f"No valid run CSVs or checkpoints found for {dataset}, {model}, window {window}"
        )

    return best_ckpt


def load_best_checkpoint(dataset, model, window, results_path="results", map_location="cpu"):
    """
    Load the best checkpoint.
    """
    best_ckpt_path = select_best_checkpoint(dataset, model, window, results_path)
    print(f"Loading best checkpoint from: {best_ckpt_path}")
    checkpoint = torch.load(best_ckpt_path, map_location=map_location)
    return checkpoint



def custom_initialize(model, manifold, verbose=False):
    """
    Initializes a model with mixed Euclidean and Lorentz manifold parameters.

    - Euclidean layers (Linear, Conv) → Kaiming init
    - Parameters named with 'prototype', 'baseline', etc. → expmap0 init
    - Tangent-space parameters → small normal init
    """

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                init.kaiming_normal_(module.weight, nonlinearity='relu')
                if verbose:
                    print(f"Kaiming init: {name}.weight")
            if hasattr(module, 'bias') and module.bias is not None:
                init.zeros_(module.bias)
                if verbose:
                    print(f"Zero init: {name}.bias")

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'prototype' in name or 'baseline' in name:
                # Manifold point — initialize via expmap0
                tangent = torch.randn_like(param) * 0.1
                with torch.no_grad():
                    new_val = manifold.projx(manifold.expmap0(tangent))
                    param.copy_(new_val)
                if verbose:
                    print(f"Manifold expmap0 init: {name}")
            elif param.ndim >= 2 and 'weight' in name:
                # Already handled above — skip
                continue
            elif 'bias' in name:
                # Already handled above — skip
                continue
            elif 'raw_s' in name or 'raw_r' in name:
                # Learnable scalar before softplus — init to 0
                init.constant_(param, 0)
                if verbose:
                    print(f"Softplus scalar init: {name}")
            elif param.ndim == 1 or 'logvar' in name or 'mu' in name:
                # Tangent-space vector or latent code projection
                init.normal_(param, mean=0.0, std=0.01)
                if verbose:
                    print(f"Tangent-space init: {name}")


def load_block_from_checkpoint(model, checkpoint, block_identifier):
    block_a_state_dict = {
        key.replace(block_identifier, ""): value
        for key, value in checkpoint.items()
        if key.startswith(block_identifier)
    }

    if "encoder" in block_identifier:
        print("Loading encoder block")
        model.encoder.baseline_block.load_state_dict(block_a_state_dict, strict=True)
        model.encoder.inception_block.load_state_dict(block_a_state_dict, strict=True)
    elif "processor" in block_identifier:
        print("Loading processor block")
        model.processor.load_state_dict(block_a_state_dict, strict=True)
    else:
        print("Block identifier not recognized")

def load_full_block_from_checkpoint(model, checkpoint, block_identifier):
    block_a_state_dict = {
        key.replace(block_identifier, ""): value
        for key, value in checkpoint.items()
        if key.startswith(block_identifier)
    }

    if "encoder" in block_identifier:
        print("Loading encoder block")
        model.encoder.load_state_dict(block_a_state_dict, strict=True)
    elif "processor" in block_identifier:
        print("Loading processor block")
        model.processor.load_state_dict(block_a_state_dict, strict=True)
    else:
        print("Block identifier not recognized")

def load_half_block_from_checkpoint(model, checkpoint, block_identifier):
    block_a_state_dict = {
        key.replace(block_identifier, ""): value
        for key, value in checkpoint.items()
        if key.startswith(block_identifier)
    }

    if "encoder" in block_identifier:
        print("Loading encoder block")

        # inception = {
        #     key.replace('inception_block.', ""): value
        #     for key, value in checkpoint.items()
        #     if key.startswith('inception_block.')
        # }
        baseline = {
            key.replace('baseline_block.', ""): value
            for key, value in checkpoint.items()
            if key.startswith('baseline_block.')
        }
        model.encoder.baseline_block.load_state_dict(baseline, strict=True)
        # model.encoder.inception_block.load_state_dict(inception, strict=True)
    elif "processor" in block_identifier:
        print("Loading processor block")
        model.processor.load_state_dict(block_a_state_dict, strict=True)
    else:
        print("Block identifier not recognized")



def get_param_groups(model, lr_manifold, weight_decay_manifold):
    no_decay = ["scale"]
    k_params = ".k"

    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                   and not any(nd in n for nd in no_decay)
                   and not isinstance(p, geoopt.ManifoldParameter)
                   and not n.endswith(k_params)
            ],
            "name": "1"
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if isinstance(p, geoopt.ManifoldParameter)
            ],
            'lr': lr_manifold,
            "weight_decay": weight_decay_manifold,
            "name": "manifold"
        },
        {  # k parameters
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                   and n.endswith(k_params)
            ],
            "weight_decay": weight_decay_manifold,
            "lr": lr_manifold,
            "name": "k_group"
        }
    ]

    return parameters


def clip_gradients_value(model, clip_value, losses=None):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)

    # for loss in losses:
    #     for name, p in loss.named_parameters():
    #         if p.grad is not None:
    #             p.grad.data.clamp_(min=-clip_value, max=clip_value)


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def get_dataset_statistics(dataloader):

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in dataloader:
        batch_samples = data[0].size(0)
        #data = data.view(batch_samples, data.size(1), -1)
        mean += data[0].mean(-1).sum(0)
        std += data[0].std(-1).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

def get_dataset_max_min(dataloader):

    for data in dataloader:
        max = torch.zeros(data[0].size(1))
        min = torch.zeros(data[0].size(1))
        break

    for data in dataloader:
        temp_max = data[0].max(dim=0)[0].max(dim=-1)[0]
        temp_min = data[0].min(dim=0)[0].min(dim=-1)[0]

        max = torch.max(temp_max, max)
        min = torch.min(temp_min, min)

    return max, min


def count_params(model):
    total_params = 0
    lora_total = 0
    component_params = {}

    for name, module in model.named_children():
        # total params of this top-level module
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        component_params[name] = params
        total_params += params
        print(f"{name}: {params:,} trainable parameters")

        # check for LoRA inside this module (Conv2dLora or LorentzFullyConnectedLora)
        for subname, submodule in module.named_modules():
            if hasattr(submodule, "Q") and hasattr(submodule, "R"):
                q_params = submodule.Q.numel()
                r_params = submodule.R.numel()
                lora_total += q_params + r_params
                print(f"    {name}.{subname}: {q_params+r_params:,} LoRA parameters "
                      f"(Q={q_params:,}, R={r_params:,})")

    # root-level params (not inside top-level children)
    child_param_ids = {id(p) for _, m in model.named_children() for p in m.parameters()}
    root_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad and id(p) not in child_param_ids
    )
    if root_params > 0:
        component_params["__root__"] = root_params
        total_params += root_params
        print(f"__root__: {root_params:,} trainable parameters")

    print(f"\nTotal Trainable Parameters: {total_params:,} | LoRA {lora_total:,}")
    return component_params, total_params


def get_params_groups(model, sup_loss=None, cluster_loss=None, lr=1e-5, fc_lr_scale=1, weight_decay=0.01):
    """
    divide the parameters into several groups, see below
    """
    pretrained_params = []
    last_layer = []
    proxies = []
    lcas = []
    ks = []
    manifold_params = []
    manifold_param_names = ["tanh_factor", "max_dist", "dist_scaler", "tanh_scaler"]

    num_param = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        num_param += param.numel()

        if any(ext in name for ext in manifold_param_names):
            if param not in manifold_params:
                manifold_params.append(param)
            continue

        if ".k" not in name:
            if "last" in name:
                last_layer.append(param)
            else:
                pretrained_params.append(param)
        else:
            #if param not in ks:
             ks.append(param)

    if sup_loss is not None:
        for name, param in sup_loss.named_parameters():
            if not param.requires_grad:
                continue

            if any(ext in name for ext in manifold_param_names):
                if param not in manifold_params:
                    manifold_params.append(param)
                continue

            if ".k" not in name:
                proxies.append(param)
            else:
                if param not in ks:
                    ks.append(param)

    if cluster_loss is not None:
        for name, param in cluster_loss.named_parameters():
            if not param.requires_grad:
                continue

            if any(ext in name for ext in manifold_param_names):
                if param not in manifold_params:
                    manifold_params.append(param)
                continue

            if ".k" not in name:
                lcas.append(param)
            else:
                if param not in ks:
                    ks.append(param)

    print('num_params: {:.2f}M'.format(num_param / 1e6))
    return [
        {"name": "pretrained_params", "params": pretrained_params, "lr_scale": 1, "weight_decay": weight_decay},
        {"name": "last_layer", "params": last_layer, "lr_scale": fc_lr_scale, "weight_decay": weight_decay},
        {"name": "lcas", "params": lcas, "lr_scale": 1e2, "weight_decay": weight_decay},
        {"name": "proxies", "params": proxies, "lr_scale": 1e3, "weight_decay": weight_decay},
        {"name": "k_group", "params": ks, "lr_scale": 1, "weight_decay": weight_decay},
        {"name": "man_group", "params": manifold_params, "lr_scale": 1, "weight_decay": weight_decay},
    ]


def get_loss(CE, net, out, yb, embeds=None, loss_1=None, loss_2=None):
    distances = net.manifold.dist(embeds, embeds.unsqueeze(-2))
    distance_mask = yb!=yb.unsqueeze(0).T
    increase_this = distances[distance_mask].mean()/2

    distance_mask = yb == yb.unsqueeze(0).T
    decrease_this = distances[distance_mask].mean() / 2

    distance_orig = net.manifold.dist0(embeds).mean()
    if loss_1 is not None and loss_2 is not None:
        temp = net.manifold.logmap0(embeds)[..., 1:]
        loss = loss_1(temp, yb)
        # loss = loss + loss_2(embeds, yb, 3)
        loss = 1*CE(out, yb)  + 0.1*torch.log(1+decrease_this+distance_orig)/(1+torch.log(increase_this+1)) #+ 0.1 * loss #+ 0.1*loss_2(embeds, yb, 3)# + 0.05 * loss #+ 0.3*torch.log(1+distance_orig)/(1+torch.log(additional_loss+1)) +#+ 0.5 * loss  # + 0.1*distance_orig
    else:
        loss = CE(out, yb)  # + 0.1*distance_orig
    return loss


def get_transforms(training=True,
                   dataloader=None,
                   apply_mean=False,
                   apply_minmax=False,
                   apply_noise=False,
                   mean=None,
                   std=None,
                   noise_mean=None,
                   noise_std=None,
                   max_v=None,
                   min_v=None,
                   return_statistics=False):

    t=[]
    pre_transforms = []
    post_transforms = [
                       #transforms.RandomWindowSlice(window_size=110),
                       #transforms.RandomMask(ratio=0.1),
                        #transforms.RandomPCANoise()
                       ]


    if not training:
        if apply_mean:
            t = [transforms.MeanStdNormalize(mean.numpy(), std.numpy(), axis=1)]
        if apply_minmax:
            t = [transforms.MinMaxNormalize(min_v.numpy(), max_v.numpy(), axis=1)]

        t.append(transforms.ToTensor())

        return transforms.Compose(t)

    if apply_mean:
        if mean is None:
            if dataloader is not None:
                mean, std = get_dataset_statistics(dataloader)
            else:
                print("no mean or dataloader was provided, using defaults")
        t = [transforms.MeanStdNormalize(mean.numpy(), std.numpy(), axis=1)]

    elif apply_minmax:
        if max_v is None:
            if dataloader is not None:
                max_v, min_v = get_dataset_max_min(dataloader)
            else:
                print("no max or dataloader was provided, using defaults")
        t = [transforms.MinMaxNormalize(min_v.numpy(), max_v.numpy(), axis=1)]


    #t.append(transforms.Downsample(num_points=64, axis=-1))

    t += pre_transforms
    t.append(transforms.ToTensor())
    t += post_transforms

    if apply_noise:
        t.append(transforms.RandomNoise(noise_mean, noise_std))

    if return_statistics:
        if apply_mean:
            return transforms.Compose(t), mean, std
        else:
            return transforms.Compose(t), min_v, max_v

    return transforms.Compose(t)

