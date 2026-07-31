import sys
from utils.pytorch_module import create_model
sys.path.append("../")
import argparse
from trainer import trainNetwork
from utils.utils import (fix_seed, get_dataloaders)

import types
import torch

torch_scatter = types.ModuleType("torch_scatter")

# --- scatter_add / scatter_sum ---
def scatter_add(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    out_shape = list(src.shape)
    out_shape[dim] = dim_size

    out = src.new_zeros(out_shape)
    return out.index_add(dim, index, src)

# PyG / many libs alias scatter_sum -> scatter_add
scatter_sum = scatter_add


# --- scatter_mean ---
def scatter_mean(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    out_shape = list(src.shape)
    out_shape[dim] = dim_size

    out = src.new_zeros(out_shape)
    count = src.new_zeros(out_shape)

    out = out.index_add(dim, index, src)
    count = count.index_add(dim, index, torch.ones_like(src))

    return out / count.clamp(min=1)


# --- scatter_max (approx via scatter_reduce if available) ---
def scatter_max(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    if hasattr(torch.Tensor, "scatter_reduce"):
        out = torch.full(
            (dim_size,) + src.shape[1:] if dim == 0 else src.shape[:dim] + (dim_size,) + src.shape[dim+1:],
            -torch.inf,
            device=src.device,
            dtype=src.dtype,
        )
        out = out.scatter_reduce(dim, index, src, reduce="amax", include_self=True)
        return out

    raise NotImplementedError("scatter_max fallback requires torch >= 2.0 with scatter_reduce")


# register API
torch_scatter.scatter_add = scatter_add
torch_scatter.scatter_sum = scatter_sum
torch_scatter.scatter_mean = scatter_mean
torch_scatter.scatter_max = scatter_max

# inject into import system
sys.modules["torch_scatter"] = torch_scatter


from torcheeg.models import Conformer, ATCNet






def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cuda:0', help='Torch Device for computations')
    ap.add_argument('--repeat', type=int, default=1, help='No.xxx repeat for training model')
    ap.add_argument('--sub', type=str, default=2, help='subjectxx you want to train')
    ap.add_argument('--iterations', type=int, default=5, help='number of training iterations')

    ap.add_argument('--model', type=str, default='EEGFormer',
                    help='type of hyperbolic convolution to use')
    ap.add_argument('--model_path', type=str, default='./checkpoint/BCIcha/',
                    help='the folder path for saving the model')
    ap.add_argument('--data_path', type=str, default=None, help='data path')
    ap.add_argument('--dataset', type=str, default='mamem',
                    help='dataset name (bci - this is the iv 2a one, mamem, bcicha)')

    ap.add_argument('--conv_type', type=str, default='original', help='type of hyperbolic convolution to use')
    ap.add_argument('--batch_type', type=str, default=None, help='type of hyperbolic batchnorm to use')
    ap.add_argument('--pool_type', type=str, default='dirty', help='type of hyperbolic batchnorm to use')
    ap.add_argument('--learnable_k', type=int, default=1, help='')
    ap.add_argument('--clip_grad', type=float, default=0, help='gradient clipping')

    ap.add_argument('--bs', type=int, default=32, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lora_lr', type=float, default=1e-1, help='learning rate')
    ap.add_argument('--wd', type=float, default=1e-2, help='weight decay')
    ap.add_argument('--dropout', type=float, default=0, help='dropout')
    ap.add_argument('--windows', type=int, default=3, help='number of windows')
    ap.add_argument('--seed', type=int, default=1100, help='')

    ap.add_argument('--finetune', type=str2bool, default=False)
    ap.add_argument('--debug', type=str2bool, default=True)

    ap.add_argument('--pre_processor', type=str2bool, default=True, help='')
    ap.add_argument('--pre_encoder', type=str2bool, default=True, help='')
    ap.add_argument('--cutfill', type=str2bool, default=False)
    ap.add_argument('--learn_predecoder', type=str2bool, default=False)
    ap.add_argument('--learn_decoder', type=str2bool, default=True)
    ap.add_argument('--learn_lora', type=str2bool, default=True)
    ap.add_argument('--tag', type=str, default="euclidcontrast", help='additional identifier for experiments')
    ap.add_argument('--type', type=str, default="default", help='')
    ap.add_argument('--interaug', type=str2bool, default=True)
    ap.add_argument('--scheduler', type=str2bool, default=True)
    ap.add_argument('--lora_type', type=str, default='boost', choices=['linear', 'boost'],
                    help='Type of LoRA adaptation: linear (Euclidean low-rank) or boost (Lorentz boost)')


    args = vars(ap.parse_args())



    resolve_k = args["pre_processor"] or args["pre_encoder"]

    print(f"Using Device: {args['device']}")
    if args["debug"]:
        print('Seed is being fixed.')
        fix_seed(args["seed"])

###############################################################  Data Loading

    trainloader, validloader, testloader, in_channels, num_pred_classes, num_subjects = get_dataloaders(
        dataset=args['dataset'],
        subject=args['sub'],
        batch_size=args['bs'],
        finetune=args['finetune'],
        interaug=args['interaug'],
    )

###############################################################  Model Loading

    print(f'subject {args["sub"]}')
    for repeat in range(1, args['repeat']+1):
        print(f'+++++++++++ Repeat: {repeat} +++++++++++ ')

        args["num_class"] = num_pred_classes
        args["enc_in"] = in_channels

        learn_k = True if args["learnable_k"] != 0 else False

        manifold = None
        checkpoint = None

        if args['dataset'] == 'mamem':
            timesteps = 128
        elif args['dataset'] == 'bci':
            timesteps = 438
        elif args['dataset'] == 'bcicha':
            timesteps = 160

        if args['model'] == 'TCF':
            # from models.tcformer import TCFormer
            # net = TCFormer(n_channels=in_channels, n_classes=num_pred_classes)
            config["n_classes"] = num_pred_classes
            config["n_channels"] = in_channels
            net = create_model("tcformer", config)
        elif args['model'] == 'EEGFormer':
            net = Conformer(num_electrodes=in_channels,
                      sampling_rate=timesteps,
                      hid_channels=40,
                      depth=6,
                      heads=10,
                      dropout=0.5,
                      forward_expansion=4,
                      forward_dropout=0.5,
                      num_classes=num_pred_classes)
        elif args['model'] == 'ATCNet':
            net = ATCNet(in_channels=1,
                          num_classes=num_pred_classes,
                          num_windows=1,
                          num_electrodes=in_channels,
                          chunk_size=timesteps)

        net.to(args['device'])
        if args['model'] == 'BaselineDeviationModelIdEmbedHeadLora':
            hyperoptim = True
        else:
            hyperoptim = False

        try:
            trainNetwork(net,
                         trainloader,
                         validloader,
                         testloader,
                         hyperbolic=hyperoptim,
                         num_classes=num_pred_classes,
                         **args
                         )

        except KeyboardInterrupt:
            print("\nInterrupt detected! Cleaning up GPU memory...")
            # This helps clear the GPU context so the IDE can detach
            del net
            torch.cuda.empty_cache()
            sys.exit(0)