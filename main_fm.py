import sys

sys.path.append("../")

import argparse

from utils.losses import PALoss_Angle, LorentzHIERLoss

from models.models import BaselineDeviationModelIdLoraCbramod
from models.cbramod import CBraMod,CbraModHead, CbramodInc, InceptionBlock, InceptionModel
from trainer import trainNetwork, testNetwork, testNetwork_auc
from trainer_multiple import trainNetworkMultiple
from utils.utils import count_params
from utils.bciDataLoaderFull import BCIDataLoaderFull
from utils.bciDataLoaderFM import BCIDataLoaderFM
from utils.mamemDataLoaderFM import MAMEMDataLoaderFM
from utils.mamemDataloaderFull import MAMEMDataLoaderFull
from utils.bcichaDataLoaderFM import BCIchaDataLoaderFM
from utils.bcichaDataLoaderFull import BCIchaDataLoaderFull
import torch.nn as nn

import torch
import random
import numpy as np

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
    ap.add_argument('--device', type=str, default='cuda', help='Torch Device for computations')
    ap.add_argument('--repeat', type=int, default=1, help='No.xxx repeat for training model')
    ap.add_argument('--sub', type=str, default='all', help='subjectxx you want to train')
    ap.add_argument('--iterations', type=int, default=2, help='number of training iterations')

    ap.add_argument('--model', type=str, default='Latte_Cbramod', help='type of hyperbolic convolution to use')
    ap.add_argument('--model_path', type=str, default='./checkpoint/BCIcha/',
                    help='the folder path for saving the model')
    ap.add_argument('--data_path', type=str, default='data/MAMEM/', help='data path')
    ap.add_argument('--dataset', type=str, default='bcicha', help='dataset name')

    ap.add_argument('--conv_type', type=str, default='original', help='type of hyperbolic convolution to use')
    ap.add_argument('--batch_type', type=str, default=None, help='type of hyperbolic batchnorm to use')
    ap.add_argument('--pool_type', type=str, default='dirty', help='type of hyperbolic batchnorm to use')
    ap.add_argument('--learnable_k', type=int, default=0, help='')
    ap.add_argument('--clip_grad', type=float, default=0, help='gradient clipping')

    ap.add_argument('--bs', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    ap.add_argument('--lora_lr', type=float, default=1e-1, help='learning rate')
    ap.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    ap.add_argument('--dropout', type=float, default=0, help='dropout')
    ap.add_argument('--windows', type=int, default=1, help='number of windows')
    ap.add_argument('--seed', type=int, default=1100, help='')

    ap.add_argument('--finetune', type=str2bool, default=False)
    ap.add_argument('--debug', type=str2bool, default=False)

    ap.add_argument('--pre_processor', type=str2bool, default=True, help='')
    ap.add_argument('--pre_encoder', type=str2bool, default=True, help='')
    ap.add_argument('--cutfill', type=str2bool, default=False)
    ap.add_argument('--learn_predecoder', type=str2bool, default=False)
    ap.add_argument('--learn_decoder', type=str2bool, default=True)
    ap.add_argument('--learn_lora', type=str2bool, default=True)
    ap.add_argument('--tag', type=str, default="1313", help='additional identifier for experiments')
    ap.add_argument('--type', type=str, default="half", help='')
    ap.add_argument('--interaug', type=str2bool, default=False)
    ap.add_argument('--scheduler', type=str2bool, default=True)
    ap.add_argument('--lora_type', type=str, default='linear', choices=['linear', 'boost'],
                    help='Type of LoRA adaptation: linear (Euclidean low-rank) or boost (Lorentz boost)')

    # Introduced argumented for foundational models
    ap.add_argument('--foundation_dir', type=str, default="pretrained_weights/pretrained_weights.pth")
    ap.add_argument('--use_pretrained_weights', type=bool,
                    default=True, help='use_pretrained_weights')
    ap.add_argument('--classifier', type=str, default='all_patch_reps',
                    help='[all_patch_reps, all_patch_reps_twolayer, '
                         'all_patch_reps_onelayer, avgpooling_patch_reps]')
    ap.add_argument('--cuda', type=int, default=1, help='cuda number (default: 1)')
    ap.add_argument('--use_linear', type=int, default=1, help='If 1, a linear layer is used as a decoder')

    args = vars(ap.parse_args())

    seed = int(args["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Using Device: {args['device']}")
    """
    Datasets:
    BCI
        Input: 22  Cls: 4
    Mamem
        Input: 8 Cls: 5
    BCICHA 
        Input: 56  Cls: 2
    """

    if args['dataset'] == 'bci':
        in_channels = 22
        num_pred_classes = 4
        num_subjects = 9
        args['data_path'] = './data/BCICIV_2a_mat/'

        if args['sub'] == 'all' and not args["finetune"]:
            dataloader = BCIDataLoaderFull(
                subject=args['sub'],
                data_path=args['data_path'],
                bs=args['bs'],
            )
        else:
            dataloader = BCIDataLoaderFM(
                subject=args['sub'],
                ratio=8,
                data_path=args['data_path'],
                bs=args['bs'],
                model=None,
                finetune=args['finetune'],
                patch_length=200
            )

        trainloader, validloader, testloader = dataloader.get_dataloader()


    elif args['dataset'].startswith('mamem'):
        in_channels = 8
        num_pred_classes = 5
        num_subjects = 11
        args['data_path'] = './data/MAMEM/'

        if args['sub'] == 'all' and not args["finetune"]:
            dataloader = MAMEMDataLoaderFull(
                subject=args['sub'],
                data_path=args['data_path'],
                bs=args['bs'],
            )
        else:
            dataloader = MAMEMDataLoaderFM(
                subject=args['sub'],
                ratio=8,
                data_path=args['data_path'],
                bs=args['bs'],
                model=args['model'],
                finetune=args['finetune'],
                patch_length=200
            )

        trainloader, validloader, testloader = dataloader.get_dataloader()


    elif args['dataset'].startswith('bcicha'):
        in_channels = 56
        num_pred_classes = 2
        num_subjects = 16
        args['data_path'] = './data/BCIcha/'

        if args['sub'] == 'all' and not args["finetune"]:
            dataloader = BCIchaDataLoaderFull(
                subject=args['sub'],
                data_path=args['data_path'],
                bs=args['bs'],
            )
        else:
            dataloader = BCIchaDataLoaderFM(
                subject=args['sub'],
                data_path=args['data_path'],
                bs=args['bs'],
                model=None,
                finetune=args['finetune'],
                patch_length=200
            )

        trainloader, validloader, testloader = dataloader.get_dataloader()


    else:
        raise ValueError('No such dataset')

    print(f'subject {args["sub"]}')

    for repeat in range(1, args['repeat'] + 1):
        print(f'+++++++++++ Repeat: {repeat} +++++++++++ ')

        args["num_class"] = num_pred_classes
        args["enc_in"] = in_channels

        if args['model'] == 'Cbramod':
            net = CbraModHead(args)
            net = net.to(args['device'])

        elif args['model'] == 'CbramodIncShallow':
            model = CBraMod()
            model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=args['device']))
            model.proj_out = nn.Identity()
            decoder = InceptionModel(in_channels=in_channels, num_pred_classes=num_pred_classes, multilayer_inception=False)
            net = CbramodInc(model, decoder, freeze_encoder=False).to(args['device'])

        elif args['model'] == 'CbramodIncShallow_frozen':
            model = CBraMod()
            model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=args['device']))
            model.proj_out = nn.Identity()
            decoder = InceptionModel(in_channels=in_channels, num_pred_classes=num_pred_classes, multilayer_inception=False)
            net = CbramodInc(model, decoder, freeze_encoder=True).to(args['device'])

        elif args['model'] == 'CbramodIncDeep':
            model = CBraMod()
            model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=args['device']))
            model.proj_out = nn.Identity()
            decoder = InceptionModel(in_channels=in_channels, num_pred_classes=num_pred_classes, multilayer_inception=True)
            net = CbramodInc(model, decoder, freeze_encoder=False).to(args['device'])

        elif args['model'] == 'CbramodIncDeep_frozen':
            model = CBraMod()
            model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=args['device']))
            model.proj_out = nn.Identity()
            decoder = InceptionModel(in_channels=in_channels, num_pred_classes=num_pred_classes, multilayer_inception=True)
            net = CbramodInc(model, decoder, freeze_encoder=True).to(args['device'])

        elif args['model'] == 'Latte_Cbramod':
            net = BaselineDeviationModelIdLoraCbramod(n_classes=num_pred_classes,
                                            features=450/4,
                                            dataset=args["dataset"],
                                            pool_type=args["pool_type"],
                                            conv_type=args["conv_type"],
                                            batch_type=args["batch_type"],
                                            dropout=args["dropout"],
                                            windows=args["windows"],
                                            recon=False,
                                            lora_lr=1e-4,
                                            proc=args["pre_processor"],
                                            freeze_FM=False,
                                            ).to(args['device'])

        elif args['model'] == 'Latte_Cbramod_frozen':
            net = BaselineDeviationModelIdLoraCbramod(n_classes=num_pred_classes,
                                            features=450/4,
                                            dataset=args["dataset"],
                                            pool_type=args["pool_type"],
                                            conv_type=args["conv_type"],
                                            batch_type=args["batch_type"],
                                            dropout=args["dropout"],
                                            windows=args["windows"],
                                            recon=False,
                                            lora_lr=1e-4,
                                            proc=args["pre_processor"],
                                            freeze_FM=True,
                                            ).to(args['device'])



        if 'data_path' in args:
            args.pop('data_path')

        try:
            if args["sub"] != 'all' and not args["finetune"]:
                trainNetwork(net,
                             trainloader,
                             validloader,
                             testloader,
                             hyperbolic=False,
                             num_classes=num_pred_classes,
                             **args
                             )
            else:
                trainNetworkMultiple(net,
                                     trainloader,
                                     validloader,
                                     testloader,
                                     hyperbolic=False,
                                     num_classes=num_pred_classes,
                                     **args
                                     )
        except KeyboardInterrupt:
            print("\nInterrupt detected! Cleaning up GPU memory...")
            # This helps clear the GPU context so the IDE can detach
            del net
            torch.cuda.empty_cache()
            sys.exit(0)