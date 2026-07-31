import sys

sys.path.append("../")
import argparse
import torch
from trainer_multiple import trainNetworkMultiple
from utils.utils import (fix_seed, get_dataloaders, load_block_from_checkpoint, get_model_cross_subject,
                         get_model_single_subject, custom_initialize, load_best_checkpoint, get_model_checkpoint_name,
                         load_full_block_from_checkpoint, load_half_block_from_checkpoint)
from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz
from models.cbramod import CBraMod,CbraModHead


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
    ap.add_argument('--sub', type=str, default='all', help='subjectxx you want to train')
    ap.add_argument('--iterations', type=int, default=3, help='number of training iterations')

    ap.add_argument('--model', type=str, default='Cbramod',
                    help='type of hyperbolic convolution to use')
    ap.add_argument('--model_path', type=str, default='./checkpoint/BCIcha/',
                    help='the folder path for saving the model')
    ap.add_argument('--data_path', type=str, default=None, help='data path')
    ap.add_argument('--dataset', type=str, default='mamem',
                    help='dataset name (bci - this is the iv 2a one, mamem, bcicha)')

    ap.add_argument('--conv_type', type=str, default='original', help='type of hyperbolic convolution to use')
    ap.add_argument('--batch_type', type=str, default=None, help='type of hyperbolic batchnorm to use')
    ap.add_argument('--pool_type', type=str, default='dirty', help='type of hyperbolic batchnorm to use')
    ap.add_argument('--learnable_k', type=int, default=0, help='')
    ap.add_argument('--clip_grad', type=float, default=0, help='gradient clipping')

    ap.add_argument('--bs', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    ap.add_argument('--lora_lr', type=float, default=1e-5, help='learning rate')
    ap.add_argument('--wd', type=float, default=1e-3, help='weight decay')
    ap.add_argument('--dropout', type=float, default=0, help='dropout')
    ap.add_argument('--windows', type=int, default=5, help='number of windows')
    ap.add_argument('--seed', type=int, default=100, help='')

    ap.add_argument('--finetune', type=str2bool, default=False)
    ap.add_argument('--debug', type=str2bool, default=True)

    ap.add_argument('--pre_processor', type=str2bool, default=False, help='')
    ap.add_argument('--pre_encoder', type=str2bool, default=False, help='')
    ap.add_argument('--cutfill', type=str2bool, default=False)
    ap.add_argument('--learn_predecoder', type=str2bool, default=False)
    ap.add_argument('--learn_decoder', type=str2bool, default=False)
    ap.add_argument('--learn_lora', type=str2bool, default=True)
    ap.add_argument('--tag', type=str, default="tcf", help='additional identifier for experiments')
    ap.add_argument('--type', type=str, default="default", help='')
    ap.add_argument('--interaug', type=str2bool, default=False)
    ap.add_argument('--scheduler', type=str2bool, default=True)
    ap.add_argument('--lora_type', type=str, default='linear', choices=['linear', 'boost'],
                    help='Type of LoRA adaptation: linear (Euclidean low-rank) or boost (Lorentz boost)')

    ap.add_argument('--foundation_dir', type=str, default="pretrained_weights/pretrained_weights.pth")
    ap.add_argument('--use_pretrained_weights', type=bool,
                    default=True, help='use_pretrained_weights')
    ap.add_argument('--classifier', type=str, default='all_patch_reps',
                    help='[all_patch_reps, all_patch_reps_twolayer, '
                         'all_patch_reps_onelayer, avgpooling_patch_reps]')
    ap.add_argument('--num_of_classes', type=int, default=None, help='number of classes')
    args = vars(ap.parse_args())

    resolve_k = args["pre_processor"] or args["pre_encoder"]

    print(f"Using Device: {args['device']}")
    if args["debug"]:
        print('Seed is being fixed.')
        fix_seed(args["seed"])

    subjects_lists = {
        'bci': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'mamem': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'bcicha': [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
    }
    args['loso'] = True

    from utils.bciDataloaderLOSO import BCIDataLoaderLOSO
    from utils.mamemDataloaderLOSO import MAMEMDataLoaderLOSO
    from utils.bcichaDataLoaderLOSO import BCIchaDataLoaderLOSO

    if args['model'] == 'Cbramod':
        patch_length = 200
    else:
        patch_length = None

    for cv_subject in subjects_lists[args['dataset']]:
        args['sub'] = cv_subject

        if args['dataset'] == 'bci':
            loader = BCIDataLoaderLOSO(cv_subject=cv_subject, bs=args['bs'],patch_length=patch_length)
            trainloader, validloader, testloader = loader.get_dataloader()
            in_channels = 22
            num_pred_classes = 4
            num_subjects = 9
        elif args['dataset'] == 'mamem':
            loader = MAMEMDataLoaderLOSO(cv_subject=cv_subject, bs=args['bs'],patch_length=patch_length)
            trainloader, validloader, testloader = loader.get_dataloader()
            in_channels = 8
            num_pred_classes = 5
            num_subjects = 11
        elif args['dataset'] == 'bcicha':
            loader = BCIchaDataLoaderLOSO(cv_subject=cv_subject, bs=args['bs'],patch_length=patch_length)
            trainloader, validloader, testloader = loader.get_dataloader()
            in_channels = 56
            num_pred_classes = 2
            num_subjects = 16


        ###############################################################  Model Loading

        print(f'Subject {cv_subject} is being used for testing')
        for repeat in range(1, args['repeat'] + 1):
            print(f'+++++++++++ Repeat: {repeat} +++++++++++ ')

            learn_k = True if args["learnable_k"] != 0 else False

            manifold = None
            checkpoint = None

            if args['model'] == 'TCF':
                from models.tcformer import TCFormer

                net = TCFormer(n_channels=in_channels, n_classes=num_pred_classes)
                net = net.to(args['device'])
            elif args['model'] == 'Cbramod':
                args["num_class"] = num_pred_classes
                args["enc_in"] = in_channels

                net = CbraModHead(args)
                # Load the checkpoint
                checkpoint = torch.load("pretrained_weights/pretrained_weights.pth", weights_only=False,
                                        map_location=torch.device('cpu'))
                # Get the state dictionary from the checkpoint
                checkpoint_state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint
                # Load the model's state dictionary
                model_state_dict = net.state_dict()
                # Filter out mismatched layers
                filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if
                                       k in model_state_dict and model_state_dict[k].size() == v.size()}
                # Update the model's state dictionary
                model_state_dict.update(filtered_state_dict)
                # Load the updated state dictionary into the model
                net.load_state_dict(model_state_dict)
                net = net.to(args['device'])
            else:

                if resolve_k:
                    checkpoint_path = get_model_checkpoint_name(**args)
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    manifold = CustomLorentz(k=checkpoint["manifold.k"].detach().clone())

                net, checkpoint_path = get_model_cross_subject(num_pred_classes, manifold, **args)

                if args["pre_processor"]:
                    load_block_from_checkpoint(net, checkpoint, 'processor.')
                if args["pre_encoder"]:
                    match args["type"]:
                        case "half":
                            load_half_block_from_checkpoint(net, checkpoint, 'encoder.')
                        case "full":
                            load_full_block_from_checkpoint(net, checkpoint, 'encoder.')
                        case "default":
                            load_block_from_checkpoint(net, checkpoint, 'encoder.')

                for row in trainloader:
                    x_orig, x_sub, yb = row

                    x_orig = x_orig.to(args["device"])
                    x_sub = x_sub.to(args["device"])

                    net.get_decoder(x_orig, x_sub)
                    break

                print('Curvature: ', net.manifold.k)

                if not args["learn_decoder"]:
                    print("freezing decoder")

                    if args['model'] != 'BaselineDeviationModelIdEmbedHeadLora':
                        for name, param in net.pre_decoder.named_parameters():
                            if "manifold.k" not in name:
                                param.requires_grad = False

                    for name, param in net.decoder.named_parameters():
                        if "manifold.k" not in name:
                            param.requires_grad = False

                if args['learn_predecoder']:
                    net.pre_decoder.weight.weight.requires_grad = True
                else:
                    net.pre_decoder.weight.weight.requires_grad = False

                if not args['learn_lora']:
                    print('Freezing Lora Weights')
                    net.processor.conv1.Q.requires_grad = False
                    net.processor.conv1.R.requires_grad = False
                    net.processor.conv2.Q.requires_grad = False
                    net.processor.conv2.R.requires_grad = False
                    # net.pre_decoder.Q.requires_grad = False
                    # net.pre_decoder.R.requires_grad = False

            if args['model'] == 'TCF':
                hyperoptim = False
            else:
                hyperoptim = True

            try:
                trainNetworkMultiple(net,
                                     trainloader,
                                     validloader,
                                     testloader,
                                     hyperbolic=hyperoptim,
                                     # loss_1=loss_1,
                                     # loss_2=loss_2,
                                     num_classes=num_pred_classes,
                                     **args
                                     )
            except KeyboardInterrupt:
                print("\nInterrupt detected! Cleaning up GPU memory...")
                # This helps clear the GPU context so the IDE can detach
                del net
                torch.cuda.empty_cache()
                sys.exit(0)