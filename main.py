import sys
from utils.pytorch_module import create_model
sys.path.append("../")
import argparse
import torch
from results.analyse import FindBestConfig
from trainer import trainNetwork, testNetwork, testNetwork_auc
from trainer_multiple import trainNetworkMultiple
from utils.utils import (fix_seed, get_dataloaders, load_block_from_checkpoint, get_model_cross_subject,
                         get_model_single_subject, custom_initialize, load_best_checkpoint, get_model_checkpoint_name,
                         load_full_block_from_checkpoint, load_half_block_from_checkpoint)
from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz
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
    ap.add_argument('--iterations', type=int, default=300, help='number of training iterations')


    ap.add_argument('--model', type=str, default='BaselineDeviationModelIdEmbedHeadLora',
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

    ap.add_argument('--bs', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    ap.add_argument('--lora_lr', type=float, default=1e-5, help='learning rate')
    ap.add_argument('--wd', type=float, default=1e-2, help='weight decay')
    ap.add_argument('--dropout', type=float, default=0, help='dropout')
    ap.add_argument('--windows', type=int, default=1, help='number of windows')
    ap.add_argument('--seed', type=int, default=1100, help='')

    ap.add_argument('--finetune', type=str2bool, default=True)#######################WHEN FINETUNE???
    ap.add_argument('--debug', type=str2bool, default=False)

    ap.add_argument('--pre_processor', type=str2bool, default=False, help='')
    ap.add_argument('--pre_encoder', type=str2bool, default=False, help='')
    ap.add_argument('--cutfill', type=str2bool, default=True)
    ap.add_argument('--learn_predecoder', type=str2bool, default=False)
    ap.add_argument('--learn_decoder', type=str2bool, default=False)
    ap.add_argument('--learn_lora', type=str2bool, default=True)
    ap.add_argument('--tag', type=str, default="1313", help='additional identifier for experiments')
    ap.add_argument('--type', type=str, default="half", help='')
    ap.add_argument('--interaug', type=str2bool, default=False)
    ap.add_argument('--scheduler', type=str2bool, default=False)
    ap.add_argument('--lora_type', type=str, default='linear', choices=['linear', 'boost'],
                    help='Type of LoRA adaptation: linear (Euclidean low-rank) or boost (Lorentz boost)')

    
    ap.add_argument('--run_id', type=int, default=999, help='unique run identifier')
    ap.add_argument('--dist', type=str, default='AE', help='distance metric for pooling')
    ap.add_argument('--filter_len', type=int, default=40, help='base filter length for InceptionTime model')
    ap.add_argument('--es', type=str2bool, default=False, help='early stopping')
    ap.add_argument('--save_results', type=str2bool, default=True, help='')
    ap.add_argument('--verbose', type=str2bool, default=False, help='')
    



    args = vars(ap.parse_args())

    if args['sub'] != 'all':
        args['sub_str'] = args['sub']
        sub = args['sub'].split('_')
        args['sub'] = [int(i) for i in sub] 

    resolve_k = args["pre_processor"] or args["pre_encoder"]

    # print(f"Using Device: {args['device']}")
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

    args["num_class"] = num_pred_classes
    args["enc_in"] = in_channels

    learn_k = True if args["learnable_k"] != 0 else False

    manifold = None
    checkpoint = None

    if args['model'] == 'TCF':
        pass
        # # from models.tcformer import TCFormer
        # # net = TCFormer(n_channels=in_channels, n_classes=num_pred_classes)
        # config["n_classes"] = num_pred_classes
        # config["n_channels"] = in_channels
        # net = create_model("tcformer", config)
        # net = net.to(args['device'])
    elif args['model'] == 'inception':
        from models.inception import InceptionTime
        net = InceptionTime(n_inputs = in_channels, 
                            n_classes = num_pred_classes,
                            n_convolutions = 3,
                            n_filters = 32,
                            kernel_size = args['filter_len'])
        net = net.to(args['device'])
    else:

        # if resolve_k:
        #     checkpoint_path = get_model_checkpoint_name(**args)
        #     checkpoint = torch.load(checkpoint_path, map_location="cpu")
        #     manifold = CustomLorentz(k=checkpoint["manifold.k"].detach().clone())

        # if args["sub"] != 'all' and not args["finetune"]:
        #     net = get_model_single_subject(num_pred_classes, **args)
        # else:
        net, checkpoint_path = get_model_cross_subject(num_pred_classes, manifold, **args)

        # if args['finetune'] and args['sub'] != 'all':
        #     analyser = FindBestConfig(args['dataset'], subject="all")
        #     best_config, _ = analyser.get_best_config(args['model'], f'win{args["windows"]}')
        #     dataset = args['dataset']
        #     model = args['model']

        #     checkpoint_path = f'./checkpoints/{dataset}_{model}_{best_config}_all.pt'
        #     print('Loading Checkpoint from: ', checkpoint_path)
        #     net = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        #     net = net.to(args['device'])

        #     args["pre_processor"] = False
        #     args["pre_encoder"] = False

        # if args["pre_processor"]:
        #     load_block_from_checkpoint(net, checkpoint,'processor.')
        # if args["pre_encoder"]:
        #     match args["type"]:
        #         case "half":
        #             load_half_block_from_checkpoint(net, checkpoint, 'encoder.')
        #         case "full":
        #             load_full_block_from_checkpoint(net, checkpoint,'encoder.')
        #         case "default":
        #             load_block_from_checkpoint(net, checkpoint,'encoder.')

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
            #net.pre_decoder.Q.requires_grad = False
            #net.pre_decoder.R.requires_grad = False

    if (args['model'] == 'TCF') or (args['model'] == 'inception'):
        hyperoptim = False
    else:
        hyperoptim = True

    try:
        # if args["sub"] != 'all' and not args["finetune"]:
        # if len(args['sub']) == 1:
        #     trainNetwork(net,
        #                  trainloader,
        #                  validloader,
        #                  testloader,
        #                  hyperbolic=hyperoptim,
        #                  num_classes=num_pred_classes,
        #                  **args
        #                  )
        # else:
        trainNetworkMultiple(net,
                        trainloader,
                        validloader,
                        testloader,
                        hyperbolic=hyperoptim,
                        num_classes=num_pred_classes,
                        **args
                        )
        print('End Training -------------------------------\n\n\n')
    except KeyboardInterrupt:
        print("\nInterrupt detected! Cleaning up GPU memory...")
        # This helps clear the GPU context so the IDE can detach
        del net
        torch.cuda.empty_cache()
        sys.exit(0)