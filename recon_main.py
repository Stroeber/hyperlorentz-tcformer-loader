import os
import argparse

import yaml
import torch
from torch.optim.lr_scheduler import LambdaLR

from hyperbolic_lib.lib.geoopt.optim import RiemannianAdam, RiemannianAdamW, RiemannianSGD
from reconstruction.losses import CutFillConfig
from reconstruction.model import TimeSeriesPretrainer, SimpleDecoder
from reconstruction.train_class import TrainConfig, train_epoch, evaluate
from utils.utils import get_param_groups, get_model_recon, get_dataloaders
from utils.lr_scheduler import linear_warmup_cosine_decay
from utils.get_datamodule_cls import get_datamodule_cls
from utils.pytorch_data import DataLoaderManager

def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


# ---- glue it all together ----
def run_training(args):

    device = args["device"]

    ###############################################################  Data Loading

    trainloader, validloader, testloader, in_channels, num_pred_classes, num_subjects = get_dataloaders(
        dataset=args['dataset'],
        subject=args['sub'],
        batch_size=args['bs'],
        finetune=args['finetune'],
    )
    if args['dataset'] == "bci": 

        config_path = os.path.join("configs", "tcformer.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        config["preprocessing"] = config["preprocessing"]["bcic2a"]
        config["preprocessing"]["z_scale"] = config["z_scale"]

        # Get datamodule class
        datamodule_cls = get_datamodule_cls("bcic2a")
        in_channels = datamodule_cls.channels
        
        # Create data loaders
        data_manager = DataLoaderManager(datamodule_cls, config["preprocessing"], "all", include_subject_ids=True)
        data_manager.setup()
        trainloader = data_manager.get_train_loader()
        validloader = data_manager.get_val_loader()
        testloader = data_manager.get_test_loader()


###############################################################  Model Loading
    

    s_embed_type = None if args["sub"] != 'all' and not args["finetune"] else args["subject_embed_type"]
    # Model
    model = get_model_recon(**args)

    for row in trainloader:

        x_orig, x_sub, yb = row
        # x_orig, yb = row

        x_orig = x_orig.to(args["device"])
        x_sub = x_sub.to(args["device"])
    #
        model.get_decoder(x_orig, x_sub)
    #
        break
    model.to(args["device"])

    # Optimizer
    param_groups = get_param_groups(model, args["lr"], args["wd"])
    # opt = RiemannianSGD(param_groups, lr=args["lr"], weight_decay=args["wd"])
    opt = RiemannianAdam(param_groups, lr=args["lr"], weight_decay=args["wd"])

    scheduler = None
    if args['scheduler']:
        # Warmup 20 epochs, total 300 epochs (now per-epoch stepping)
        scheduler = LambdaLR(opt, linear_warmup_cosine_decay(warmup_steps=20, total_steps=300))

    # Training config
    train_cfg = TrainConfig(
        task_mix_recon=0.5,  # 50% reconstruction, 50% cut-fill         #\s changed from 1
        cutfill_cfg=CutFillConfig(min_frac=0.1, max_frac=0.30, fill_value=0.0, loss_on_full=False),
        # grad_clip=1.0,
        contrastive_weight=0.1,  # Weight for InfoNCE contrastive loss
        n_mask_spans=2,  # Number of spans for multi-span masking
        channel_drop_prob=0.1  # Probability of dropping each channel
    )

    # Loop
    best_val = float("inf")
    best_test = float("inf")
    for epoch in range(args['iterations']):
        stats = train_epoch(model, itr_merge(trainloader, validloader, testloader), opt, device, train_cfg, s_embed_type)
        
        # Scheduler step per-epoch (not per-batch)
        if scheduler is not None:
            scheduler.step()
        
        val_stats = evaluate(model, validloader, device, cutfill_eval_cfg=CutFillConfig(), subject=s_embed_type)
        test_stats = evaluate(model, testloader, device, cutfill_eval_cfg=CutFillConfig(), subject=s_embed_type)
        print(f"Epoch {epoch:02d} | train loss {stats['loss']:.4f} "
              f"(recon {stats['recon_loss']:.4f}, cutfill {stats['cutfill_loss']:.4f}, "
              f"contrast {stats['contrastive_loss']:.4f}) | "
              f"val recon {val_stats['val_recon_mse']:.4f}, val cutfill {val_stats['val_cutfill_mse']:.4f}")

        # Simple checkpointing on cut-fill metric
        metric = val_stats["val_cutfill_mse"]
        if metric < best_val:
            best_val = metric
            torch.save(model.state_dict(), f"./reconstruction/checkpoints/{args['dataset']}_{args['model']}_w{args['slice_window']}_pretrained_encoder_decoder_val_{args['tag']}.pt")

        metric = test_stats["val_cutfill_mse"]
        if metric < best_test:
            best_test = metric
            torch.save(model.state_dict(), f"./reconstruction/checkpoints/{args['dataset']}_{args['model']}_w{args['slice_window']}_pretrained_encoder_decoder_{args['tag']}.pt")


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument('--device', type=str, default='cuda:0', help='Torch Device for computations')
    ap.add_argument('--repeat', type=int, default=1, help='No.xxx repeat for training model')
    ap.add_argument('--sub', type=str, default="all", help='subjectxx you want to train')
    ap.add_argument('--iterations', type=int, default=300, help='number of training iterations')

    ap.add_argument('--model', type=str, default='BaselineBlockModelIdEmbedLora', help='type of hyperbolic convolution to use')
    ap.add_argument('--model_path', type=str, default='./checkpoint/BCIcha/', help='the folder path for saving the model')
    ap.add_argument('--data_path', type=str, default=None, help='data path')
    ap.add_argument('--dataset', type=str, default='bci', help='dataset name')

    ap.add_argument('--conv_type', type=str, default='original', help='type of hyperbolic convolution to use')
    ap.add_argument('--batch_type', type=str, default=None, help='type of hyperbolic batchnorm to use')
    ap.add_argument('--pool_type', type=str, default='dirty', help='type of hyperbolic batchnorm to use')
    ap.add_argument('--learnable_k', type=int, default=1, help='')
    ap.add_argument('--clip_grad', type=float, default=0, help='gradient clipping')

    ap.add_argument('--bs', type=int, default=128, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    ap.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    ap.add_argument('--dropout', type=float, default=0, help='dropout')
    ap.add_argument('--seed', type=int, default=100, help='')
    ap.add_argument('--finetune', type=bool, default=False)
    ap.add_argument('--finetune_recon', type=bool, default=False)

    ap.add_argument('--scheduler', type=bool, default=False)
    ap.add_argument('--lora_type', type=str, default='boost', choices=['linear', 'boost'],
                    help='Type of LoRA adaptation: linear (Euclidean low-rank) or boost (Lorentz boost)')

    ap.add_argument("--slice_type", type=str, choices=["absolute", "padded", "None"], default="absolute",
                    help="Slicing mode")
    ap.add_argument("--slice_stride", type=int, default=56, help="stride for the padded slicing method")
    ap.add_argument("--slice_window", type=int, default=3,
                    help="number of windows for the absolute slice and window_size for the padded slice")

    ap.add_argument("--subject_embed_dim", type=int, default=3, help="Embedding/conditioning dim.")
    ap.add_argument("--subject_embed_type", choices=["simple", "film", None], default="simple",
                    help="how to embed the subject id")
    ap.add_argument("--subject_embed_loc", choices=["pre", "post", None], default="pre",
                    help="whether to embed the subject id before or after the denoise")
    ap.add_argument('--tag', type=str, default="euclidcontrast", help='additional identifier for experiments')

    args = vars(ap.parse_args())

    run_training(args)

