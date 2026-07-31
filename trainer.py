import sys

sys.path.append("..")
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, StepLR, LambdaLR
import torcheeg.transforms as transforms
from torcheeg.transforms import Compose, ToTensor, Resize, CWTSpectrum, BandSignal

from sklearn.metrics import roc_auc_score as ras
import numpy as np

from focal import FocalLoss
from utils.save_results import save_output
from utils.utils import (get_loss,
                         get_dataset_statistics,
                         get_param_groups,
                         get_dataset_max_min,
                         get_transforms,
                         get_params_groups,
                         clip_gradients_value)

from utils.lr_scheduler import linear_warmup_cosine_decay
from hyperbolic_lib.lib.geoopt.optim import RiemannianSGD, RiemannianAdam, RiemannianAdamW

APPLY_MEAN = False
APPLY_MAX = False


def trainNetwork(net, trainloader, validloader, testloader, model_path=None, model=None, dataset=None, bs=64,
                 iterations=500, lr=5 * 1e-4, wd=None, repeat=None, sub=None, epochs=None, subject_weights=None,
                 verbose=True, save_model=False, save_results=True, early_stopping=False, grace_period=20,
                 hyperbolic=False, clip_grad=0, loss_1=None, loss_2=None, num_classes=0, scheduler=False, **kwargs):
    best_test = 0
    best_val = 0
    test_from_best_val = 0
    test_from_best_val_loss = 0
    test_from_best_train_loss = 0
    best_train_loss = 1e10
    update_best_test = 0

    device = kwargs['device']
    sd_log = scheduler
    CE = nn.CrossEntropyLoss(label_smoothing=0.0)
    alpha = [1.0] * num_classes
    # CE = FocalLoss(gamma=2., alpha=alpha, task_type='multi-class', num_classes=num_classes)

    if not hyperbolic:
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd, momentum=0.6, nesterov=False)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=wd
        )
    else:
        print("hyperbolic")
        param_groups = get_params_groups(net, loss_1, loss_2,
                                         weight_decay=wd) if loss_1 is not None else get_param_groups(net, lr, wd)
        # optimizer = RiemannianSGD(get_param_groups(net, lr, wd), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
        optimizer = RiemannianAdam(param_groups, lr=lr, weight_decay=wd)
        # optimizer = RiemannianAdam(param_groups, lr=lr, weight_decay=wd)
        # optimizer = MixOptimizer(optimizer)
    if scheduler:
        scheduler = LambdaLR(optimizer, linear_warmup_cosine_decay(20, 1000))
    else:
        scheduler = None

    bestLoss = 1e10
    bestLoss_early_stopping = 1e10

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_accs = []
    test_losses = []

    no_improvement_count = 0

    dropout = kwargs['dropout']
    windows = kwargs['windows']
    processor = kwargs['pre_processor']
    encoder = kwargs['pre_encoder']
    cut = kwargs['cutfill']
    decoder = kwargs['learn_decoder']
    predecoder = kwargs['learn_predecoder']
    lora = kwargs['learn_lora']
    lora_lr = kwargs['lora_lr']

    for ite in range(iterations):
        net.train()
        acc_val = 0
        acc_tr = 0
        tr_len = 0
        val_len = 0

        train_embeds = []
        train_ys = []

        # for i, param_group in enumerate(optimizer.param_groups):
            #lr = args.lr * (kwargs.bs * utils.get_world_size()) / 180.
        #    param_group["lr"] = lr * param_group["lr_scale"]

############################################# train ####################################################################

        for row in trainloader:
            x_orig, yb = row
            if model == 'ATCNet' or model == 'EEGFormer':
                x_orig = x_orig.unsqueeze(1)
            tr_len += yb.shape[0]

            x_orig = x_orig.to(device)
            yb = yb.to(device)

            #out, embeds= net(x0, x1, return_embeds=True)
            out = net(x_orig)
            # loss = get_loss(CE, net, out, yb, embeds=embeds, loss_1=loss_1, loss_2=loss_2)

            # distances = net.manifold.dist(net.decoder.prototypes, net.decoder.prototypes.unsqueeze(-2))
            # increase_this = distances.mean() / 2
            # decrease_this = net.manifold.dist0(net.decoder.prototypes).mean() / 2
            # add_loss = torch.log(1+decrease_this)/(1+torch.log(increase_this+1))

            # out = nn.Softmax(dim=-1)(out)
            loss = CE(out, yb) #+ 0.1*add_loss
            #train_embeds.append(embeds)
            #train_ys.append(yb)

            optimizer.zero_grad()

            loss.backward(retain_graph=True)

            if clip_grad > 0:
                param_norms = clip_gradients_value(net, clip_grad, losses=[loss])

            optimizer.step()
            acc_tr += (torch.max(out, 1).indices == yb).sum().item()

            torch.cuda.empty_cache()

        # Track best training loss
        if loss.item() < best_train_loss:
            best_train_loss = loss.item()
            update_best_train_loss = True
        else:
            update_best_train_loss = False

        if scheduler is not None:
            scheduler.step()

########################################## validate ####################################################################
        net.eval()
        TL = 0

        for row in validloader:
            x_orig, yb = row

            if model == 'ATCNet' or model == 'EEGFormer':
                x_orig = x_orig.unsqueeze(1)

            x_orig = x_orig.to(device)
            yb = yb.to(device)

            val_len += yb.shape[0]

            with torch.no_grad():
                out = net(x_orig)
                acc_val += (torch.max(out, 1).indices == yb).sum().item()

                TL += CE(out,yb)

########################################## logging and checks ##########################################################
        if early_stopping:
            if TL < bestLoss_early_stopping:
                bestLoss_early_stopping = TL
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= grace_period:
                if verbose:
                    print(f'Early stopping at iteration {ite} as no improvement has been observed in {grace_period} '
                          f'iterations for the validation loss.')
                final_path = os.path.join(model_path, f'repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}.pt')
                torch.save(net, final_path)
                break

        if verbose:
            print('')
            print(f'Iteration{ite}=====')
            print(f'train_loss:{loss:.4f}    val_loss:{TL / len(validloader):.4f}')
            print(f'train_acc:{acc_tr / tr_len:.4f}    val_acc:{acc_val / val_len:.4f}')
            #print(f"k: {net.manifold.k}")

        if acc_val / val_len > best_val:
            best_val = acc_val / val_len
            update_best_test = 1
            # torch.save(torch.cat(train_embeds), "embeds.pt")
            # torch.save(torch.cat(train_ys), "y.pt")
            # torch.save(net.manifold.k, "k.pt")

            if sub == 'all':
                torch.save(net, f"./checkpoints/{dataset}_{model}_win{windows}_bs{bs}_lr{lr}_wd{wd}_dp{dropout}_llr{lora_lr}_proc{processor}_enc{encoder}_cut{cut}_dec{decoder}_predec{predecoder}_lora{lora}_sd{sd_log}_all.pt")
            else:
                torch.save(net, f"./checkpoints/{dataset}_{model}_best_{sub}.pt")

        train_losses.append(loss.detach().cpu())
        train_accs.append(acc_tr / tr_len)
        val_losses.append(TL / len(validloader))
        val_accs.append(acc_val / val_len)

############################################## test ####################################################################
        if dataset.startswith('bcicha'):
            test_acc, test_loss = testNetwork_auc(net, testloader, device, model)
        else:
            test_acc, test_loss = testNetwork(net, testloader, device, model)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        if update_best_test == 1:
            test_from_best_val = test_acc
        update_best_test = 0

        if update_best_train_loss:
            test_from_best_train_loss = test_acc

        best_test = test_acc if test_acc > best_test else best_test
        if verbose:
            print(f'test_acc:{test_acc.__round__(3)}')
            print(f"best_test:{best_test.__round__(3)}")
            print(f"best_loss:{bestLoss.__round__(3)}")
            print(f"test_from_best_val_acc:{test_from_best_val.__round__(3)}")
            print(f"test_from_best_val_loss:{test_from_best_val_loss.__round__(3)}")
            print(f"test_from_best_train_loss:{test_from_best_train_loss.__round__(3)}")

        if TL / len(validloader) < bestLoss:
            bestLoss = (TL / len(validloader)).item()
            test_from_best_val_loss = test_acc

    if save_results:
        save_output(
            model_name=model, dataset=dataset, sub=sub, bs=bs, lr=lr, wd=wd, alpha=subject_weights,
            epochs=epochs, sd=sd_log,
            results=[
                np.array([loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]),
                np.array([acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in train_accs]),
                np.array([loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]),
                np.array([acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in val_accs]),
                np.array([loss if not isinstance(loss, torch.Tensor) else loss.cpu().item() for loss in test_losses]),
                np.array([acc.cpu().item() if isinstance(acc, torch.Tensor) else acc for acc in test_accs])
            ],
            **kwargs
        )

    return TL/len(validloader), acc_val/val_len


def testNetwork(net, testloader, device, model):
    net.eval()
    acc = 0
    test_len = 0
    total_loss = 0.0
    CE = nn.CrossEntropyLoss()

    for row in testloader:

        x_orig, yb = row
        if model == 'ATCNet' or model == 'EEGFormer':
            x_orig = x_orig.unsqueeze(1)
        x_orig = x_orig.to(device)

        yb = yb.to(device)

        test_len += yb.shape[0]
        with torch.no_grad():
            pred = net(x_orig)
            acc += (torch.max(pred, 1).indices == yb).sum().item()
            total_loss += CE(pred, yb).item()

    return acc / test_len, total_loss / len(testloader)


def testNetwork_auc(net, testloader, device, model):
    net.eval()
    softmax = nn.Softmax(dim=1)
    y_pred = torch.empty(0, device=device)
    y_true = torch.empty(0, device=device)
    CE = nn.CrossEntropyLoss()
    total_loss = 0.0

    for row in testloader:
        with torch.no_grad():
            x_orig, yb = row
            if model == 'ATCNet' or model == 'EEGFormer':
                x_orig = x_orig.unsqueeze(1)
            x_orig = x_orig.to(device)
            yb = yb.to(device)

            pred = net(x_orig)
            total_loss += CE(pred, yb).item()

            y_pred = torch.cat((y_pred, pred[:, 1]), 0)
            y_true = torch.cat((y_true, yb), 0)

    auc = ras(y_true.detach().cpu().numpy(),
              y_pred.detach().cpu().numpy())
    return auc, total_loss / len(testloader)
