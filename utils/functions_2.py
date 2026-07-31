from idlelib.pyparse import trans

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR

import sys
import os

from hyperbolic_lib.toybox.mnist_euclidean import train

sys.path.append("..")
from models.optimizer import MixOptimizer
from sklearn.metrics import roc_auc_score as ras
import numpy as np
from utils.save_results import save_output
import hyperbolic_lib.lib.geoopt as geoopt
from hyperbolic_lib.lib.geoopt.optim import RiemannianSGD, RiemannianAdam, RiemannianAdamW
import argparse
import torcheeg.transforms as transforms
from utils.helpers import get_recall, eval_dataset

def get_param_groups(model, lr_manifold, weight_decay_manifold):
    no_decay = ["scale"]
    k_params = [".k"]

    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                   and not any(nd in n for nd in no_decay)
                   and not isinstance(p, geoopt.ManifoldParameter)
                   and not any(nd in n for nd in k_params)
            ],
            "name": "1"
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                   and isinstance(p, geoopt.ManifoldParameter)
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
                   and any(nd in n for nd in k_params)
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
        {"name": "lcas", "params": lcas, "lr_scale": 1, "weight_decay": weight_decay},
        {"name": "proxies", "params": proxies, "lr_scale": 1, "weight_decay": weight_decay},
        {"name": "k_group", "params": ks, "lr_scale": 1, "weight_decay": weight_decay},
        {"name": "man_group", "params": manifold_params, "lr_scale": 1, "weight_decay": weight_decay},
    ]


def trainNetwork(net, trainloader, validloader, testloader, model_path=None, model=None, dataset=None, bs=64,
                 iterations=500, lr=5 * 1e-4, wd=None, repeat=None, sub=None, epochs=None, subject_weights=None,
                 verbose=True, save_model=False, save_results=True, early_stopping=False, grace_period=20,
                 hyperbolic=False, clip_grad=0, loss_1=None, loss_2=None, **kwargs):
    best_test = 0
    best_val = 0
    test_from_best_val = 0
    update_best_test = 0


    train_mean, train_std = get_dataset_statistics(trainloader)
    train_mean= train_mean.squeeze()
    train_std = train_std.squeeze()
    train_t = transforms.Compose([transforms.MeanStdNormalize(train_mean.numpy(), train_std.numpy(), axis=1),
                                  transforms.ToTensor(),
                                  #transforms.RandomNoise(mean=train_mean.unsqueeze(-1).unsqueeze(0).numpy(), std=0.1, p=0.4)
                                  ])

    device = kwargs['device']
    softmax = nn.Softmax(dim=1)
    CE = nn.CrossEntropyLoss()

    if not hyperbolic:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
        # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.6)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
        optimizer = MixOptimizer(optimizer)
    else:
        # optimizer = RiemannianSGD(get_param_groups(net, lr, wd), lr=lr, weight_decay=wd, momentum=0.6, nesterov=False)
        print("hyperbolic")

        param_groups = get_params_groups(net, loss_1, loss_2, weight_decay=wd) if loss_1 is not None else get_param_groups(net, lr, wd)
        optimizer = RiemannianAdamW(param_groups, lr=lr, weight_decay=wd)
        # optimizer = MixOptimizer(optimizer)
    # train_scheduler = MultiStepLR(
    #     optimizer, milestones=[250], gamma=0.1
    # )
    bestLoss = 1e10
    bestLoss_early_stopping = 1e10
    val_len = len(validloader)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_accs = []

    no_improvement_count = 0
    for i, param_group in enumerate(optimizer.param_groups):
        # lr = args.lr * (kwargs.bs * utils.get_world_size()) / 180.
        param_group["lr"] = lr * param_group["lr_scale"]
    for ite in range(iterations):
        net.train()
        acc_val = 0
        acc_tr = 0
        tr_len = 0

        train_embeds = []
        train_ys = []

        for row in trainloader:
            xb, yb = row[:-1], row[-1]
            tr_len += yb.shape[0]
            xb = [train_t(eeg=x.cpu().numpy())["eeg"].to(device) for x in xb]
            #xb = [x.to(device) for x in xb]
            yb = yb.to(device)
            out, embeds= net(*xb, return_embeds=True)

            temp = net.manifold.logmap0(out)[..., 1:]
            loss = loss_1(temp, yb)
            loss = loss #+ loss_2(embeds, yb, 3)

            train_embeds.append(embeds)
            train_ys.append(yb)
            # distances = net.manifold.dist(embeds, embeds.unsqueeze(-2))
            # distance_mask = yb!=yb.unsqueeze(0).T
            # additional_loss = distances[distance_mask].sum()/2
            # distance_orig = net.manifold.dist0(embeds).sum()
            # loss = loss + 0.2*distance_orig/additional_loss# loss + 0.05 / (1 + torch.log(1 + additional_loss)) + 0.05* torch.log(1 + distance_orig) #- 0.0001*additional_loss + 0.0001*distance_orig

            optimizer.zero_grad()
            if torch.isnan(loss).sum() > 0:
                print("break")
            loss.backward(retain_graph=True)

            if clip_grad > 0:
                param_norms = clip_gradients_value(net, clip_grad, losses=[loss])

            optimizer.step()
            acc_tr += (torch.max(out, 1).indices == yb).sum().item()

        net.eval()
        TL = 0
        t = transforms.MeanStdNormalize(train_mean.numpy(), train_std.numpy(), axis=-2)

        all_embeds, all_y = eval_dataset(net, validloader, t, net.manifold.k.device)
        acc_val, _ = get_recall(all_embeds, all_y, manifold=net.manifold)

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
            print(f'train_loss:{loss:.4f}    val_loss:{TL / val_len:.4f}')
            print(f'train_acc:{acc_tr / tr_len:.4f}    val_acc:{acc_val / val_len:.4f}')
            #print(f"k: {net.manifold.k}")
        if acc_val  > best_val:
            best_val = acc_val
            update_best_test = 1
            torch.save(torch.cat(train_embeds), "embeds.pt")
            torch.save(torch.cat(train_ys), "y.pt")
            torch.save(net.manifold.k, "k.pt")
        #train_scheduler.step()

        train_losses.append(loss.detach().cpu())
        train_accs.append(acc_tr / tr_len)
        val_losses.append(TL / val_len)
        val_accs.append(acc_val)

        # if min(val_losses) not in val_losses[-50:]:
        #     print(min(val_losses))
        #     break

        if TL < bestLoss:
            if save_model:
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                if verbose:
                    print(f'create {model_path}.......')
                bestLoss = TL
                final_path = os.path.join(model_path, f'repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}.pt')
                if verbose:
                    print(f'saving to {final_path}')
                # torch.save(net, final_path)
                # testnet = torch.load(final_path)
                # if dataset.startswith('bcicha'):
                #    test_acc = testNetwork_auc(testnet, testloader, device)
                # else:
                #    test_acc = testNetwork(testnet, testloader, device)

                test_acc = 0
                #test_from_best_val = test_acc
            else:
                if dataset.startswith('bcicha'):
                    test_acc = testNetwork_auc(net, testloader, device, train_mean, train_std )
                else:
                    test_acc = testNetwork(net, testloader, device, train_mean, train_std )
                #test_from_best_val = test_acc
            test_accs.append(test_acc)
            test_from_best_val = test_acc if update_best_test == 1 else test_from_best_val
            update_best_test = 0
            best_test = test_acc if test_acc > best_test else best_test
            if verbose:
                print(f'test_acc:{test_acc.__round__(3)}')
                print(f"best_test:{best_test.__round__(3)}")
                print(f"best_val:{best_val.__round__(3)}")
                print(f"test_from_best_val:{test_from_best_val.__round__(3)}")
    if save_model:
        net = torch.load(os.path.join(model_path, f'repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}.pt'))
    net.train()

    if save_results:
        save_output(model_name=model, dataset=dataset, sub=sub, bs=bs, lr=lr, wd=wd, alpha=subject_weights, epochs=epochs,
                    results=[np.array(train_losses),
                             np.array(train_accs),
                             np.array(val_losses),
                             np.array(val_accs),
                             np.array(test_accs)],
                    **kwargs)

    return TL / val_len, acc_val / val_len  # return last validation loss and acc
    # return net


def testNetwork(net, testloader, device, train_mean, train_std ):
    net.eval()

    t = transforms.MeanStdNormalize(train_mean.numpy(), train_std.numpy(), axis=1)
    all_embeds, all_y = eval_dataset(net, testloader, t, net.manifold.k.device)
    acc, _ = get_recall(all_embeds, all_y, manifold=net.manifold)

    return acc


def testNetwork_auc(net, testloader, device, train_mean, train_std ):
    net.eval()
    acc = 0
    softmax = nn.Softmax(dim=1)
    y_pred = torch.empty(0, device=device)
    y_true = torch.empty(0, device=device)
    t = transforms.MeanStdNormalize(train_mean.numpy(), train_std.numpy(), axis=1)
    for row in testloader:
        with torch.no_grad():
            xb, yb = row[:-1], row[-1]
            xb = [torch.tensor(t(eeg=x.cpu().numpy())["eeg"]).to(device) for x in xb]
            #xb = [x.to(device) for x in xb]
            yb = yb.to(device)
            pred = net(*xb)
            y_pred = torch.cat((y_pred, pred[:, 1]), 0)
            y_true = torch.cat((y_true, yb), 0)

    return ras(y_true.detach().cpu().numpy(),
               y_pred.detach().cpu().numpy())



