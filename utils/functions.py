import sys

sys.path.append("..")
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR
import torcheeg.transforms as transforms

from sklearn.metrics import roc_auc_score as ras
import numpy as np

from utils.save_results import save_output
from utils.utils import (get_loss,
                         get_dataset_statistics,
                         get_param_groups,
                         get_dataset_max_min,
                         get_transforms,
                         get_params_groups,
                         clip_gradients_value)

from models.optimizer import MixOptimizer
from hyperbolic_lib.lib.geoopt.optim import RiemannianSGD, RiemannianAdam, RiemannianAdamW


APPLY_MEAN = False

def trainNetwork(net, trainloader, validloader, testloader, model_path=None, model=None, dataset=None, bs=64,
                 iterations=500, lr=5 * 1e-4, wd=None, repeat=None, sub=None, epochs=None, subject_weights=None,
                 verbose=True, save_model=False, save_results=True, early_stopping=False, grace_period=20,
                 hyperbolic=False, clip_grad=0, loss_1=None, loss_2=None, **kwargs):


    best_test = 0
    best_val = 0
    test_from_best_val = 0
    update_best_test = 0

    device = kwargs['device']
    CE = nn.CrossEntropyLoss()

    if kwargs.get('finetune', False):
        saved_model_path = os.path.join(model_path, f"{dataset}/best_model_all.pth")
        if os.path.exists(saved_model_path):
            print(f"Loading saved model from {saved_model_path} for fine-tuning...")
            net.load_state_dict(torch.load(saved_model_path))
            net = net.to(device)
        else:
            print(f"Saved model not found at {saved_model_path}. Proceeding without fine-tuning.")

    if not hyperbolic:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
        optimizer = MixOptimizer(optimizer)
    else:
        # optimizer = RiemannianSGD(get_param_groups(net, lr, wd), lr=lr, weight_decay=wd, momentum=0.6, nesterov=False)
        print("hyperbolic")

        param_groups = get_params_groups(net, loss_1, loss_2, weight_decay=wd) if loss_1 is not None else get_param_groups(net, lr, wd)

        optimizer = RiemannianAdamW(param_groups, lr=lr, weight_decay=wd)
        #optimizer = MixOptimizer(optimizer)

    #train_scheduler = MultiStepLR(
    #    optimizer, milestones=[60, 120], gamma=0.1
    #)

    bestLoss = 1e10
    bestLoss_early_stopping = 1e10

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_accs = []

    no_improvement_count = 0

    for ite in range(iterations):
        net.train()
        acc_val = 0
        acc_tr = 0
        tr_len = 0
        val_len = 0


        train_embeds = []
        train_ys = []

        #for i, param_group in enumerate(optimizer.param_groups):
            #lr = args.lr * (kwargs.bs * utils.get_world_size()) / 180.
        #    param_group["lr"] = lr * param_group["lr_scale"]


############################################# train ####################################################################

        t, train_mean, train_std = get_transforms(training=True,
                                                  dataloader=trainloader,
                                                  apply_mean=APPLY_MEAN,
                                                  return_statistics=True)

        for row in trainloader:
            xb, yb = row[:-1], row[-1]
            tr_len += yb.shape[0]
            xb = [t(eeg=x.cpu().numpy())["eeg"].to(device) for x in xb]

            yb = yb.to(device)
            if kwargs['sub_aux_loss']:
                out, embeds, subject_pred = net(*xb, return_embeds=True, aux_loss_sub=True)
            else:
                out, embeds = net(*xb, return_embeds=True, aux_loss_sub=False)
            loss = get_loss(CE, net, out, yb, embeds=embeds, loss_1=loss_1, loss_2=loss_2)

            if kwargs['sub_aux_loss']:
                # print(row)
                subject_ids = row[1][..., 0].squeeze(1).to(device).long() -1


                # print(subject_ids.shape)
                # print(subject_pred.shape)
                # print(set(subject_ids))
                aux_loss_sub = F.cross_entropy(subject_pred, subject_ids)
                loss += 0.2 * aux_loss_sub


            train_embeds.append(embeds)
            train_ys.append(yb)

            optimizer.zero_grad()

            loss.backward(retain_graph=True)

            if clip_grad > 0:
                param_norms = clip_gradients_value(net, clip_grad, losses=[loss])

            optimizer.step()
            acc_tr += (torch.max(out, 1).indices == yb).sum().item()

        #train_scheduler.step()

########################################## validate ####################################################################
        #net.eval()
        TL = 0

        t = get_transforms(training=False,
                           mean=train_mean,
                           std=train_std,
                           apply_mean=APPLY_MEAN)

        for row in validloader:
            xb, yb = row[:-1], row[-1]
            xb = [t(eeg=x.cpu().numpy())["eeg"].to(device) for x in xb]

            yb = yb.to(device)
            val_len += yb.shape[0]
            with torch.no_grad():
                if kwargs['sub_aux_loss']:
                    out, embeds, subject_pred = net(*xb, return_embeds=True, aux_loss_sub=True)
                else:
                    out, embeds = net(*xb, return_embeds=True)
                acc_val += (torch.max(out, 1).indices == yb).sum().item()

                TL += get_loss(CE, net, out, yb, embeds=embeds, loss_1=loss_1, loss_2=loss_2)

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
            print(f"k: {net.manifold.k}")

        if acc_val / val_len > best_val:
            best_val = acc_val / val_len
            update_best_test = 1
            torch.save(torch.cat(train_embeds), "embeds.pt")
            torch.save(torch.cat(train_ys), "y.pt")
            torch.save(net.manifold.k, "k.pt")
            torch.save(net.state_dict(), os.path.join(model_path, f"{dataset}/best_model_{sub}.pth"))

        train_losses.append(loss.detach().cpu())
        train_accs.append(acc_tr / tr_len)
        val_losses.append(TL / len(validloader))
        val_accs.append(acc_val / val_len)

############################################## test ####################################################################
        if dataset.startswith('bcicha'):
            test_acc = testNetwork_auc(net, testloader, device, train_mean, train_std)
        else:
            test_acc = testNetwork(net, testloader, device, train_mean, train_std, kwargs)
        test_accs.append(test_acc)

        test_from_best_acc = test_acc if update_best_test == 1 else test_from_best_val
        update_best_test = 0

        best_test = test_acc if test_acc > best_test else best_test
        if verbose:
            print(f'test_acc:{test_acc.__round__(3)}')
            print(f"best_test:{best_test.__round__(3)}")
            print(f"best_loss:{bestLoss.__round__(3)}")
            print(f"test_from_best_val:{test_from_best_val.__round__(3)}")
            print(f"test_from_best_acc:{test_from_best_acc.__round__(3)}")

        if TL/ len(validloader)< bestLoss:
            bestLoss = (TL/ len(validloader)).item()
            test_from_best_val = test_acc

    if save_results:
        save_output(model_name=model, dataset=dataset, sub=sub, bs=bs, lr=lr, wd=wd, alpha=subject_weights,
                    epochs=epochs,
                    results=[
                        np.array([loss.cpu() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]),
                        np.array([acc.cpu() if isinstance(acc, torch.Tensor) else acc for acc in train_accs]),
                        np.array([loss.cpu() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]),
                        np.array([acc.cpu() if isinstance(acc, torch.Tensor) else acc for acc in val_accs]),
                        np.array([acc.cpu() if isinstance(acc, torch.Tensor) else acc for acc in test_accs])],
                    **kwargs)

    return TL/len(validloader), acc_val/val_len

def testNetwork(net, testloader, device, train_mean, train_std, kwargs ):
    net.eval()
    acc = 0
    test_len=0

    t = get_transforms(training=False,
                       mean=train_mean,
                       std=train_std,
                       apply_mean=APPLY_MEAN)

    for row in testloader:
        xb, yb = row[:-1], row[-1]
        xb = [t(eeg=x.cpu().numpy())["eeg"].to(device) for x in xb]

        yb = yb.to(device)
        test_len += yb.shape[0]

        with torch.no_grad():
            pred = net(*xb)

            pred = pred[0]

            acc += (torch.max(pred, 1).indices == yb).sum().item()

    return acc / test_len


def testNetwork_auc(net, testloader, device, train_mean, train_std ):
    net.eval()
    acc = 0
    softmax = nn.Softmax(dim=1)
    y_pred = torch.empty(0, device=device)
    y_true = torch.empty(0, device=device)

    t = get_transforms(training=False,
                       mean=train_mean,
                       std=train_std,
                       apply_mean=APPLY_MEAN)

    for row in testloader:
        with torch.no_grad():
            xb, yb = row[:-1], row[-1]
            xb = [torch.tensor(t(eeg=x.cpu().numpy())["eeg"]).to(device) for x in xb]
            #xb = [x.to(device) for x in xb]
            yb = yb.to(device)
            pred, _ = net(*xb)
            y_pred = torch.cat((y_pred, pred[:, 1]), 0)
            y_true = torch.cat((y_true, yb), 0)

    return ras(y_true.detach().cpu().numpy(),
               y_pred.detach().cpu().numpy())



