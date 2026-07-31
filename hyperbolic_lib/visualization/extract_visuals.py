import os
import sys

from datetime import datetime

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)

import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from classification.utils import create_parser
from classification.utils.utils import (
    accuracy,
    save_checkpoint,
    copy_files,
    select_optimizer,
    init_default_trackers
)
from lib.models.resnet import resnet_selector
from datasets.datasets import get_dataset

best_acc1 = 0
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'


def validate(args, val_loader, model):

    # switch to evaluate mode

    embeds = []
    targets = []

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):

            if args.gpu != -1:
                gpu_device = "cuda:" + str(args.gpu)
                images = images.to(device=gpu_device)
                if type(target) in (list, tuple):
                    target = [label.to(device=gpu_device) for label in target]
                else:
                    target = target.to(device=gpu_device)
            targets.append(target.detach().cpu())
            if args.highprecision:
                images = images.to(dtype=torch.float64)

            outputs = model(images, return_embeddings=True)
            embeds.append(outputs[-1].detach().cpu())


    return embeds, targets


def get_embeddings(resume_path):

    resume_path_1 = resume_path.split("/")[:-2]
    args = "/".join(resume_path_1) + "/args.json"

    with open(args, "r") as args_file:
        args = json.load(args_file)

    from argparse import Namespace

    args = Namespace(**args)
    args.gpu = 0

    # data preprocessing:
    (training_loader, test_loader, _, _), num_classes = get_dataset(
        args.dataset, args.datapath, batch_size=args.b, n_workers=args.num_workers
    )

    # create temporary args used for training the model

    # get the model

    net = resnet_selector(loss=nn.CrossEntropyLoss, num_classes=num_classes, args=args)
    net = net.to(device="cuda:" + str(args.gpu))

    time_run = datetime.now().strftime(DATE_FORMAT)

 # create checkpoint folder to save model
    project_path = os.path.join(
        working_dir, "classification", "runs", args.run_name, time_run
    )
    checkpoint_path = os.path.join(project_path, args.checkpoint)

    if not os.path.exists(project_path):
        os.makedirs(project_path)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    args_path = os.path.join(project_path, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f)

    if args.backup:
        copy_files(
            os.path.dirname(os.path.abspath(__file__)) + "/",
            project_path + "/",
            ["models", "conf"],
        )
        copy_files(
            os.path.dirname(os.path.abspath(__file__)) + "/../",
            project_path + "/",
            ["lib"],
        )

################################################# resume checkpoints #######################################################

    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        if args.gpu == -1:
            checkpoint = torch.load(resume_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(args.gpu)
            checkpoint = torch.load(resume_path, map_location=loc)

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k.replace("module.", "")  # remove module.
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)
        if args.gpu != -1:
            gpu_device = "cuda:" + str(args.gpu)
            net = net.to(gpu_device)

        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))

    gpu_device = "cuda:" + str(args.gpu)
    net = net.to(gpu_device)

    with torch.no_grad():
        # validate and get acc1 for performance comparison
        embeddings, targets = validate(args, test_loader, net)

    return net, embeddings, targets
