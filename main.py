# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
import torch.multiprocessing

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils_bp_seg import get_loader_bp_seg
from utils.data_utils_bp_2c_seg import get_loader_2c_bp_seg

import monai
from monai.transforms import Activations, AsDiscrete, Compose
from monai.metrics import ROCAUCMetric
from model import DenseNet121_plus_1

parser = argparse.ArgumentParser(description="Clinical Significant Cancer Classification pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=120, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=8, type=int, help="number of batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--model_name", default="unetr", type=str, help="model name")
parser.add_argument("--in_channels", default=5, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=60, type=int, help="number of warmup epochs")
parser.add_argument("--loss_function", default="mean_var", type=str, help="mean and variance function")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--dropout", action="store_true", help="enable dropout")
parser.add_argument("--tr_list", default=[1], type=list, help="training fold")
parser.add_argument("--val_list", default=[5], type=list, help="validation fold")
parser.add_argument("--data_loader", default="original", type=str, help="data loader name")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs_250311_noval/" + args.logdir
    torch.multiprocessing.set_sharing_strategy('file_system')
    main_worker(gpu=0, args=args)

def main_worker(gpu, args):
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    if args.data_loader == 'bp_seg':
        loader = get_loader_bp_seg(args)
    elif args.data_loader == '2c_bp_seg':
        loader = get_loader_2c_bp_seg(args)

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    pretrained_dir = args.pretrained_dir
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (args.model_name is not None):
        if args.model_name == 'dense121':
            if args.dropout:
                model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=args.in_channels, out_channels=2, dropout_prob=0.3).to(args.device)
            else:
                model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=args.in_channels, out_channels=2).to(args.device)
        elif args.model_name == 'dense121_plus_1':
            if args.dropout:
                model = DenseNet121_plus_1(spatial_dims=3, in_channels=args.in_channels, out_channels=2, dropout_prob=0.3).to(args.device)
            else:
                model = DenseNet121_plus_1(spatial_dims=3, in_channels=args.in_channels, out_channels=2).to(args.device)

        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict)
            print("Use pretrained weights")

        if args.resume_jit:
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError("Unsupported model " + str(args.model_name))
    
    # Loss Function
    loss_function = torch.nn.CrossEntropyLoss()
    # Metric Function
    auc_metric = ROCAUCMetric()
    # Post Processing Function
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=loss_function,
        metric_func=auc_metric,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_pred=post_pred,
        post_label=post_label,
    )
    return accuracy


if __name__ == "__main__":
    main()
