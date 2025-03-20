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

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from torch import einsum
import torch.nn.functional as F
import torch.nn as nn
import scipy.ndimage as ndimage
from torch import einsum

from monai.data import decollate_batch, MetaTensor

def dilation(seg):
    seg = seg.astype(np.int8)
    for i in range(np.shape(seg)[0]):
        seg_i = seg[i]
        seg_i = ndimage.binary_dilation(seg_i, structure=np.ones((10,10,2)))
        seg_i = MetaTensor(seg_i)
        seg_i = torch.unsqueeze(seg_i, 0)
        seg_i = torch.unsqueeze(seg_i, 0)
        if i == 0:
            res = seg_i
        else:
            res = torch.cat([res, seg_i], dim = 0)
    return res

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    epoch_loss = 0
    step = 0
    start_time = time.time()
    for idx, batch_data in enumerate(loader):
        step += 1
        T2, labels = batch_data["image"].to(args.device), batch_data["label"].to(args.device)
        
        if args.data_loader == 'bp_seg':
            adc, dwi = batch_data["adc"].to(args.device), batch_data["dwi"].to(args.device)
            seg = batch_data["seg"]
            seg = torch.squeeze(seg, 1)
            # print(seg.size())
            seg_dil = dilation(seg)
            seg_dil = seg_dil.to(args.device)
            T2 = einsum("bkwhz,bkwhz->bkwhz", T2, seg_dil)
            adc = einsum("bkwhz,bkwhz->bkwhz", adc, seg_dil)
            dwi = einsum("bkwhz,bkwhz->bkwhz", dwi, seg_dil)
            inputs = torch.cat((T2,adc,dwi),1)
        elif args.data_loader == '2c_bp_seg':
            adc, dwi = batch_data["adc"].to(args.device), batch_data["dwi"].to(args.device)
            c0, c1 = batch_data["c0"].to(args.device), batch_data["c1"].to(args.device)
            seg = batch_data["seg"]
            seg = torch.squeeze(seg, 1)
            seg_dil = dilation(seg)
            seg_dil = seg_dil.to(args.device)
            T2 = einsum("bkwhz,bkwhz->bkwhz", T2, seg_dil)
            adc = einsum("bkwhz,bkwhz->bkwhz", adc, seg_dil)
            dwi = einsum("bkwhz,bkwhz->bkwhz", dwi, seg_dil)
            c0 = einsum("bkwhz,bkwhz->bkwhz", c0, seg_dil)
            c1 = einsum("bkwhz,bkwhz->bkwhz", c1, seg_dil)
            inputs = torch.cat((T2,adc,dwi,c0,c1),1)

        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            if args.model_name == 'dense121':
                outputs = model(inputs)
            elif args.model_name == 'dense121_plus_1':
                rsi_max= batch_data["RSIrs_max"] / 1000.0
                rsi_max = MetaTensor(rsi_max)
                rsi_max = rsi_max.to(args.device)
                rsi_max = torch.unsqueeze(rsi_max, 1)
                outputs = model(inputs, rsi_max)
                del inputs
            loss = loss_func(outputs, labels)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        epoch_loss +=  loss.item()
        
    
    epoch_loss /= step

    print(
        "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
        "loss: {:.4f}".format(epoch_loss),
        "time {:.2f}s".format(time.time() - start_time),
    )
        
    for param in model.parameters():
        param.grad = None
    return epoch_loss


def val_epoch(model, loader, metric_func, epoch, args, post_pred, post_label):
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        start_time = time.time()
        y_pred = torch.tensor([], dtype=torch.float32, device=args.device)
        y = torch.tensor([], dtype=torch.long, device=args.device)
        for idx, batch_data in enumerate(loader):
            T2, labels = batch_data["image"].to(args.device), batch_data["label"].to(args.device)

            if args.data_loader == 'bp_seg':
                adc, dwi = batch_data["adc"].to(args.device), batch_data["dwi"].to(args.device)
                seg = batch_data["seg"]
                seg = torch.squeeze(seg, 1)
                # print(seg.size())
                seg_dil = dilation(seg)
                seg_dil = seg_dil.to(args.device)
                T2 = einsum("bkwhz,bkwhz->bkwhz", T2, seg_dil)
                adc = einsum("bkwhz,bkwhz->bkwhz", adc, seg_dil)
                dwi = einsum("bkwhz,bkwhz->bkwhz", dwi, seg_dil)
                inputs = torch.cat((T2,adc,dwi),1)
            elif args.data_loader == '2c_bp_seg':
                adc, dwi = batch_data["adc"].to(args.device), batch_data["dwi"].to(args.device)
                c0, c1 = batch_data["c0"].to(args.device), batch_data["c1"].to(args.device)
                seg = batch_data["seg"]
                seg = torch.squeeze(seg, 1)
                # print(seg.size())
                seg_dil = dilation(seg)
                seg_dil = seg_dil.to(args.device)
                T2 = einsum("bkwhz,bkwhz->bkwhz", T2, seg_dil)
                adc = einsum("bkwhz,bkwhz->bkwhz", adc, seg_dil)
                dwi = einsum("bkwhz,bkwhz->bkwhz", dwi, seg_dil)
                c0 = einsum("bkwhz,bkwhz->bkwhz", c0, seg_dil)
                c1 = einsum("bkwhz,bkwhz->bkwhz", c1, seg_dil)
                inputs = torch.cat((T2,adc,dwi,c0,c1),1)
            

            if args.model_name == 'dense121':
                y_pred = torch.cat([y_pred, model(inputs)], dim=0)
            elif args.model_name == 'dense121_plus_1':
                rsi_max = batch_data["RSIrs_max"] / 1000.0
                rsi_max = MetaTensor(rsi_max)
                rsi_max = rsi_max.to(args.device)
                rsi_max = torch.unsqueeze(rsi_max, 1)
                y_pred = torch.cat([y_pred, model(inputs, rsi_max)], dim=0)
            y = torch.cat([y, labels], dim=0)
        print(y)
        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
        metric_func(y_pred_act, y_onehot)
        auc_result = metric_func.aggregate()
        metric_func.reset()
        del y_pred_act, y_onehot
        print(
            "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
            "acc",
            acc_metric,
            "curren_auc:",
            auc_result,
            "time {:.2f}s".format(time.time() - start_time),
        )
            
    return auc_result

def save_checkpoint(model, epoch, args, filename="model.pt", best_loss=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_loss": best_loss, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    metric_func,
    args,
    scheduler,
    start_epoch,
    post_pred,
    post_label,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    min_train_loss = 10000
    for epoch in range(start_epoch, args.max_epochs):
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler=scaler,
            epoch=epoch,
            loss_func=loss_func,
            args=args
        )
        b_new_best = False
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            if train_loss < min_train_loss:
                print("new best ({:.6f} --> {:.6f}). ".format(min_train_loss, train_loss))
                min_train_loss = train_loss
                b_new_best = True
                if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(
                        model, epoch, args, best_loss=train_loss, optimizer=optimizer, scheduler=scheduler
                    )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        
        if (epoch + 1) == args.max_epochs:
            save_checkpoint(
                model, epoch, args, filename="model_final.pt",best_loss=train_loss, optimizer=optimizer, scheduler=scheduler
            )
            epoch_time = time.time()
            val_auc = val_epoch(
                model,
                val_loader,
                metric_func=metric_func,
                epoch=epoch,
                args=args,
                post_pred=post_pred,
                post_label=post_label,
            )
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "auc",
                    val_auc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_auc", val_auc, epoch)

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_auc)

    return val_auc
