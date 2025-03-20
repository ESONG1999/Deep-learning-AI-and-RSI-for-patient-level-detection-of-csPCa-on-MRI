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

import json
import math
import os

import numpy as np
import torch

from monai import data, transforms

def datafold_read(datalist, basedir, args):
    with open(datalist) as f:
        json_data = json.load(f)

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and (d["fold"] == int(args.val_list[0]) or d["fold"] == int(args.val_list[-1])):
            val.append(d)
        # elif "fold" in d and (d["fold"] == int(args.tr_list[0]) or d["fold"] == int(args.tr_list[-1])):
        else:
            tr.append(d)

    return tr, val

def get_loader_bp_seg(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, args=args)
    print('Number of training files:' + str(len(train_files)))
    print('Number of validation files:' + str(len(validation_files)))
    train_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
            # keys=["image", "adc", "dwi", "c0", "c1", "c2", "c3"]
            # keys=["image", "c0", "c1", "c2", "c3"]
            # keys=["image", "c0", "c1", "c2", "c3", "c0_gx", "c0_gy", "c0_gz"]
            transforms.LoadImaged(keys=["image", "adc", "dwi", "seg"], ensure_channel_first=True),
            transforms.Spacingd(
            keys=["image", "adc", "dwi","seg"],
            pixdim=(0.5, 0.5, 3.0),
            mode=("bilinear", "bilinear", "bilinear", "nearest")),
            transforms.CenterSpatialCropd(keys=["image", "adc", "dwi", "seg"], roi_size=(256, 256, 32)),
            transforms.NormalizeIntensityd(
                keys=["image", "adc", "dwi"], nonzero=True, channel_wise=True),
            transforms.SpatialPadd(keys=["image", "adc", "dwi", "seg"], spatial_size=(256, 256, 32)),
            transforms.RandFlipd(keys=["image", "adc", "dwi", "seg"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "adc", "dwi", "seg"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "adc", "dwi", "seg"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "adc", "dwi", "seg"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys=["image", "adc", "dwi"], factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys=["image", "adc", "dwi"], offsets=0.1, prob=args.RandShiftIntensityd_prob),
        ]
    )
    val_transform = transforms.Compose(
        [
            # transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
            transforms.LoadImaged(keys=["image", "adc", "dwi", "seg"], ensure_channel_first=True),
            transforms.Spacingd(
            keys=["image", "adc", "dwi", "seg"],
            pixdim=(0.5, 0.5, 3.0),
            mode=("bilinear", "bilinear", "bilinear", "nearest")),
            transforms.CenterSpatialCropd(keys=["image", "adc", "dwi", "seg"], roi_size=(256, 256, 32)),
            transforms.NormalizeIntensityd(
                keys=["image", "adc", "dwi"], nonzero=True, channel_wise=True),
            transforms.SpatialPadd(keys=["image", "adc", "dwi", "seg"], spatial_size=(256, 256, 32)),
        ]
    )

    if args.test_mode:
        test_ds = data.Dataset(data=validation_files, transform=val_transform)
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle = True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        loader = [train_loader, val_loader]

    return loader
