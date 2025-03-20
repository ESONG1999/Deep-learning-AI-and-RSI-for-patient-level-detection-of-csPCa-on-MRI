# Copyright 2020 - 2022 MONAI Consortium
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
from functools import partial

# import nibabel as nib
import numpy as np
import torch
from utils.data_utils_bp_seg import get_loader_bp_seg
from utils.data_utils_bp_2c_seg import get_loader_2c_bp_seg
from torch import einsum
import torch.multiprocessing


import pandas as pd 
import monai
from monai.transforms import Activations, AsDiscrete, Compose
from monai.metrics import ROCAUCMetric
from model import DenseNet121_plus_1
from torch import einsum
from monai.data import decollate_batch, MetaTensor
import scipy.ndimage as ndimage


from PIL import Image

parser = argparse.ArgumentParser(description="Clinical Significant Cancer Classification pipeline")
parser.add_argument("--data_dir", default="dataset/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="exp_name", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset.json", type=str, help="dataset json file")
parser.add_argument("--tr_list", default=[0], type=list, help="training fold")
parser.add_argument("--val_list", default=[1], type=list, help="testing fold")
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--in_channels", default=5, type=int, help="number of input channels")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--model_name", default="unetr", type=str, help="model name")
parser.add_argument("--data_loader", default="original", type=str, help="data loader name")
parser.add_argument(
    "--pretrained_dir",
    default="pretrained_dir/",
    type=str,
    help="pretrained checkpoint directory",
)

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

def _binary_clf_curve(y_true, y_score):
    """
    Calculate true and false positives per binary classification
    threshold (can be used for roc curve or precision/recall curve);
    the calcuation makes the assumption that the positive case
    will always be labeled as 1
    Parameters
    ----------
    y_true : 1d ndarray, shape = [n_samples]
        True targets/labels of binary classification
    y_score : 1d ndarray, shape = [n_samples]
        Estimated probabilities or scores
    Returns
    -------
    tps : 1d ndarray
        True positives counts, index i records the number
        of positive samples that got assigned a
        score >= thresholds[i].
        The total number of positive samples is equal to
        tps[-1] (thus false negatives are given by tps[-1] - tps)
    fps : 1d ndarray
        False positives counts, index i records the number
        of negative samples that got assigned a
        score >= thresholds[i].
        The total number of negative samples is equal to
        fps[-1] (thus true negatives are given by fps[-1] - fps)
    thresholds : 1d ndarray
        Predicted score sorted in decreasing order
    References
    ----------
    Github: scikit-learn _binary_clf_curve
    - https://github.com/scikit-learn/scikit-learn/blob/ab93d65/sklearn/metrics/ranking.py#L263
    """

    # sort predicted scores in descending order
    # and also reorder corresponding truth values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically consists of tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve
    distinct_indices = np.where(np.diff(y_score))[0]
    end = np.array([y_true.size - 1])
    threshold_indices = np.hstack((distinct_indices, end))

    thresholds = y_score[threshold_indices]
    tps = np.cumsum(y_true)[threshold_indices]

    # (1 + threshold_indices) = the number of positives
    # at each index, thus number of data points minus true
    # positives = false positives
    fps = (1 + threshold_indices) - tps
    return tps, fps, thresholds


def _roc_auc_score(y_true, y_score):
    """
    Compute Area Under the Curve (AUC) from prediction scores
    Parameters
    ----------
    y_true : 1d ndarray, shape = [n_samples]
        True targets/labels of binary classification
    y_score : 1d ndarray, shape = [n_samples]
        Estimated probabilities or scores
    Returns
    -------
    auc : float
    """

    # ensure the target is binary
    if np.unique(y_true).size != 2:
        raise ValueError('Only two class should be present in y_true. ROC AUC score '
                         'is not defined in that case.')

    tps, fps, _ = _binary_clf_curve(y_true, y_score)

    # convert count to rate
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # compute AUC using the trapezoidal rule;
    # appending an extra 0 is just to ensure the length matches
    zero = np.array([0])
    tpr_diff = np.hstack((np.diff(tpr), zero))
    fpr_diff = np.hstack((np.diff(fpr), zero))
    auc = np.dot(tpr, fpr_diff) + np.dot(tpr_diff, fpr_diff) / 2
    return auc

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Input with biparametric MRI
    if args.data_loader == 'bp_seg':
        test_loader = get_loader_bp_seg(args)
    # Input with biparametric and RSI MRI
    elif args.data_loader == '2c_bp_seg':
        test_loader = get_loader_2c_bp_seg(args)
    
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    if args.model_name == 'dense121_plus_1':
        model = DenseNet121_plus_1(spatial_dims=3, in_channels=args.in_channels, out_channels=2).to(device)
    else:
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=args.in_channels, out_channels=2).to(device)

    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)
    post_pred = Compose([Activations(softmax=True)])


    with torch.no_grad():
        GT_list = np.array([])
        pred_onehot_list = np.array([])
        pred_prob_list = np.array([])
        PDS_list = np.array([])
        date_list = np.array([])
        for i, batch in enumerate(test_loader):
            T2 = batch["image"].cuda()
            # c0, c1, c2, c3 = batch["c0"].cuda(), batch["c1"].cuda(), batch["c2"].cuda(), batch["c3"].cuda()
            # image_ = image[:,:2,:,:,:]
            target= batch["label"]
            rsi_max, psad = batch["RSIrs_max"] / 1000.0, batch["PSAD"]
            rsi_max = MetaTensor(rsi_max)
            psad = MetaTensor(rsi_max)
            rsi_max, psad = rsi_max.cuda(), psad.cuda()
            rsi_max = torch.unsqueeze(rsi_max, 1)
            psad = torch.unsqueeze(psad, 1)
            file_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            # l = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")[1]
            # num = num.split(".")[0]
            img_name = file_name.split("_")[0]
            img_date = file_name.split("_")[1]
            # print(img_date)
            PDS_list = np.append(PDS_list, img_name[3:])
            date_list = np.append(date_list, img_date)
            if args.data_loader == 'bp_seg':
                adc, dwi = batch["adc"].cuda(), batch["dwi"].cuda()
                seg = batch["seg"]
                seg = torch.squeeze(seg, 1)
                # print(seg.size())
                seg_dil = dilation(seg)
                seg_dil = seg_dil.cuda()
                T2 = einsum("bkwhz,bkwhz->bkwhz", T2, seg_dil)
                adc = einsum("bkwhz,bkwhz->bkwhz", adc, seg_dil)
                dwi = einsum("bkwhz,bkwhz->bkwhz", dwi, seg_dil)
                inputs = torch.cat((T2,adc,dwi),1)
            elif args.data_loader == '2c_bp_seg':
                adc, dwi = batch["adc"].cuda(), batch["dwi"].cuda()
                c0, c1 = batch["c0"].cuda(), batch["c1"].cuda()
                seg = batch["seg"]
                seg = torch.squeeze(seg, 1)
                seg_dil = dilation(seg)
                seg_dil = seg_dil.cuda()
                T2 = einsum("bkwhz,bkwhz->bkwhz", T2, seg_dil)
                adc = einsum("bkwhz,bkwhz->bkwhz", adc, seg_dil)
                dwi = einsum("bkwhz,bkwhz->bkwhz", dwi, seg_dil)
                c0 = einsum("bkwhz,bkwhz->bkwhz", c0, seg_dil)
                c1 = einsum("bkwhz,bkwhz->bkwhz", c1, seg_dil)
                inputs = torch.cat((T2,adc,dwi,c0,c1),1)
            
            print("Inference on case {}".format(img_name))
            if args.model_name == 'dense121_plus':
                label = model(inputs, rsi_max, psad)
            elif args.model_name == 'dense121_plus_1':
                label = model(inputs, rsi_max)
            else:
                label = model(inputs)
            acc_value = label.argmax(dim=1)
            label = post_pred(label[0])
            label = label.detach().cpu().numpy()
            acc_value = acc_value.detach().cpu().numpy()

            GT_list = np.append(GT_list,np.float64(target[0]))
            pred_onehot_list = np.append(pred_onehot_list,np.float64(acc_value[0]))
            pred_prob_list = np.append(pred_prob_list,np.float64(label[1]))

            print('Probability of lesion or not:')
            print(label)
            print('GT is: ' + str(target[0]) + ' Pred is: ' + str(acc_value[0]))
        
        print("Probability AUC:" + str(_roc_auc_score(GT_list, pred_prob_list)))
        print("One hot AUC:" + str(_roc_auc_score(GT_list, pred_onehot_list)))
        All = np.zeros(GT_list.size, dtype=[('PDS_id', 'U6'), ('date', 'U8'),('GT', float),('Pred', float)]);
        All['PDS_id'] = PDS_list
        All['date'] = date_list
        All['GT'] = GT_list
        All['Pred'] = pred_prob_list
        file_name = 'evals/data/' + args.data_loader + '_' + args.model_name + str(val_list[0]) + '.csv'
        np.savetxt(file_name, All, delimiter=",", fmt="%s, %s, %.7f, %.7f", header="PDS_id, date, GT, Pred", comments="")
        
        print("Finished inference!")


if __name__ == "__main__":
    main()
