import sys

sys.path.append(".")  # noqa

import argparse
import datetime
import json
import math
import os
from pathlib import Path
from functools import partial
import time

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from schedulers import OneCycleLRWithWarmup
from utils import dino as utils
from utils.algonauts_parser import get_args_parser
from datasets.algonauts_2023 import AlgonautsTestDataset, AlgonautsDataset
from scipy.stats import pearsonr as corr
from tqdm import tqdm
import pickle

"""
command:
python scripts/get_oof.py --checkpoint_dir logs/finetune_onecycle --model_name seresnext101d_32x8d --output_dir oof/ && \
"""


def get_args_parser():
    parser = argparse.ArgumentParser("Test", add_help=False)

    # dataset parameters
    parser.add_argument("--folds", default="0,1,2,3,4", type=str)
    parser.add_argument("--model_name", default="seresnext101d_32x8d", type=str)
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="predictions")
    return parser


def get_model(args):
    from models.timm_model import AlgonautsTimm

    model = AlgonautsTimm(args)
    model = model.cuda()
    model = nn.DataParallel(model)

    return model


def get_dataloader(args):
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_dataset = AlgonautsTestDataset(
        data_dir=args.data_dir,
        transform=train_transform,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return valid_loader


def get_valid_dataloader(args):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    valid_dataset = AlgonautsDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        transform=transform,
        fold=args.fold,
        num_folds=args.num_folds,
        is_train=False,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return valid_loader


def post_process_output(outputs, args):
    subject_metadata = args.subject_metadata
    pred_l, pred_r = None, None
    counter_l, counter_r = None, None
    for side in ["l", "r"]:
        roi_names = outputs[side].keys()
        for roi_name in roi_names:
            pred = outputs[side][roi_name]
            roi_idx = subject_metadata[side][roi_name]

            batch_size = pred.shape[0]

            if side == "l":
                if pred_l is None:
                    pred_l = np.zeros((batch_size, args.num_lh_output))
                    counter_l = np.zeros((batch_size, args.num_lh_output))
                # counter_l[roi_idx[np.where(pred_l[roi_idx] != 0)[0]]] += 1
                counter_l += roi_idx
                pred_l[:, np.where(roi_idx)[0]] += pred.detach().cpu().numpy()
            else:
                if pred_r is None:
                    pred_r = np.zeros((batch_size, args.num_rh_output))
                    counter_r = np.zeros((batch_size, args.num_rh_output))

                counter_r += roi_idx
                pred_r[:, np.where(roi_idx)[0]] += pred.detach().cpu().numpy()

    counter_l[np.where(counter_l == 0)] = 1
    counter_r[np.where(counter_r == 0)] = 1

    pred_l = pred_l / counter_l
    pred_r = pred_r / counter_r
    return pred_l, pred_r


def train(args):
    cudnn.benchmark = True

    folds = args.folds
    folds = folds.split(",")

    for subject_id in [
        "subj01",
        "subj02",
        "subj03",
        "subj04",
        "subj05",
        "subj06",
        "subj07",
        "subj08",
    ]:
        for fold in folds:
            checkpoint = (
                f"{args.checkpoint_dir}/{subject_id}/{args.model_name}/{fold}/best.pth"
            )
            checkpoint = torch.load(checkpoint)
            train_args = checkpoint["args"]
            print(train_args)
            best_score = checkpoint["best_score"]
            subject_id = train_args.data_dir.split("/")[-1]
            print(f"[+] Predicting {fold} of {subject_id}, best score: {best_score}")

            test_data_loader = get_dataloader(train_args)
            valid_data_loader = get_valid_dataloader(train_args)

            # ============ building Clusformer ... ============
            model = get_model(train_args)
            model.load_state_dict(checkpoint["model"])
            model.eval()

            data_dict = {"valid": {}, "test": {}, "valid_gt": {}}

            pred_rh_fmris = []
            pred_lh_fmris = []

            for batch in tqdm(test_data_loader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = batch[k].cuda(non_blocking=True)

                with torch.no_grad():
                    outputs = model(batch)

                    pred_lh_fmri, pred_rh_fmri = post_process_output(
                        outputs, train_args
                    )

                    pred_lh_fmris.append(pred_lh_fmri)
                    pred_rh_fmris.append(pred_rh_fmri)

            pred_rh_fmris = np.concatenate(pred_rh_fmris)
            pred_lh_fmris = np.concatenate(pred_lh_fmris)

            data_dict["test"]["l"] = pred_lh_fmris
            data_dict["test"]["r"] = pred_rh_fmris

            # validation

            pred_rh_fmris = []
            pred_lh_fmris = []

            gt_lh_fmris = []
            gt_rh_fmris = []

            for batch in tqdm(valid_data_loader):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = batch[k].cuda(non_blocking=True)

                with torch.no_grad():
                    outputs = model(batch)

                    pred_lh_fmri, pred_rh_fmri = post_process_output(
                        outputs, train_args
                    )

                    gt_lh_fmri = batch["l"].detach().cpu().numpy()
                    gt_rh_fmri = batch["r"].detach().cpu().numpy()

                    pred_lh_fmris.append(pred_lh_fmri)
                    pred_rh_fmris.append(pred_rh_fmri)

                    gt_lh_fmris.append(gt_lh_fmri)
                    gt_rh_fmris.append(gt_rh_fmri)

            pred_rh_fmris = np.concatenate(pred_rh_fmris)
            pred_lh_fmris = np.concatenate(pred_lh_fmris)

            gt_lh_fmris = np.concatenate(gt_lh_fmris)
            gt_rh_fmris = np.concatenate(gt_rh_fmris)

            data_dict["valid"]["l"] = pred_lh_fmris
            data_dict["valid"]["r"] = pred_rh_fmris

            data_dict["valid_gt"]["l"] = gt_lh_fmris
            data_dict["valid_gt"]["r"] = gt_rh_fmris

            output_dir = args.output_dir
            output_dir = f"{output_dir}/{args.model_name}/{subject_id}"
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/fold_{fold}.pkl", "wb") as f:
                pickle.dump(data_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test algonauts", parents=[get_args_parser()])
    args = parser.parse_args()
    train(args)
