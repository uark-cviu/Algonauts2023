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
from datasets.algonauts_2023 import AlgonautsTestDataset
from scipy.stats import pearsonr as corr
from tqdm import tqdm

"""
command:
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir logs/stage1_lr/subj01/seresnextaa101d_32x8d_l,r --output_dir predictions/stage1_lr_seresnextaa101d_32x8d/ && \
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir logs/stage1_lr/subj02/seresnextaa101d_32x8d_l,r --output_dir predictions/stage1_lr_seresnextaa101d_32x8d/ && \
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir logs/stage1_lr/subj03/seresnextaa101d_32x8d_l,r --output_dir predictions/stage1_lr_seresnextaa101d_32x8d/ && \
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir logs/stage1_lr/subj04/seresnextaa101d_32x8d_l,r --output_dir predictions/stage1_lr_seresnextaa101d_32x8d/ && \
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir logs/stage1_lr/subj05/seresnextaa101d_32x8d_l,r --output_dir predictions/stage1_lr_seresnextaa101d_32x8d/ && \
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir logs/stage1_lr/subj06/seresnextaa101d_32x8d_l,r --output_dir predictions/stage1_lr_seresnextaa101d_32x8d/ && \
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir logs/stage1_lr/subj07/seresnextaa101d_32x8d_l,r --output_dir predictions/stage1_lr_seresnextaa101d_32x8d/ && \
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir logs/stage1_lr/subj08/seresnextaa101d_32x8d_l,r --output_dir predictions/stage1_lr_seresnextaa101d_32x8d/
"""


def get_args_parser():
    parser = argparse.ArgumentParser("Test", add_help=False)

    # dataset parameters
    parser.add_argument("--folds", default="0,1,2,3,4", type=str)
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="predictions")
    return parser


def get_model(args):
    from models.timm_model import AlgonautsTimm

    model = AlgonautsTimm(args)

    # move networks to gpu
    model = model.cuda()
    # if args.use_ema:
    #     model_ema = timm.utils.ModelEmaV2(model, decay=args.ema_decay)
    # else:
    #     model_ema = None

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
                # counter_r[roi_idx[np.where(pred_r[roi_idx] != 0)[0]]] += 1
                pred_r[:, np.where(roi_idx)[0]] += pred.detach().cpu().numpy()

    # import pdb; pdb.set_trace()
    counter_l[np.where(counter_l == 0)] = 1
    counter_r[np.where(counter_r == 0)] = 1

    pred_l = pred_l / counter_l
    pred_r = pred_r / counter_r
    return pred_l, pred_r


def train(args):
    cudnn.benchmark = True

    folds = args.folds
    folds = folds.split(",")
    # ============ preparing data ... ============
    # train_loader, valid_loader = get_dataloader(args)

    data_loader = None

    pred_rh_final = 0
    pred_lh_final = 0

    for fold in folds:
        checkpoint = f"{args.checkpoint_dir}/{fold}/best.pth"
        checkpoint = torch.load(checkpoint)
        train_args = checkpoint["args"]
        print(train_args)
        best_score = checkpoint["best_score"]
        subject_id = train_args.data_dir.split("/")[-1]
        print(f"[+] Predicting {fold} of {subject_id}, best score: {best_score}")

        if data_loader is None:
            data_loader = get_dataloader(train_args)

        # ============ building Clusformer ... ============
        model = get_model(train_args)
        model.load_state_dict(checkpoint['model'])
        # model.load_state_dict(checkpoint["ema"])
        model.eval()

        pred_rh_fmris = []
        pred_lh_fmris = []

        for batch in tqdm(data_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = batch[k].cuda(non_blocking=True)

            with torch.no_grad():
                outputs = model(batch)

                pred_lh_fmri, pred_rh_fmri = post_process_output(outputs, train_args)

                # pred_lh_fmri = outputs["lh_fmri"].detach().cpu().numpy()
                # pred_rh_fmri = outputs["rh_fmri"].detach().cpu().numpy()

                pred_lh_fmris.append(pred_lh_fmri)
                pred_rh_fmris.append(pred_rh_fmri)

        pred_rh_fmris = np.concatenate(pred_rh_fmris)
        pred_lh_fmris = np.concatenate(pred_lh_fmris)

        pred_rh_final += pred_rh_fmris / len(folds)
        pred_lh_final += pred_lh_fmris / len(folds)

    pred_lh_final = pred_lh_final.astype(np.float32)
    pred_rh_final = pred_rh_final.astype(np.float32)

    print(pred_rh_final.min(), pred_rh_final.max())
    print(pred_lh_final.min(), pred_lh_final.max())
    output_dir = args.output_dir
    output_dir = f"{output_dir}/{subject_id}"
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/lh_pred_test.npy", pred_lh_final)
    np.save(f"{output_dir}/rh_pred_test.npy", pred_rh_final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test algonauts", parents=[get_args_parser()])
    args = parser.parse_args()
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
