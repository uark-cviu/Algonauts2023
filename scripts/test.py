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



def get_args_parser():
    parser = argparse.ArgumentParser('Test', add_help=False)

    # dataset parameters
    parser.add_argument('--folds', default='0,1,2,3,4', type=str)
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='predictions')
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
        drop_last=False
    )

    return valid_loader


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
        train_args = checkpoint['args']
        print(train_args)
        best_score = checkpoint['best_score']
        subject_id = train_args.data_dir.split("/")[-1]
        print(f"[+] Predicting {fold} of {subject_id}, best score: {best_score}")


        if data_loader is None:
            data_loader = get_dataloader(train_args)

        # ============ building Clusformer ... ============
        model = get_model(train_args)
        # model.load_state_dict(checkpoint['model'])
        model.load_state_dict(checkpoint['ema'])
        model.eval()

        pred_rh_fmris = []
        pred_lh_fmris = []

        for batch in tqdm(data_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = batch[k].cuda(non_blocking=True)

            with torch.no_grad():
                outputs = model(batch)

                pred_lh_fmri = outputs['lh_fmri'].detach().cpu().numpy()
                pred_rh_fmri = outputs['rh_fmri'].detach().cpu().numpy()

                pred_lh_fmris.append(pred_lh_fmri)
                pred_rh_fmris.append(pred_rh_fmri)

        pred_rh_fmris = np.concatenate(pred_rh_fmris)
        pred_lh_fmris = np.concatenate(pred_lh_fmris)

        pred_rh_final += pred_rh_fmris / len(folds)
        pred_lh_final += pred_lh_fmris / len(folds)

    print(pred_rh_final.min(), pred_rh_final.max())
    print(pred_lh_final.min(), pred_lh_final.max())
    output_dir = args.output_dir
    output_dir = f"{output_dir}/{subject_id}"
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/lh_pred_test.npy", pred_lh_final)
    np.save(f"{output_dir}/rh_pred_test.npy", pred_rh_final)


def train_one_epoch(
    model,
    model_ema,
    criterion,
    data_loader,
    optimizer,
    scheduler,
    epoch,
    fp16_scaler,
    is_train,
    args,
):
    if is_train:
        model.train()
        prefix = "TRAIN"
    else:
        model.eval()
        prefix = "VALID"
        metric_fn = Metric(args)
        pred_lh_fmris = []
        pred_rh_fmris = []
        gt_lh_fmris = []
        gt_rh_fmris = []
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    for batch in metric_logger.log_every(data_loader, 50, header):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = batch[k].cuda(non_blocking=True)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if not is_train:
                with torch.no_grad():
                    outputs = model(batch)
            else:
                outputs = model(batch)

            loss = criterion(outputs, batch)
            if not args.distributed:
                loss = loss.mean()


            if not is_train:
                pred_lh_fmri = outputs['lh_fmri'].detach().cpu().numpy()
                pred_rh_fmri = outputs['rh_fmri'].detach().cpu().numpy()
                gt_lh_fmri = batch['lh_fmri'].detach().cpu().numpy()
                gt_rh_fmri = batch['rh_fmri'].detach().cpu().numpy()

                pred_lh_fmris.append(pred_lh_fmri)
                pred_rh_fmris.append(pred_rh_fmri)
                gt_lh_fmris.append(gt_lh_fmri)
                gt_rh_fmris.append(gt_rh_fmri)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        if is_train:
            # student update
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if model_ema is not None:
                model_ema.update(model)

            scheduler.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        # for k, v in metric_dict.items():
        #     metric_logger.update(**{k: v})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    if not is_train:

                # pred_lh_fmris.append(pred_lh_fmri)
                # pred_rh_fmris.append(pred_rh_fmri)
                # gt_lh_fmris.append(gt_lh_fmri)
                # gt_rh_fmris.append(gt_rh_fmri)
        pred_lh_fmris = np.concatenate(pred_lh_fmris, axis=0)
        pred_rh_fmris = np.concatenate(pred_rh_fmris, axis=0)
        gt_lh_fmris = np.concatenate(gt_lh_fmris, axis=0)
        gt_rh_fmris = np.concatenate(gt_rh_fmris, axis=0)
        metric_dict = metric_fn(pred_lh_fmris, pred_rh_fmris, gt_lh_fmris, gt_rh_fmris)
        for k, v in metric_dict.items():
            metric_logger.update(**{k: v})
    # gather the stats from all processes
    if args.distributed:
        metric_logger.synchronize_between_processes()
    print(f"[{prefix}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test algonauts", parents=[get_args_parser()])
    args = parser.parse_args()
    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)

