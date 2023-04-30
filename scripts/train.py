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
from datasets.algonauts_2023 import AlgonautsDataset
from scipy.stats import pearsonr as corr


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, outputs, batch):
        pred_lh_fmri = outputs['lh_fmri']
        pred_rh_fmri = outputs['rh_fmri']
        gt_lh_fmri = batch['lh_fmri']
        gt_rh_fmri = batch['rh_fmri']

        loss = self.l1_loss(pred_lh_fmri, gt_lh_fmri) + self.l1_loss(pred_rh_fmri, gt_rh_fmri)
        return loss


class Metric:
    def __init__(self, args):
        self.args = args

    def __call__(self, outputs, batch):
        pred_lh_fmri = outputs['lh_fmri']
        pred_rh_fmri = outputs['rh_fmri']
        gt_lh_fmri = batch['lh_fmri']
        gt_rh_fmri = batch['rh_fmri']

        # Empty correlation array of shape: (LH vertices)
        lh_correlation = np.zeros(pred_lh_fmri.shape[1])
        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in range(pred_lh_fmri.shape[1]):
            lh_correlation[v] = corr(pred_lh_fmri[:,v], gt_lh_fmri[:,v])[0]

        # Empty correlation array of shape: (RH vertices)
        rh_correlation = np.zeros(pred_rh_fmri.shape[1])
        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in range(pred_rh_fmri.shape[1]):
            rh_correlation[v] = corr(pred_rh_fmri[:,v], gt_rh_fmri[:,v])[0]

        avg = lh_correlation.mean() + rh_correlation.mean()
        return avg


def get_model(args, distributed=True):
    from models.timm_model import AlgonautsTimm
    model = AlgonautsTimm(args)

    # move networks to gpu
    model = model.cuda()
    if args.use_ema:
        model_ema = timm.utils.ModelEmaV2(model, decay=args.ema_decay)
    else:
        model_ema = None

    # synchronize batch norms (if any)
    if distributed:
        if utils.has_batchnorms(model):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
    else:
        model = nn.DataParallel(model)

    print(model)

    return model, model_ema


def get_dataloader(args):
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = AlgonautsDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        transform=train_transform,
        fold=args.fold,
        num_folds=args.num_folds,
        is_train=True
    )

    valid_dataset = AlgonautsDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        transform=train_transform,
        fold=args.fold,
        num_folds=args.num_folds,
        is_train=False
    )

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if args.distributed
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )


    valid_sampler = (
        torch.utils.data.distributed.DistributedSampler(valid_dataset)
        if args.distributed
        else None
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=valid_sampler
    )

    return train_loader, valid_loader


def train(args):

    # args.distributed = True

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.distributed:
        utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_loader, valid_loader = get_dataloader(args)

    # ============ building Clusformer ... ============
    model, model_ema = get_model(args, args.distributed)

    # ============ preparing loss ... ============
    criterion = Criterion(args)

    # ============ preparing optimizer and scheduler ... ============
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.min_lr
        )
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLRWithWarmup(
            optimizer=optimizer,
            num_steps=total_steps,
            lr_range=(args.lr, args.lr / 10),
            warmup_fraction=0.1,
            init_lr=args.lr / 10,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=2, factor=0.1
        )

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "best_score": np.inf}
    is_save_best = False
    if os.path.isfile(args.resume):
        utils.restart_from_checkpoint(
            args.resume,
            model=model,
            run_variables=to_restore,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            scheduler=scheduler,
            criterion=criterion,
        )

    start_epoch = to_restore["epoch"]
    best_score = to_restore["best_score"]

    # checkpoint storage
    start_time = time.time()
    print("Starting training !")

    patient_counter = 0
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # ============ training one epoch ... ============
        train_stats = train_one_epoch(
            model,
            model_ema,
            criterion,
            train_loader,
            optimizer,
            scheduler,
            epoch,
            fp16_scaler,
            True,
            args,
        )

        # Distributed bn
        if args.distributed:
            timm.utils.distribute_bn(model, torch.distributed.get_world_size(), True)

        # ============ validate one epoch ... ============
        if valid_loader is not None:
            valid_stats = train_one_epoch(
                model,
                None,
                criterion,
                valid_loader,
                optimizer,
                scheduler,
                epoch,
                fp16_scaler,
                False,
                args,
            )
        else:
            valid_stats = train_stats

        if model_ema is not None:
            if args.distributed:
                timm.utils.distribute_bn(
                    model_ema, torch.distributed.get_world_size(), True
                )
            ema_valid_stats = train_one_epoch(
                model_ema.module,
                None,
                criterion,
                valid_loader,
                optimizer,
                scheduler,
                epoch,
                fp16_scaler,
                False,
                args,
            )

            current_score = min(valid_stats["l1"], ema_valid_stats["l1"])
        else:
            current_score = valid_stats["l1"]

        if current_score < best_score:
            best_score = current_score
            is_save_best = True
            patient_counter = 0
        else:
            is_save_best = False
            patient_counter += 1

        valid_stats["best_score"] = best_score

        # ============ writing logs ... ============
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "criterion": criterion.state_dict(),
            "best_score": best_score,
        }

        if model_ema is not None:
            save_dict["ema"] = model_ema.state_dict()

        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()

        if is_save_best:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, "best.pth"))

        # utils.save_on_master(save_dict, os.path.join(args.output_dir, "last.pth"))

        log_train_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        log_valid_stats = {
            **{f"valid_{k}": v for k, v in valid_stats.items()},
            "epoch": epoch,
        }
        if not args.distributed or utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_train_stats) + "\n")
                f.write(json.dumps(log_valid_stats) + "\n")

        # if best_score == 100:  # (args.subject != 0 and patient_counter == 15):
        #     break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


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

    metric_fn = Metric(args)
    if is_train:
        model.train()
        prefix = "TRAIN"
    else:
        model.eval()
        prefix = "VALID"
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

            metric_dict = metric_fn(outputs, batch)
            # metric_dict = {
            #     'l1': loss
            # }

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
        for k, v in metric_dict.items():
            metric_logger.update(**{k: v})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    if args.distributed:
        metric_logger.synchronize_between_processes()
    print(f"[{prefix}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train algonauts", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
