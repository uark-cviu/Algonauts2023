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
from datasets.algonauts_coco import AlgonautsCOCODataset
from scipy.stats import pearsonr as corr
from criterions.pcc import PCCLoss
import robust_loss_pytorch
import pickle


def mixup_data(batch, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    x = batch["image"]

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    # y_a, y_b = y, y[index]
    batch["image"] = mixed_x
    batch["index"] = index
    batch["lam"] = lam
    return batch


# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def vox_loss_fn(r, v, nu=0.5, delta=1.0):
    # err = T.sum(huber(r, v, delta), dim=0)
    err = torch.sum((r - v) ** 2, dim=0)
    # squared correlation coefficient with 'leak'
    cr = r - torch.mean(r, dim=0, keepdim=True)
    cv = v - torch.mean(v, dim=0, keepdim=True)
    wgt = torch.clamp(
        torch.pow(torch.mean(cr * cv, dim=0), 2)
        / ((torch.mean(cr**2, dim=0)) * (torch.mean(cv**2, dim=0)) + 1e-6),
        min=nu,
        max=1,
    ).detach()

    weighted_err = wgt * err  # error per voxel
    loss = torch.sum(weighted_err) / torch.mean(wgt)
    return loss


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.subject_metadata = args.subject_metadata
        self.nc_lh = args.nc_lh
        self.nc_rh = args.nc_rh
        self.l1_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()
        self.pcc = PCCLoss()

        self.adaptive_loss_dict = {}
        for side in ["l", "r"]:
            self.adaptive_loss_dict[side] = {}
            roi_names = self.subject_metadata[side].keys()
            for roi_name in roi_names:
                num_output = self.subject_metadata[side][roi_name].sum()
                loss_fn = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
                    num_dims=num_output,
                    float_dtype=np.float32,
                    device=f"cuda:{args.gpu}",
                )
                # loss_fn = self.l1_loss
                self.adaptive_loss_dict[side][roi_name] = loss_fn

        # self.adaptive_lh = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        #     num_dims = args.num_lh_output, float_dtype=np.float32, device='cuda:0'
        # )

        # self.adaptive_rh = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        #     num_dims = args.num_rh_output, float_dtype=np.float32, device='cuda:0'
        # )

    def get_gt_roi(self, gt_fmri, side, roi_name):
        roi_idx = self.subject_metadata[side][roi_name]
        return gt_fmri[:, np.where(roi_idx)[0]]

    def get_nc_roi(self, side, roi_name):
        roi_idx = self.subject_metadata[side][roi_name]
        if side == "l":
            nc = self.nc_lh
        else:
            nc = self.nc_rh
        return nc[np.where(roi_idx)[0]]

    def loss(self, pred, gt, side=None, roi_name=None, nc=None):
        pcc_loss = self.pcc(pred, gt, nc)
        l1_loss = self.l1_loss(pred, gt)
        # adaptive_loss = self.adaptive_loss(pred, gt, side, roi_name)
        return pcc_loss + l1_loss

        # return vox_loss_fn(pred, gt)
        # adaptive_loss = self.adaptive_loss(pred, gt, side, roi_name)
        # return pcc_loss + l1_loss

    def adaptive_loss(self, pred, gt, side, roi_name):
        loss_fn = self.adaptive_loss_dict[side][roi_name]
        return torch.mean(loss_fn.lossfun((pred - gt)))

    def forward(self, outputs, batch):
        total_loss = 0
        count = 0
        if "lam" in batch:
            mixup = True
            # print("Mix")
        else:
            mixup = False

        for side in ["l", "r"]:
            # GT
            gt_fmri = batch[side]
            if mixup:
                index = batch["index"]
                lam = batch["lam"]
                gt_fmri_mixed = gt_fmri[index]

            pred_all = outputs[side + "_all"]
            gt_all = gt_fmri

            loss_all = self.loss(pred_all, gt_all, None, None)
            total_loss += loss_all
            count += 1

            roi_names = outputs[side].keys()
            for roi_name in roi_names:
                if "embedding" in roi_name:
                    continue

                pred = outputs[side][roi_name]
                gt = self.get_gt_roi(gt_fmri, side, roi_name)
                nc = self.get_nc_roi(side, roi_name)
                loss = self.loss(pred, gt, side, roi_name, nc)

                # embedding = outputs[side][roi_name + "_embedding"]
                # embedding = torch.nn.functional.normalize(embedding)
                # pred_corr = embedding @ embedding.T
                # gt_corr = torch.corrcoef(gt).detach()
                # loss_corr = (gt_corr - pred_corr) ** 2
                # loss_corr = loss_corr.mean()
                # loss += loss_corr

                if mixup:
                    gt_mix = self.get_gt_roi(gt_fmri_mixed, side, roi_name)
                    loss_mix = self.loss(pred, gt_mix, side, roi_name)

                    total_loss += lam * loss + (1 - lam) * loss_mix
                else:
                    total_loss += loss
                count += 1

        return total_loss / count


class Metric:
    def __init__(self, args):
        # from torchmetrics import PearsonCorrCoef
        # selpearson = PearsonCorrCoef()
        self.args = args
        # Load the ROI classes mapping dictionaries
        roi_mapping_files = [
            "mapping_prf-visualrois.npy",
            "mapping_floc-bodies.npy",
            "mapping_floc-faces.npy",
            "mapping_floc-places.npy",
            "mapping_floc-words.npy",
            "mapping_streams.npy",
        ]
        self.roi_name_maps = []
        for r in roi_mapping_files:
            self.roi_name_maps.append(
                np.load(
                    os.path.join(args.data_dir, "roi_masks", r), allow_pickle=True
                ).item()
            )

        # Load the ROI brain surface maps
        lh_challenge_roi_files = [
            "lh.prf-visualrois_challenge_space.npy",
            "lh.floc-bodies_challenge_space.npy",
            "lh.floc-faces_challenge_space.npy",
            "lh.floc-places_challenge_space.npy",
            "lh.floc-words_challenge_space.npy",
            "lh.streams_challenge_space.npy",
        ]
        rh_challenge_roi_files = [
            "rh.prf-visualrois_challenge_space.npy",
            "rh.floc-bodies_challenge_space.npy",
            "rh.floc-faces_challenge_space.npy",
            "rh.floc-places_challenge_space.npy",
            "rh.floc-words_challenge_space.npy",
            "rh.streams_challenge_space.npy",
        ]
        self.lh_challenge_rois = []
        self.rh_challenge_rois = []
        for r in range(len(lh_challenge_roi_files)):
            self.lh_challenge_rois.append(
                np.load(
                    os.path.join(args.data_dir, "roi_masks", lh_challenge_roi_files[r])
                )
            )
            self.rh_challenge_rois.append(
                np.load(
                    os.path.join(args.data_dir, "roi_masks", rh_challenge_roi_files[r])
                )
            )

    def __call__(self, pred_lh_fmri, pred_rh_fmri, gt_lh_fmri, gt_rh_fmri):
        # Empty correlation array of shape: (LH vertices)
        lh_correlation = np.zeros(pred_lh_fmri.shape[1])
        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in range(pred_lh_fmri.shape[1]):
            lh_correlation[v] = corr(pred_lh_fmri[:, v], gt_lh_fmri[:, v])[0]

        # Empty correlation array of shape: (RH vertices)
        rh_correlation = np.zeros(pred_rh_fmri.shape[1])
        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in range(pred_rh_fmri.shape[1]):
            rh_correlation[v] = corr(pred_rh_fmri[:, v], gt_rh_fmri[:, v])[0]

        # Select the correlation results vertices of each ROI
        roi_names = []
        lh_roi_correlation = []
        rh_roi_correlation = []
        for r1 in range(len(self.lh_challenge_rois)):
            for r2 in self.roi_name_maps[r1].items():
                if (
                    r2[0] != 0
                ):  # zeros indicate to vertices falling outside the ROI of interest
                    roi_names.append(r2[1])
                    lh_roi_idx = np.where(self.lh_challenge_rois[r1] == r2[0])[0]
                    rh_roi_idx = np.where(self.rh_challenge_rois[r1] == r2[0])[0]
                    lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                    rh_roi_correlation.append(rh_correlation[rh_roi_idx])
        # roi_names.append("All vertices")
        # lh_roi_correlation.append(lh_correlation)
        # rh_roi_correlation.append(rh_correlation)

        # Create the plot
        lh_median_roi_correlation = [
            np.mean(lh_roi_correlation[r]) for r in range(len(lh_roi_correlation))
        ]
        rh_median_roi_correlation = [
            np.mean(rh_roi_correlation[r]) for r in range(len(rh_roi_correlation))
        ]

        lh_median_roi_correlation = np.array(lh_median_roi_correlation)
        rh_median_roi_correlation = np.array(rh_median_roi_correlation)

        lh_median_roi_correlation = lh_median_roi_correlation[
            ~np.isnan(lh_median_roi_correlation)
        ]
        rh_median_roi_correlation = rh_median_roi_correlation[
            ~np.isnan(rh_median_roi_correlation)
        ]

        avg = (lh_median_roi_correlation.mean() + rh_median_roi_correlation.mean()) / 2

        return {"corr": avg}


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

    # print(model)

    return model, model_ema


def get_dataloader(args):
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            # transforms.RandomResizedCrop(args.img_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            # transforms.Resize(256),
            # transforms.CenterCrop(args.img_size),
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
        is_train=True,
    )

    args.num_lh_output = train_dataset.num_lh_output
    args.num_rh_output = train_dataset.num_rh_output

    args.min_max_lh = train_dataset.min_max_lh
    args.min_max_rh = train_dataset.min_max_rh

    if args.pseudo_dir != "none":
        print("[+] Pseudo Training")
        from datasets.algonauts_2023 import AlgonautsPseudoDataset

        pseudo_dataset = AlgonautsPseudoDataset(
            data_dir=args.data_dir,
            transform=train_transform,
            pseudo_dir=args.pseudo_dir,
        )

        train_dataset = torch.utils.data.ConcatDataset([train_dataset, pseudo_dataset])

    valid_dataset = AlgonautsDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        transform=valid_transform,
        fold=args.fold,
        num_folds=args.num_folds,
        is_train=False,
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
        sampler=train_sampler,
        drop_last=True,
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
        sampler=valid_sampler,
        drop_last=True,
    )

    return train_loader, valid_loader


def train_one_fold(args):
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
    to_restore = {"epoch": 0, "best_score": -np.inf}
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

        # if epoch <= 2:
        #     model.module.freeze_backbone()
        # else:
        #     model.module.unfreeze_backbone()

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

            current_score = max(valid_stats["corr"], ema_valid_stats["corr"])
        else:
            current_score = valid_stats["corr"]

        if current_score > best_score:
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

        if patient_counter == 3:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


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

    output_dir = args.output_dir
    pretrained = args.pretrained

    for fold in [0, 1, 2, 3, 4]:
        print("training fold ", fold)
        args.fold = fold
        args.output_dir = f"{output_dir}/{fold}/"
        args.pretrained = f"{pretrained}/best.pth"
        os.makedirs(args.output_dir, exist_ok=True)
        train_one_fold(args)


def post_process_output(outputs, args):
    subject_metadata = args.subject_metadata
    pred_l, pred_r = None, None
    counter_l, counter_r = None, None
    pred_l_all, pred_r_all = None, None
    for side in ["l", "r"]:
        pred_all = outputs[side + "_all"]
        roi_names = outputs[side].keys()
        for roi_name in roi_names:
            if "_embedding" in roi_name:
                continue
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

        if side == "l":
            pred_l_all = pred_all.detach().cpu().numpy()
        else:
            pred_r_all = pred_all.detach().cpu().numpy()

    # import pdb; pdb.set_trace()
    counter_l[np.where(counter_l == 0)] = 1
    counter_r[np.where(counter_r == 0)] = 1

    pred_l = pred_l / counter_l
    pred_r = pred_r / counter_r

    pred_l = (pred_l + pred_l_all) / 2
    pred_r = (pred_r + pred_r_all) / 2
    return pred_l, pred_r


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

        # if is_train and np.random.rand() < 0.5:
        #     # import pdb; pdb.set_trace()
        #     batch = mixup_data(batch, alpha=1.0, use_cuda=True)

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
                # if True:
                pred_lh_fmri, pred_rh_fmri = post_process_output(outputs, args)
                gt_lh_fmri = batch["l"].detach().cpu().numpy()
                gt_rh_fmri = batch["r"].detach().cpu().numpy()

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
    parser = argparse.ArgumentParser("Train algonauts", parents=[get_args_parser()])
    args = parser.parse_args()
    with open("subject_meta.pkl", "rb") as f:
        subject_metadata = pickle.load(f)
        args.subject_metadata = subject_metadata[args.subject]

    nc_lh = np.load(f"data/noise_ceiling/{args.subject}/lh.npy")
    nc_rh = np.load(f"data/noise_ceiling/{args.subject}/rh.npy")

    args.nc_lh = nc_lh
    args.nc_rh = nc_rh

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)