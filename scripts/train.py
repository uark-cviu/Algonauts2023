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
from criterions.pcc import PCCLoss
import robust_loss_pytorch


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.l1_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()
        self.pcc = PCCLoss()
        self.adaptive_lh = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims = args.num_lh_output, float_dtype=np.float32, device=f'cuda:{args.gpu}'
        )

        self.adaptive_rh = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims = args.num_rh_output, float_dtype=np.float32, device=f'cuda:{args.gpu}'
        )

    def forward(self, outputs, batch):
        pred_lh_fmri = outputs['lh_fmri']
        pred_rh_fmri = outputs['rh_fmri']
        gt_lh_fmri = batch['lh_fmri']
        gt_rh_fmri = batch['rh_fmri']

        # l1_loss = self.l1_loss(pred_lh_fmri, gt_lh_fmri) + self.l1_loss(pred_rh_fmri, gt_rh_fmri)
        loss_lh = torch.mean(self.adaptive_lh.lossfun((pred_lh_fmri - gt_lh_fmri)))
        loss_rh = torch.mean(self.adaptive_rh.lossfun((pred_rh_fmri - gt_rh_fmri)))
        l1_loss = loss_lh + loss_rh
        pcc_loss = self.pcc(pred_lh_fmri, gt_lh_fmri) + self.pcc(pred_rh_fmri, gt_rh_fmri)
        loss = l1_loss + pcc_loss
        # import pdb; pdb.set_trace()

        return loss


class Metric:
    def __init__(self, args):
        # from torchmetrics import PearsonCorrCoef
        # selpearson = PearsonCorrCoef()
        self.args = args
        # Load the ROI classes mapping dictionaries
        roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
            'mapping_floc-faces.npy', 'mapping_floc-places.npy',
            'mapping_floc-words.npy', 'mapping_streams.npy']
        self.roi_name_maps = []
        for r in roi_mapping_files:
            self.roi_name_maps.append(np.load(os.path.join(args.data_dir, 'roi_masks', r),
                allow_pickle=True).item())

        # Load the ROI brain surface maps
        lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
            'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
            'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
            'lh.streams_challenge_space.npy']
        rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
            'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
            'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
            'rh.streams_challenge_space.npy']
        self.lh_challenge_rois = []
        self.rh_challenge_rois = []
        for r in range(len(lh_challenge_roi_files)):
            self.lh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
                lh_challenge_roi_files[r])))
            self.rh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
                rh_challenge_roi_files[r])))


    def __call__(self, pred_lh_fmri, pred_rh_fmri, gt_lh_fmri, gt_rh_fmri):

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

        # Select the correlation results vertices of each ROI
        roi_names = []
        lh_roi_correlation = []
        rh_roi_correlation = []
        for r1 in range(len(self.lh_challenge_rois)):
            for r2 in self.roi_name_maps[r1].items():
                if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                    roi_names.append(r2[1])
                    lh_roi_idx = np.where(self.lh_challenge_rois[r1] == r2[0])[0]
                    rh_roi_idx = np.where(self.rh_challenge_rois[r1] == r2[0])[0]
                    lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                    rh_roi_correlation.append(rh_correlation[rh_roi_idx])
        roi_names.append('All vertices')
        lh_roi_correlation.append(lh_correlation)
        rh_roi_correlation.append(rh_correlation)

        # Create the plot
        lh_median_roi_correlation = [np.median(lh_roi_correlation[r])
            for r in range(len(lh_roi_correlation))]
        rh_median_roi_correlation = [np.median(rh_roi_correlation[r])
            for r in range(len(rh_roi_correlation))]


        avg = (lh_median_roi_correlation[-1] + rh_median_roi_correlation[-1])/2
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

    # print(model)

    return model, model_ema


def get_dataloader(args):
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            # transforms.RandomResizedCrop(args.img_size),
            # transforms.RandomHorizontalFlip(),
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

    train_datasets = []
    valid_datasets = []
    root_data_dir = args.data_dir
    args.root_data_dir = root_data_dir
    # for subject in ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj07']:
    for subject in ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']:
        args.data_dir = f"{root_data_dir}/{subject}"
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
            transform=valid_transform,
            fold=args.fold,
            num_folds=args.num_folds,
            is_train=False
        )

        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)

    train_datasets = torch.utils.data.ConcatDataset(train_datasets)
    valid_datasets = torch.utils.data.ConcatDataset(valid_datasets)

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_datasets)
        if args.distributed
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        batch_size=args.batch_size_per_gpu,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )


    valid_sampler = (
        torch.utils.data.distributed.DistributedSampler(valid_datasets)
        if args.distributed
        else None
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_datasets,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=valid_sampler
    )


    # args.num_lh_output = train_dataset.num_lh_output
    # args.num_rh_output = train_dataset.num_rh_output

    args.num_lh_output = train_dataset.max_lh_length
    args.num_rh_output = train_dataset.max_rh_length

    args.min_max_lh = train_dataset.min_max_lh
    args.min_max_rh = train_dataset.min_max_rh

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
            is_ema_better = ema_valid_stats["corr"] > valid_stats["corr"]
        else:
            current_score = valid_stats["corr"]
            is_ema_better = None

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
            if is_ema_better:
                print("EMA is better")
                save_dict["model"] = model_ema.state_dict()

        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()

        if is_save_best:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, "best.pth"))

        utils.save_on_master(save_dict, os.path.join(args.output_dir, "last.pth"))

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

    folds = args.folds.split(",")
    for fold in folds:
        print("training fold ", fold)
        args.fold = fold
        args.output_dir = f"{output_dir}/{fold}/"
        os.makedirs(args.output_dir, exist_ok=True)
        train_one_fold(args)


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
        pred_lh_fmris = []
        pred_rh_fmris = []
        gt_lh_fmris = []
        gt_rh_fmris = []
        subject_ids = []
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
                subject_id = batch['subject'].detach().cpu().numpy()

                pred_lh_fmris.append(pred_lh_fmri)
                pred_rh_fmris.append(pred_rh_fmri)
                gt_lh_fmris.append(gt_lh_fmri)
                gt_rh_fmris.append(gt_rh_fmri)
                subject_ids.append(subject_id)

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

        # if is_train:
        #     break

    if not is_train:
        pred_lh_fmris = np.concatenate(pred_lh_fmris, axis=0)
        pred_rh_fmris = np.concatenate(pred_rh_fmris, axis=0)
        gt_lh_fmris = np.concatenate(gt_lh_fmris, axis=0)
        gt_rh_fmris = np.concatenate(gt_rh_fmris, axis=0)
        subject_ids = np.concatenate(subject_ids, axis=0)
        # import pdb; pdb.set_trace()
        root_data_dir = args.root_data_dir
        subject_unique_id = np.unique(subject_ids)
        total = 0
        for subject in subject_unique_id:
            subject_idx = np.where(subject_ids == subject)[0]
            pred_lh_fmris_ = pred_lh_fmris[subject_idx]
            pred_rh_fmris_ = pred_rh_fmris[subject_idx]
            gt_lh_fmris_ = gt_lh_fmris[subject_idx]
            gt_rh_fmris_ = gt_rh_fmris[subject_idx]

            if subject + 1 == 6:
                lh_length, rh_length = 18978, 20220
            elif subject + 1 == 8:
                lh_length, rh_length = 18981, 20530
            else:
                lh_length, rh_length = 19004, 20544

            pred_lh_fmris_ = pred_lh_fmris_[:, :lh_length]
            gt_lh_fmris_ = gt_lh_fmris_[:, :lh_length]

            pred_rh_fmris_ = pred_rh_fmris_[:, :rh_length]
            gt_rh_fmris_ = gt_rh_fmris_[:, :rh_length]

            args.data_dir = f"{root_data_dir}/subj0{subject+1}"
            metric_fn = Metric(args)

            avg = metric_fn(pred_lh_fmris_, pred_rh_fmris_, gt_lh_fmris_, gt_rh_fmris_)
            metric_logger.update(**{f'corr_{subject+1}': avg})
            total += avg

        total = total / len(subject_unique_id)
        metric_logger.update(**{'corr': total})
        # for k, v in metric_dict.items():
        #     metric_logger.update(**{k: v})
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

