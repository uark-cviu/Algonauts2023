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
CUDA_VISIBLE_DEVICES=5 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir logs/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/convnext_base_in22ft1k/ --output_dir predictions/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/convnext_base_in22ft1k --subject subj01 && \\
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir logs/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/convnext_base_in22ft1k/ --output_dir predictions/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/convnext_base_in22ft1k --subject subj02 && \\
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir /scr1/1594489/logs/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/seresnext101d_32x8d/ --output_dir predictions/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/ --subject subj03 && \\
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir /scr1/1594489/logs/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/seresnext101d_32x8d/ --output_dir predictions/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/ --subject subj04 && \\
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir /scr1/1594489/logs/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/seresnext101d_32x8d/ --output_dir predictions/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/ --subject subj05 && \\
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir /scr1/1594489/logs/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/seresnext101d_32x8d/ --output_dir predictions/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/ --subject subj06 && \\
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir /scr1/1594489/logs/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/seresnext101d_32x8d/ --output_dir predictions/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/ --subject subj07 && \\
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --folds 0,1,2,3,4 --checkpoint_dir /scr1/1594489/logs/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/seresnext101d_32x8d/ --output_dir predictions/baseline_pcc_l1_384_ema_adaptive_loss_multisub0/ --subject subj08


"""



def get_args_parser():
    parser = argparse.ArgumentParser('Test', add_help=False)

    # dataset parameters
    parser.add_argument('--folds', default='0,1,2,3,4', type=str)
    parser.add_argument('--checkpoint_dir', type=str, default='')
    parser.add_argument('--subject', type=str, default='subj01')
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

    for subject_id in ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']:
        data_loader = None

        pred_rh_final = 0
        pred_lh_final = 0

        for fold in folds:
            checkpoint = f"{args.checkpoint_dir}/{fold}/last.pth"
            checkpoint = torch.load(checkpoint)
            train_args = checkpoint['args']
            print(train_args)
            root_data_dir = train_args.root_data_dir
            best_score = checkpoint['best_score']
            epoch = checkpoint['epoch']
            # subject_id = args.subject
            train_args.data_dir = f"{root_data_dir}/{subject_id}"
            print(f"[+] Predicting {fold} of {subject_id}, best score: {best_score} @ epoch {epoch}")


            if data_loader is None:
                data_loader = get_dataloader(train_args)

            # ============ building Clusformer ... ============
            model = get_model(train_args)
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(checkpoint['ema'])
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


        if subject_id == 'subj06':
            lh_length, rh_length = 18978, 20220
        elif subject_id == 'subj08':
            lh_length, rh_length = 18981, 20530
        else:
            lh_length, rh_length = 19004, 20544

        pred_rh_final = pred_rh_final[:, :rh_length].astype(np.float32)
        pred_lh_final = pred_lh_final[:, :lh_length].astype(np.float32)

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

