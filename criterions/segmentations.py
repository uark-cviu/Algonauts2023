import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn

from .poly_loss import SCELoss


def dice_coef(y_pred, y_true, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_pred, y_true, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


class ComboSegmentationLoss(nn.Module):
    def __init__(self, multilabel=True, dice=True, tvk=False, poly=False):
        super(ComboSegmentationLoss, self).__init__()
        self.dice = dice
        self.tvk = tvk
        self.poly = poly

        mode = "multilabel" if multilabel else "multiclass"
        self.dice_loss_fn = smp.losses.DiceLoss(mode=mode)
        self.tvk_loss_fn = smp.losses.TverskyLoss(mode=mode)
        self.poly_loss_fn = nn.BCEWithLogitsLoss() if multilabel else SCELoss()

    def forward(self, y_pred, y_true):
        loss = 0
        count = 0
        if self.dice:
            loss += self.dice_loss_fn(y_pred, y_true)
            count += 1

        if self.tvk:
            loss += self.tvk_loss_fn(y_pred, y_true)
            count += 1

        if self.poly:
            loss += self.poly_loss_fn(y_pred, y_true)
            count += 1

        return loss / count
