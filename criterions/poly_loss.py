import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional
from torch.nn.modules.loss import _Loss


def to_one_hot(
    labels: torch.Tensor,
    num_classes: int,
    dtype: torch.dtype = torch.float,
    dim: int = 1,
) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


class PolyLoss(_Loss):
    def __init__(
        self,
        softmax: bool = True,
        ce_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        epsilon: float = 1.0,
    ) -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: if target is in one-hot format, its shape should be BNH[WD],
                if it is not one-hot encoded, it should has shape B1H[WD] or BH[WD], where N is the number of classes,
                It should contain binary values
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if len(input.shape) - len(target.shape) == 1:
            target = target.unsqueeze(1).long()
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        # target not in one-hot encode format, has shape B1H[WD]
        if n_pred_ch != n_target_ch:
            # squeeze out the channel dimension of size 1 to calculate ce loss
            self.ce_loss = self.cross_entropy(
                input, torch.squeeze(target, dim=1).long()
            )
            # convert into one-hot format to calculate ce loss
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1)  # BH[WD]
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        if self.reduction == "mean":
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == "sum":
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == "none":
            # BH[WD]
            polyl = poly_loss
        else:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )
        return polyl


class PolyBCELoss(_Loss):
    def __init__(
        self,
        reduction: str = "mean",
        epsilon: float = 1.0,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (∗), where * means any number of dimensions.
            target: same shape as the input
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """

        # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
        self.bce_loss = self.bce(input, target)
        pt = torch.sigmoid(input)
        pt = torch.where(target == 1, pt, 1 - pt)
        poly_loss = self.bce_loss + self.epsilon * (1 - pt)

        if self.reduction == "mean":
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == "sum":
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == "none":
            # BH[WD]
            polyl = poly_loss
        else:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )
        return polyl


class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, num_classes=4):
        super(SCELoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        if len(pred.shape) - len(labels.shape) == 1:
            labels = labels.unsqueeze(1).long()

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)  # b x h x w
        label_one_hot = to_one_hot(labels, num_classes=self.num_classes)
        #         label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
