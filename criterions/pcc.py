import torch
import torch.nn as nn


class PCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x1, x2):
        pearson = self.cos(x1 - x1.mean(dim=1, keepdim=True), x2 - x2.mean(dim=1, keepdim=True))
        pearson = pearson.mean()
        pearson = (pearson + 1)/2
        return 1 - pearson
