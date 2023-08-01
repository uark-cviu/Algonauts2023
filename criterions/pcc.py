import torch
import torch.nn as nn


class PCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, pred, gt, nc=None):
        # pearson = self.cos(
        #     x1 - x1.mean(dim=1, keepdim=True), x2 - x2.mean(dim=1, keepdim=True)
        # )
        # pearson = pearson.mean()
        # pearson = (pearson + 1) / 2
        # return 1 - pearson
        pred_mean = torch.mean(pred, axis=0)
        gt_mean = torch.mean(gt, axis=0)

        # num_index = pred.shape[1]
        # num_sample = pred.shape[0]

        # rvs = 0

        # for v in range(num_index):
        #     ts, ms1, ms2 = 0, 0, 0
        #     for t in range(num_sample):
        #         ts += (gt[t, v] - gt_mean[v]) * (pred[t, v] - pred_mean[v])
        #         ms1 += (gt[t, v] - gt_mean[v]) ** 2
        #         ms2 += (pred[t, v] - pred_mean[v]) ** 2
        #     ms = (ms1 * ms2) ** 0.5
        #     rv = ts / (ms + 1e-8)
        #     rv = 1 - (rv + 1) / 2
        #     rvs += rv / num_index

        gt_t = gt.T
        pred_t = pred.T

        ts = (gt_t - gt_mean.view(-1, 1)) * (pred_t - pred_mean.view(-1, 1))
        ts = ts.sum(axis=1)

        ms1 = (gt_t - gt_mean.view(-1, 1)) ** 2
        ms1 = ms1.sum(axis=1)

        ms2 = (pred_t - pred_mean.view(-1, 1)) ** 2
        ms2 = ms2.sum(axis=1)
        ms = (ms1 * ms2) ** 0.5

        rv = ts / (ms + 1e-8)
        rv = 1 - (rv + 1) / 2

        if nc is not None:
            nc[nc == 0] = 1
            rv = rv**2 / torch.from_numpy(nc).to(rv.device)

        return rv.mean()
