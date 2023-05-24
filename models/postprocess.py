import torch
import torch.nn as nn
import numpy as np


class PostProcessModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.subject_metadata = args.subject_metadata
        self.num_models = len(args.model_names)

        self.side = ["l", "r"]
        # self.fc = nn.ModuleDict()

        weight = nn.Parameter(torch.Tensor([1/self.num_models for _ in range(self.num_models)]).float(), requires_grad=True)

        for side in self.side:
            for roi_name, roi_index in self.subject_metadata[side].items():
                roi_size = sum(roi_index)
                if roi_size > 0:
                    self.__setattr__(f"{side}_{roi_name}", weight)

    def forward(self, batch):
        data_l = batch["data_l"] # N x (n_models) x seq
        data_r = batch["data_r"] # N x (n_models) x seq

        output_dict = {}

        for side in self.side:
            if side == 'r':
                data = data_r
            else:
                data = data_l

            output_dict[side] = {}

            for roi_name, roi_index in self.subject_metadata[side].items():
                roi_size = sum(roi_index)
                if roi_size > 0:
                    roi_signal = data[:, :, np.where(roi_index)[0]]
                    weight = self.__getattr__(f"{side}_{roi_name}") 

                    total = 0
                    weight_sum = 0
                    for i in range(self.num_models):
                        total += roi_signal[:, i] * weight[i]

                        weight_sum += weight[i]
                    total /= weight_sum
                    # import pdb; pdb.set_trace()
                    output_dict[side][roi_name] = total

        return output_dict