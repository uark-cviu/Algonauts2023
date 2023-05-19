import torch
import torch.nn as nn
from transformers import RobertaModel


class RobertaClass(torch.nn.Module):
    def __init__(self, args):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        in_features = 768
        subject_metadata = args.subject_metadata

        # self.side = ["l", "r"]
        self.side = args.side  # ["l", "r"]
        self.fc = nn.ModuleDict()

        for side in self.side:
            fc = nn.ModuleDict()
            for roi_name, roi_index in subject_metadata[side].items():
                roi_size = sum(roi_index)
                if roi_size > 0:
                    fc[roi_name] = nn.Linear(in_features, roi_size)
            self.fc[side] = fc

    def forward(self, batch):
        input_ids, attention_mask, token_type_ids = (
            batch["ids"],
            batch["mask"],
            batch["token_type_ids"],
        )
        output_1 = self.l1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_state = output_1[0]
        features = hidden_state[:, 0]

        output_dict = {}
        for side in self.side:
            output_dict[side] = {}
            for roi_name, roi_fc in self.fc[side].items():
                output = roi_fc(features)
                output_dict[side][roi_name] = output

        return output_dict
