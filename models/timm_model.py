import torch.nn as nn
import timm
import os
import torch
from transformers import RobertaModel


class AlgonautsTimm(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            model_name=args.model_name, pretrained=True, num_classes=0
        )

        self.text_backbone = RobertaModel.from_pretrained("roberta-base")
        text_num_features = 768
        self.text_fc = nn.Sequential(
            nn.Linear(text_num_features, text_num_features),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)['model']
            backbone_dict = {}
            for k, v in checkpoint.items():
                if 'backbone' in k:
                    k_ = k[7:].replace('backbone.', '')
                    backbone_dict[k_] = v
                
            self.backbone.load_state_dict(backbone_dict)
            print(f"[+] Loaded: ", args.pretrained)

        in_features = self.backbone.num_features + text_num_features

        subject_metadata = args.subject_metadata

        # self.side = ["l", "r"]
        self.side = args.side # ["l", "r"]
        self.fc = nn.ModuleDict()

        for side in self.side:
            fc = nn.ModuleDict()
            for roi_name, roi_index in subject_metadata[side].items():
                roi_size = sum(roi_index)
                if roi_size > 0:
                    fc[roi_name] = nn.Linear(in_features, roi_size)
            self.fc[side] = fc

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, batch):
        image = batch["image"]
        features = self.backbone(image)

        input_ids, attention_mask = (
            batch["ids"],
            batch["mask"]        
        )
        output_1 = self.text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_state = output_1[0]
        text_features = hidden_state[:, 0]
        text_features = self.text_fc(text_features)

        features = torch.cat([features, text_features], axis=-1)

        output_dict = {}
        for side in self.side:
            output_dict[side] = {}
            for roi_name, roi_fc in self.fc[side].items():
                output = roi_fc(features)
                output_dict[side][roi_name] = output

        return output_dict


def get_timm_models(args):
    model = timm.create_model(
        model_name=args.model_name, pretrained=True, num_classes=args.num_outputs
    )
