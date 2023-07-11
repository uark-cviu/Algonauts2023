import torch.nn as nn
import timm
import os
import torch
from torchvision import models as torchvision_models


class AlgonautsTimm(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        # self.backbone = timm.create_model(
        #     model_name=args.model_name, pretrained=True, num_classes=0
        # )

        if args.model_name in torchvision_models.__dict__.keys():
            self.backbone = torchvision_models.__dict__[args.model_name](
                weights="DEFAULT"
            )
            if hasattr(self.backbone, "fc"):
                embed_dim = self.backbone.fc.weight.shape[1]
                self.backbone.fc = nn.Identity()
            elif hasattr(self.backbone, "classifier"):
                embed_dim = self.backbone.classifier[-1].weight.shape[1]
                self.backbone.classifier = nn.Identity()

        # load_pretrained_weights(
        #     self.backbone,
        #     args.pretrained,
        #     "teacher",
        #     args.model_name,
        #     16,
        # )

        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)["model"]
            backbone_dict = {}
            for k, v in checkpoint.items():
                if "backbone" in k:
                    k_ = k[7:].replace("backbone.", "")
                    backbone_dict[k_] = v

            self.backbone.load_state_dict(backbone_dict)
            print(f"[+] Loaded: ", args.pretrained)

        # in_features = self.backbone.num_features
        in_features = embed_dim

        subject_metadata = args.subject_metadata

        self.side = ["l", "r"]
        # self.fc_embedding = nn.ModuleDict()
        self.fc = nn.ModuleDict()

        # embedding_size = 512

        for side in self.side:
            fc = nn.ModuleDict()
            # fc_embedding = nn.ModuleDict()
            for roi_name, roi_index in subject_metadata[side].items():
                roi_size = sum(roi_index)
                if roi_size > 0:
                    # fc_embedding[roi_name] = nn.Sequential(
                    #     nn.Linear(in_features, embedding_size),
                    #     # nn.BatchNorm1d(embedding_size, eps=1e-05),
                    # )
                    fc[roi_name] = nn.Linear(in_features, roi_size)
            self.fc[side] = fc
            self.fc[side + "_all"] = nn.Linear(in_features, roi_index.shape[0])
            # self.fc_embedding[side] = fc_embedding

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, batch):
        image = batch["image"]
        bs = image.shape[0]
        features = self.backbone(image)
        features = features.view(bs, -1)

        output_dict = {}
        for side in self.side:
            output_dict[side] = {}
            # output_dict[side + "_all"] = {}
            for roi_name, roi_fc in self.fc[side].items():
                # fc_embedding = self.fc_embedding[side][roi_name]
                # embedding = fc_embedding(features)
                output = roi_fc(features)
                output_dict[side][roi_name] = output
                # output_dict[side][roi_name + "_embedding"] = embedding
            output_dict[side + "_all"] = self.fc[side + "_all"](features)
        return output_dict


def get_timm_models(args):
    model = timm.create_model(
        model_name=args.model_name, pretrained=True, num_classes=args.num_outputs
    )
