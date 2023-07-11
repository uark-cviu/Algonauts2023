import torch.nn as nn
import timm
import torch
import os
from torchvision import models as torchvision_models
from .convnext import convnext_xlarge, convnext_large


convnext_coco_checkpoint = {
    "convnext_xlarge_coco": "https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_xlarge_22k_3x.pth",
    "convnext_large_coco": "https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_large_22k_3x.pth",
}


def get_convnext_coco(model_name):
    if model_name == "convnext_xlarge_coco":
        model = convnext_xlarge()
    elif model_name == "convnext_large_coco":
        model = convnext_large()

    model_state_dict = model.state_dict()
    url = convnext_coco_checkpoint[model_name]
    checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")[
        "state_dict"
    ]
    backbone_dict = {}
    for k, v in checkpoint.items():
        if "backbone" in k:
            k_ = k.replace("backbone.", "")
            if k_ in model_state_dict:
                backbone_dict[k_] = v
            else:
                print(f"{k_} is not used.")

    for k, v in model_state_dict.items():
        if not k in backbone_dict:
            print(f"{k} is initialized by default.")
            backbone_dict[k] = model_state_dict[k]

    model.load_state_dict(backbone_dict)
    return model


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
        else:
            self.backbone = get_convnext_coco(args.model_name)

        if hasattr(self.backbone, "fc"):
            embed_dim = self.backbone.fc.weight.shape[1]
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            embed_dim = self.backbone.classifier[-1].weight.shape[1]
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "head"):
            embed_dim = self.backbone.head.weight.shape[1]
            self.backbone.head = nn.Identity()

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

        embedding_dim = 512
        intermedia_features = 1024
        self.embedding = nn.Embedding(num_embeddings=8, embedding_dim=embedding_dim)

        self.lh_fmri_fc = nn.Sequential(
            nn.Linear(in_features + embedding_dim, intermedia_features),
            nn.BatchNorm1d(intermedia_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(intermedia_features, args.num_lh_output),
        )
        self.rh_fmri_fc = nn.Sequential(
            nn.Linear(in_features + embedding_dim, intermedia_features),
            nn.BatchNorm1d(intermedia_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(intermedia_features, args.num_rh_output),
        )

    def forward(self, batch):
        image = batch["image"]
        bs = image.shape[0]
        subject = batch["subject"]
        features = self.backbone(image)
        features = features.view(bs, -1)
        embedding = self.embedding(subject)
        features = torch.cat([features, embedding], axis=1)
        lh_fmri = self.lh_fmri_fc(features)
        rh_fmri = self.rh_fmri_fc(features)

        return {
            "lh_fmri": lh_fmri,
            "rh_fmri": rh_fmri,
        }


def get_timm_models(args):
    model = timm.create_model(
        model_name=args.model_name, pretrained=True, num_classes=args.num_outputs
    )
