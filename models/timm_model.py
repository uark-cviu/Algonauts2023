import torch.nn as nn
import timm
import torch



class AlgonautsTimm(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            model_name=args.model_name,
            pretrained=True,
            num_classes=0
        )

        in_features = self.backbone.num_features
        self.lh_fmri_fc = nn.Linear(in_features, args.num_lh_output)
        self.rh_fmri_fc = nn.Linear(in_features, args.num_rh_output)

    def forward(self, batch):
        image = batch['image']
        features = self.backbone(image)
        lh_fmri = self.lh_fmri_fc(features)
        rh_fmri = self.rh_fmri_fc(features)

        # lh_fmri = torch.tanh(lh_fmri)
        # rh_fmri = torch.tanh(rh_fmri)

        return {
            'lh_fmri': lh_fmri,
            'rh_fmri': rh_fmri,
        }


def get_timm_models(args):
    model = timm.create_model(
        model_name=args.model_name,
        pretrained=True,
        num_classes=args.num_outputs
    )
