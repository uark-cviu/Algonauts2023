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

        embedding_dim = 512
        intermedia_features = 1024
        self.embedding = nn.Embedding(
            num_embeddings=8,
            embedding_dim=embedding_dim
        )

        in_features = self.backbone.num_features
        self.lh_fmri_fc = nn.Sequential(
            nn.Linear(in_features + embedding_dim, intermedia_features),
            nn.BatchNorm1d(intermedia_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(intermedia_features, args.num_lh_output)
        )
        self.rh_fmri_fc = nn.Sequential(
            nn.Linear(in_features + embedding_dim, intermedia_features),
            nn.BatchNorm1d(intermedia_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(intermedia_features, args.num_rh_output)
        )

    def forward(self, batch):
        image = batch['image']
        subject = batch['subject']
        features = self.backbone(image)
        embedding = self.embedding(subject)
        features = torch.cat([features, embedding], axis=1)
        lh_fmri = self.lh_fmri_fc(features)
        rh_fmri = self.rh_fmri_fc(features)

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
