from utils import dino as utils

from .base_parser import get_base_parser


def get_args_parser():
    parser = get_base_parser()

    # Model, data, etc
    parser.add_argument("--model_name", default="SwinGAR", type=str)
    parser.add_argument("--backbone", default="swin_tiny", type=str)
    parser.add_argument("--num_actions", default=9, type=int)
    parser.add_argument("--num_activities", default=8, type=int)
    parser.add_argument("--backbone_lr_mult", default=1.0, type=float)

    parser.add_argument("--pretrained_checkpoint", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--pretrained2d", type=utils.bool_flag, default=False)

    # dataset
    parser.add_argument("--img_h", default=360, type=int)
    parser.add_argument("--img_w", default=640, type=int)
    parser.add_argument("--data_path", default="data/volleyball/videos", type=str)
    parser.add_argument(
        "--tracks", default="data/volleyball/tracks_normalized.pkl", type=str
    )

    return parser
