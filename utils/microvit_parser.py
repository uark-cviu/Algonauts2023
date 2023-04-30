from utils import dino as utils

from .base_parser import get_base_parser


def get_args_parser():
    parser = get_base_parser()

    # Model, data, etc
    parser.add_argument("--model_name", default="MicroVIT", type=str)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--drop_rate", default=0.4, type=float)
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--pretrained_checkpoint", type=str, default="")
    parser.add_argument("--mae_weight", type=str, default="")
    parser.add_argument("--is_train", default=True, type=utils.bool_flag)

    # dataset
    parser.add_argument("--dataset", default="SAMM", type=str)

    parser.add_argument("--samm_path", default="data/SAMM_update/", type=str)
    parser.add_argument(
        "--samm_csv_file", default="data/SAMM_update/SAMM_v2.csv", type=str
    )

    parser.add_argument("--casme2_path", default="data/CASME2-RAW/", type=str)
    parser.add_argument("--casme3_path", default="data/CASME3_A_onoffset/", type=str)

    parser.add_argument("--csv_file", default="data/CASME2-anno.csv", type=str)
    parser.add_argument("--subject", default=1, type=int)
    parser.add_argument("--extract_features", default=False, type=utils.bool_flag)
    parser.add_argument("--feature_dir", default="", type=str)

    # parser.add_argument(
    #     "--tracks", default="data/volleyball/tracks_normalized.pkl", type=str
    # )

    return parser
