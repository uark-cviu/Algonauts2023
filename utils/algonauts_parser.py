from utils import dino as utils

from .base_parser import get_base_parser


def get_args_parser():
    parser = get_base_parser()

    # Model, data, etc
    parser.add_argument("--model_name", default="MicroVIT", type=str)
    parser.add_argument("--num_lh_output", default=19004, type=int) # Will be overrided 
    parser.add_argument("--num_rh_output", default=20544, type=int) # Will be overrided
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--is_train", default=True, type=utils.bool_flag)
    parser.add_argument("--pretrained", default="logs/multisub/", type=str)

    # dataset
    parser.add_argument("--dataset", default="Algonauts", type=str)

    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--subject", default="subj01", type=str)
    parser.add_argument(
        "--csv_file", default="data/5folds_splits.csv", type=str
    )
    parser.add_argument("--num_folds", default=5, type=int)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--side", default='l,r', type=str)

    return parser
