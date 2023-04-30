from utils import dino as utils

from .base_parser import get_base_parser


def get_args_parser():
    parser = get_base_parser()

    # Model, data, etc
    parser.add_argument("--model_name", default="MicroFormer", type=str)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--embed_dim", default=512, type=int)
    parser.add_argument("--depth", default=4, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--patch_size", default=16, type=int)

    parser.add_argument("--decoder_embed_dim", default=512, type=int)
    parser.add_argument("--decoder_depth", default=4, type=int)
    parser.add_argument("--decoder_num_heads", default=4, type=int)

    parser.add_argument("--drop_rate", default=0.0, type=float)

    parser.add_argument("--has_decoder", default=True, type=utils.bool_flag)
    parser.add_argument("--aux_cls", default=True, type=utils.bool_flag)
    parser.add_argument("--att_loss", default=False, type=utils.bool_flag)
    parser.add_argument("--diag_att", default=False, type=utils.bool_flag)
    parser.add_argument("--segmentation", default=False, type=utils.bool_flag)
    parser.add_argument("--replace_ratio", default=0.75, type=float)
    parser.add_argument("--pretrained_checkpoint", type=str, default="")
    parser.add_argument("--pretrained_encoder", type=str, default="dummy")

    # DINO
    parser.add_argument("--enable_dino", default=True, type=utils.bool_flag)
    parser.add_argument("--out_dim", default=65635, type=int)
    parser.add_argument("--local_crops_number", default=8, type=int)
    parser.add_argument("--teacher_temp", default=0.04, type=float)
    parser.add_argument("--warmup_teacher_temp", default=0.04, type=float)
    parser.add_argument("--warmup_teacher_temp_epochs", default=0, type=int)

    # dataset
    parser.add_argument("--dataset", default="SAMM", type=str)
    parser.add_argument("--samm_path", default="data/SAMM/", type=str)
    parser.add_argument("--casme2_path", default="data/CASME2-RAW/", type=str)
    parser.add_argument("--casme3_path", default="data/CASME3_A/", type=str)
    # parser.add_argument(
    #     "--tracks", default="data/volleyball/tracks_normalized.pkl", type=str
    # )

    return parser
