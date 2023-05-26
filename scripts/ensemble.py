import numpy as np
import os


subs = [
    [
        "predictions_vision/convnext_xlarge_onecycle",
        "predictions_vision/convnext_xlarge_onecycle_euloss",
        "predictions_vision/convnext_xlarge_onecycle_mse",
        "predictions_vision/convnext_xlarge_onecycle_bs4",
        "predictions_vision/convnext_xlarge_onecycle_augmix",
    ],
    [
        "predictions_vision/seresnext101d_32x8d_onecycle",
        "predictions_vision/seresnext101d_32x8d_onecycle_bs4",
    ],
    ["predictions_vision/swin_large_patch4_window12_384_onecycle"],
]


def average(all_subs, sub_id):
    pred_lh = 0
    pred_rh = 0
    for sub in all_subs:
        print(sub)
        pred_lh += np.load(f"{sub}/subj0{sub_id}/lh_pred_test.npy") / len(all_subs)
        pred_rh += np.load(f"{sub}/subj0{sub_id}/rh_pred_test.npy") / len(all_subs)

    return pred_lh, pred_rh


weights = [0.5, 0.3, 0.2]

save_dir = "predictions_vision/ensemble_convnext_xlarge_seresnext101d_32x8d_swin_large_patch4_window12_384_multiple"


for sub_id in range(1, 9):
    pred_lh = 0
    pred_rh = 0
    for sub, weight in zip(subs, weights):
        # print(sub)
        pred_lh_sub, pred_rh_sub = average(sub, sub_id)
        pred_lh += pred_lh_sub * weight
        pred_rh += pred_rh_sub * weight

    os.makedirs(f"{save_dir}/subj0{sub_id}/", exist_ok=True)
    np.save(f"{save_dir}/subj0{sub_id}/lh_pred_test.npy", pred_lh)
    np.save(f"{save_dir}/subj0{sub_id}/rh_pred_test.npy", pred_rh)
