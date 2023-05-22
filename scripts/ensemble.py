import numpy as np
import os


subs = [
    "predictions_vision/convnext_xlarge_onecycle",
    "predictions_vision/seresnext101d_32x8d_onecycle",
]

weights = [0.65, 0.35]

save_dir = "predictions_vision/ensemble_convnext_xlarge_seresnext101d_32x8d_onecycle"


for sub_id in range(1, 9):
    pred_lh = 0
    pred_rh = 0
    for sub, weight in zip(subs, weights):
        print(sub)
        pred_lh += np.load(f"{sub}/subj0{sub_id}/lh_pred_test.npy") * weight
        pred_rh += np.load(f"{sub}/subj0{sub_id}/rh_pred_test.npy") * weight

    os.makedirs(f"{save_dir}/subj0{sub_id}/", exist_ok=True)
    np.save(f"{save_dir}/subj0{sub_id}/lh_pred_test.npy", pred_lh)
    np.save(f"{save_dir}/subj0{sub_id}/rh_pred_test.npy", pred_rh)
