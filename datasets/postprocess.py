from torch.utils.data import Dataset
import numpy as np
import pickle

class EnsembleDataset(Dataset):
    def __init__(
        self, oof_dir, subject="subj01", models=[], fold=0, num_folds=5, is_train=False
    ):
        self.data_l, self.data_r = [], []
        self.lh_fmris = []
        self.rh_fmris = []

        loaded_folds = (
            [fold] if not is_train else [i for i in range(num_folds) if i != fold]
        )
        for i in loaded_folds:
            data_l, data_r = [], []
            for model_name in models:
                oof_file = f"{oof_dir}/{model_name}/{subject}/fold_{i}.pkl"
                l, r, lh_fmri, rh_fmri = self.load_valid_data(oof_file)
                data_l.append(l)
                data_r.append(r)
            self.lh_fmris.append(lh_fmri)
            self.rh_fmris.append(rh_fmri)
            data_l = np.concatenate(
                data_l, axis=1
            ) # N x num_models x seq
            data_r = np.concatenate(
                data_r, axis=1
            ) # N x num_models x seq

            self.data_l.append(data_l)
            self.data_r.append(data_r)

        self.data_l = np.concatenate(self.data_l, axis=0)
        self.data_r = np.concatenate(self.data_r, axis=0)
        self.lh_fmris = np.concatenate(self.lh_fmris, axis=0)  # N x seq_lh
        self.rh_fmris = np.concatenate(self.rh_fmris, axis=0)  # N x seq_rh

        self.num_lh_output = self.lh_fmris.shape[1]
        self.num_rh_output = self.rh_fmris.shape[1]

    def pad_if_need(self, l, r):
        max_len = max(l.shape[1], r.shape[1])
        N = l.shape[0]
        new_l = np.zeros((N, max_len))
        new_r = np.zeros((N, max_len))

        new_l[:, : l.shape[1]] = l
        new_r[:, : r.shape[1]] = r

        return new_l, new_r

    def load_valid_data(self, pickle_file):
        with open(pickle_file, "rb") as f:
            data_dict = pickle.load(f)

        l, r = data_dict["valid"]["l"], data_dict["valid"]["r"] # N x seq
        l = np.expand_dims(l, axis=1)
        r = np.expand_dims(r, axis=1)
        # l, r = self.pad_if_need(l, r)  # N x seq
        # l = np.expand_dims(l, axis=1)  # N x 1 x seq
        # r = np.expand_dims(r, axis=1)  # N x 1 x seq
        # data = np.concatenate([l, r], axis=1)  # N x 2 x seq
        # data = np.expand_dims(data, axis=-1)  # N x 2 x seq x 1
        gt_l, gt_r = data_dict["valid_gt"]["l"], data_dict["valid_gt"]["r"]
        return l, r, gt_l, gt_r

    def __len__(self):
        return len(self.data_l)

    def __getitem__(self, index):
        data_l = self.data_l[index].astype(np.float32)
        data_r = self.data_r[index].astype(np.float32)
        # data = np.transpose(data, (, 1, 2))
        lh_fmri = self.lh_fmris[index].astype(np.float32)
        rh_fmri = self.rh_fmris[index].astype(np.float32)

        return {"data_l": data_l, "data_r": data_r, "l": lh_fmri, "r": rh_fmri}


class EnsembleTestDataset(Dataset):
    def __init__(
        self, oof_dir, subject="subj01", models=[], fold=0, num_folds=5, is_train=False
    ):
        self.data_l, self.data_r = [], []
        # self.lh_fmris = []
        # self.rh_fmris = []

        loaded_folds = (
            [fold] if not is_train else [i for i in range(num_folds) if i != fold]
        )
        for i in loaded_folds:
            data_l, data_r = [], []
            for model_name in models:
                oof_file = f"{oof_dir}/{model_name}/{subject}/fold_{i}.pkl"
                l, r = self.load_valid_data(oof_file)
                data_l.append(l)
                data_r.append(r)
            # self.lh_fmris.append(lh_fmri)
            # self.rh_fmris.append(rh_fmri)
            data_l = np.concatenate(
                data_l, axis=1
            ) # N x num_models x seq
            data_r = np.concatenate(
                data_r, axis=1
            ) # N x num_models x seq

            self.data_l.append(data_l)
            self.data_r.append(data_r)

        self.data_l = np.concatenate(self.data_l, axis=0)
        self.data_r = np.concatenate(self.data_r, axis=0)
        # self.lh_fmris = np.concatenate(self.lh_fmris, axis=0)  # N x seq_lh
        # self.rh_fmris = np.concatenate(self.rh_fmris, axis=0)  # N x seq_rh

        # self.num_lh_output = self.lh_fmris.shape[1]
        # self.num_rh_output = self.rh_fmris.shape[1]

    def pad_if_need(self, l, r):
        max_len = max(l.shape[1], r.shape[1])
        N = l.shape[0]
        new_l = np.zeros((N, max_len))
        new_r = np.zeros((N, max_len))

        new_l[:, : l.shape[1]] = l
        new_r[:, : r.shape[1]] = r

        return new_l, new_r

    def load_valid_data(self, pickle_file):
        with open(pickle_file, "rb") as f:
            data_dict = pickle.load(f)

        l, r = data_dict["test"]["l"], data_dict["test"]["r"] # N x seq
        l = np.expand_dims(l, axis=1)
        r = np.expand_dims(r, axis=1)
        # l, r = self.pad_if_need(l, r)  # N x seq
        # l = np.expand_dims(l, axis=1)  # N x 1 x seq
        # r = np.expand_dims(r, axis=1)  # N x 1 x seq
        # data = np.concatenate([l, r], axis=1)  # N x 2 x seq
        # data = np.expand_dims(data, axis=-1)  # N x 2 x seq x 1
        # gt_l, gt_r = data_dict["valid_gt"]["l"], data_dict["valid_gt"]["r"]
        return l, r

    def __len__(self):
        return len(self.data_l)

    def __getitem__(self, index):
        data_l = self.data_l[index].astype(np.float32)
        data_r = self.data_r[index].astype(np.float32)

        return {"data_l": data_l, "data_r": data_r}