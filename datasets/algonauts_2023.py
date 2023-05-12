import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import KFold
import glob
import os
from pathlib import Path

# class AlgonautsDataset(Dataset):
#     def __init__(self, data_dir, csv_file=None, transform=None, fold=0, num_folds=5, is_train=True):
#         if not os.path.isfile(csv_file):
#             images = []
#             kf = KFold(num_folds, shuffle=True, random_state=42)
#             for subject in ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]:
#                 images += glob.glob(f"{data_dir}/{subject}/training_split/training_images/*.png")

#             df = pd.DataFrame({
#                 'image': images
#             })
#             df['fold'] = 0

#             for fold, (_, valid_idx) in enumerate(kf.split(images)):
#                 df.iloc[valid_idx, -1] = fold

#             df.to_csv(f"{csv_file}", index=False)
#         else:
#             df = pd.read_csv(f"{csv_file}")

#         if is_train:
#             self.df = df[df['fold'] != fold].reset_index(drop=True)
#         else:
#             self.df = df[df['fold'] == fold].reset_index(drop=True)

#         self.images = self.df['image'].values
#         self.transform = transform
#         self.fmri_dict = {}

#     def load_image(self, img_path):
#         return Image.open(img_path).convert('RGB')

#     def load_fmri(self, img_path, prefix='lh'):
#         subject_id = img_path.split("/")[-4]
#         index = int(img_path.split("/")[-1].split(".")[0].split("-")[1].split("_")[0])- 1
#         assert index >= 0
#         if subject_id in self.fmri_dict and prefix in self.fmri_dict[subject_id]:
#             fmri_data = self.fmri_dict[subject_id][prefix]
#         else:
#             mri_path = "/".join(img_path.split("/")[:-2])
#             mri_path = f"{mri_path}/training_fmri/{prefix}_training_fmri.npy"
#             fmri_data = np.load(mri_path)
#             if not subject_id in self.fmri_dict:
#                 self.fmri_dict[subject_id] = {}

#             if not prefix in self.fmri_dict[subject_id]:
#                 self.fmri_dict[subject_id][prefix] = fmri_data

#         return fmri_data[index]

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         # Load the image
#         img_path = self.images[idx]
#         img = self.load_image(img_path)
#         lh_fmri = self.load_fmri(img_path, prefix='lh')[:18978]
#         rh_fmri = self.load_fmri(img_path, prefix='rh')[:20220]
#         if self.transform:
#             img = self.transform(img)
#         return {
#             "image": img,
#             "lh_fmri": lh_fmri,
#             "rh_fmri": rh_fmri
#         }


class AlgonautsDataset(Dataset):
    def __init__(self, data_dir, csv_file=None, transform=None, fold=0, num_folds=5, is_train=True):
        if not os.path.isfile(csv_file):
            images = []
            kf = KFold(num_folds, shuffle=True, random_state=42)
            # for subject in ["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07", "subj08"]:
            images += glob.glob(f"{data_dir}/training_split/training_images/*.png")

            df = pd.DataFrame({
                'image': images
            })
            df['fold'] = 0

            for fold, (_, valid_idx) in enumerate(kf.split(images)):
                df.iloc[valid_idx, -1] = fold

            df.to_csv(f"{csv_file}", index=False)
        else:
            df = pd.read_csv(f"{csv_file}")

        if is_train:
            self.df = df[df['fold'] != fold].reset_index(drop=True)
        else:
            self.df = df[df['fold'] == fold].reset_index(drop=True)

        self.images = self.df['image'].values
        self.transform = transform
        self.fmri_dict = {}

        subject = data_dir.split("/")[-1]
        if subject in ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj07']:
            self.num_lh_output = 19004
            self.num_rh_output = 20544
        elif subject in ['subj06']:
            self.num_lh_output = 18978
            self.num_rh_output = 20220
        elif subject in ['subj08']:
            self.num_lh_output = 18981
            self.num_rh_output = 20530

        if subject == 'subj01':
            self.min_max_lh = [-5.5488534, 6.3958163]
            self.min_max_rh = [-6.224722, 6.2803955]

        elif subject == 'subj02':
            self.min_max_lh = [-6.2695646 ,  7.303040]
            self.min_max_rh = [-7.10663 ,  6.722497]

        elif subject == 'subj03':
            self.min_max_lh = [-9.143569 ,  7.1541204]
            self.min_max_rh = [-7.3392262 ,  7.5514464]

        elif subject == 'subj04':
            self.min_max_lh = [-10.779788 ,  12.100054]
            self.min_max_rh = [-9.624842 ,  14.346183]

        elif subject == 'subj05':
            self.min_max_lh = [-6.7292423 ,  7.357646]
            self.min_max_rh = [-7.615682 ,  7.400172]

        elif subject == 'subj06':
            self.min_max_lh = [-8.748659 ,  9.179561]
            self.min_max_rh = [-9.317543 ,  9.996536]

        elif subject == 'subj07':
            self.min_max_lh = [-7.806317 ,  8.991561]
            self.min_max_rh = [-10.895111 ,  8.987522]

        elif subject == 'subj08':
            self.min_max_lh = [ -11.138769 ,  10.5266905]
            self.min_max_rh = [-9.20942 ,  10.573892]

    def load_image(self, img_path):
        return Image.open(img_path).convert('RGB')

    def load_fmri(self, img_path, prefix='lh'):
        subject_id = img_path.split("/")[-4]
        index = int(img_path.split("/")[-1].split(".")[0].split("-")[1].split("_")[0])- 1
        assert index >= 0
        if subject_id in self.fmri_dict and prefix in self.fmri_dict[subject_id]:
            fmri_data = self.fmri_dict[subject_id][prefix]
        else:
            mri_path = "/".join(img_path.split("/")[:-2])
            mri_path = f"{mri_path}/training_fmri/{prefix}_training_fmri.npy"
            fmri_data = np.load(mri_path)
            if not subject_id in self.fmri_dict:
                self.fmri_dict[subject_id] = {}

            if not prefix in self.fmri_dict[subject_id]:
                self.fmri_dict[subject_id][prefix] = fmri_data

        return fmri_data[index]

    # def min_max_transform(self, arr, prefix='lh'):
    #     if prefix == 'lh':
    #         min_val, max_val = self.min_max_lh
    #     else:
    #         min_val, max_val = self.min_max_rh
    #     return 2 * (arr - min_val) / (max_val - min_val) - 1

    # def min_max_transform(self, arr, prefix='lh'):
    #     if prefix == 'lh':
    #         min_val, max_val = self.min_max_lh
    #     else:
    #         min_val, max_val = self.min_max_rh
    #     return arr / max_val


    def min_max_transform(self, arr, prefix='lh'):
        return arr


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.images[idx]
        img = self.load_image(img_path)
        lh_fmri = self.load_fmri(img_path, prefix='lh')
        lh_fmri = self.min_max_transform(lh_fmri, prefix='lh')

        rh_fmri = self.load_fmri(img_path, prefix='rh')
        rh_fmri = self.min_max_transform(rh_fmri, prefix='rh')
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "l": lh_fmri,
            "r": rh_fmri
        }


class AlgonautsTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        test_img_dir = f"{data_dir}/test_split/test_images/"
        self.images = sorted(list(Path(test_img_dir).iterdir()))
        self.transform = transform

        subject = data_dir.split("/")[-1]

        if subject in ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj07']:
            self.num_lh_output = 19004
            self.num_rh_output = 20544
        elif subject in ['subj06']:
            self.num_lh_output = 18978
            self.num_rh_output = 20220
        elif subject in ['subj08']:
            self.num_lh_output = 18981
            self.num_rh_output = 20530

        if subject == 'subj01':
            self.min_max_lh = [-5.5488534, 6.3958163]
            self.min_max_rh = [-6.224722, 6.2803955]

        elif subject == 'subj02':
            self.min_max_lh = [-6.2695646 ,  7.303040]
            self.min_max_rh = [-7.10663 ,  6.722497]

        elif subject == 'subj03':
            self.min_max_lh = [-9.143569 ,  7.1541204]
            self.min_max_rh = [-7.3392262 ,  7.5514464]

        elif subject == 'subj04':
            self.min_max_lh = [-10.779788 ,  12.100054]
            self.min_max_rh = [-9.624842 ,  14.346183]

        elif subject == 'subj05':
            self.min_max_lh = [-6.7292423 ,  7.357646]
            self.min_max_rh = [-7.615682 ,  7.400172]

        elif subject == 'subj06':
            self.min_max_lh = [-8.748659 ,  9.179561]
            self.min_max_rh = [-9.317543 ,  9.996536]

        elif subject == 'subj07':
            self.min_max_lh = [-7.806317 ,  8.991561]
            self.min_max_rh = [-10.895111 ,  8.987522]

        elif subject == 'subj08':
            self.min_max_lh = [ -11.138769 ,  10.5266905]
            self.min_max_rh = [-9.20942 ,  10.573892]

    def load_image(self, img_path):
        return Image.open(img_path).convert('RGB')

    def load_fmri(self, img_path, prefix='lh'):
        subject_id = img_path.split("/")[-4]
        index = int(img_path.split("/")[-1].split(".")[0].split("-")[1].split("_")[0])- 1
        assert index >= 0
        if subject_id in self.fmri_dict and prefix in self.fmri_dict[subject_id]:
            fmri_data = self.fmri_dict[subject_id][prefix]
        else:
            mri_path = "/".join(img_path.split("/")[:-2])
            mri_path = f"{mri_path}/training_fmri/{prefix}_training_fmri.npy"
            fmri_data = np.load(mri_path)
            if not subject_id in self.fmri_dict:
                self.fmri_dict[subject_id] = {}

            if not prefix in self.fmri_dict[subject_id]:
                self.fmri_dict[subject_id][prefix] = fmri_data

        return fmri_data[index]

    # def min_max_transform(self, arr, prefix='lh'):
    #     if prefix == 'lh':
    #         min_val, max_val = self.min_max_lh
    #     else:
    #         min_val, max_val = self.min_max_rh
    #     return 2 * (arr - min_val) / (max_val - min_val) - 1

    # def min_max_transform(self, arr, prefix='lh'):
    #     if prefix == 'lh':
    #         min_val, max_val = self.min_max_lh
    #     else:
    #         min_val, max_val = self.min_max_rh
    #     return arr / max_val


    def min_max_transform(self, arr, prefix='lh'):
        return arr


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.images[idx]
        img = self.load_image(img_path)
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
        }
