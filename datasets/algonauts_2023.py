import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import KFold
import glob
import os

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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.images[idx]
        img = self.load_image(img_path)
        lh_fmri = self.load_fmri(img_path, prefix='lh')
        rh_fmri = self.load_fmri(img_path, prefix='rh')
        if self.transform:
            img = self.transform(img)
        return {
            "image": img,
            "lh_fmri": lh_fmri,
            "rh_fmri": rh_fmri
        }