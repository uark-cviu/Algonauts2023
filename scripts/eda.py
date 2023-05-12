import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr


data_dir = "data"
parent_submission_dir = "predictions"

subj_dict = {}
for subj in [1, 2, 3, 4, 5, 6, 7, 8]:

    class argObj:
        def __init__(self, data_dir, parent_submission_dir, subj):
            self.subj = format(subj, "02")
            self.subj_ = "subj" + self.subj
            self.data_dir = os.path.join(data_dir, "subj" + self.subj)
            self.parent_submission_dir = parent_submission_dir
            self.subject_submission_dir = os.path.join(
                self.parent_submission_dir, "subj" + self.subj
            )

            # Create the submission directory if not existing
            if not os.path.isdir(self.subject_submission_dir):
                os.makedirs(self.subject_submission_dir)

    args = argObj(data_dir, parent_submission_dir, subj)

    subj_dict[args.subj_] = {}
    for hemisphere in ["left", "right"]:
        subj_dict[args.subj_][hemisphere[0]] = {}
        for roi in [
            "V1v",
            "V1d",
            "V2v",
            "V2d",
            "V3v",
            "V3d",
            "hV4",
            "EBA",
            "FBA-1",
            "FBA-2",
            "mTL-bodies",
            "OFA",
            "FFA-1",
            "FFA-2",
            "mTL-faces",
            "aTL-faces",
            "OPA",
            "PPA",
            "RSC",
            "OWFA",
            "VWFA-1",
            "VWFA-2",
            "mfs-words",
            "mTL-words",
            "early",
            "midventral",
            "midlateral",
            "midparietal",
            "ventral",
            "lateral",
            "parietal",
        ]:
            # Define the ROI class based on the selected ROI
            if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
                roi_class = "prf-visualrois"
            elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
                roi_class = "floc-bodies"
            elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
                roi_class = "floc-faces"
            elif roi in ["OPA", "PPA", "RSC"]:
                roi_class = "floc-places"
            elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
                roi_class = "floc-words"
            elif roi in [
                "early",
                "midventral",
                "midlateral",
                "midparietal",
                "ventral",
                "lateral",
                "parietal",
            ]:
                roi_class = "streams"

            # Load the ROI brain surface maps
            challenge_roi_class_dir = os.path.join(
                args.data_dir,
                "roi_masks",
                hemisphere[0] + "h." + roi_class + "_challenge_space.npy",
            )
            fsaverage_roi_class_dir = os.path.join(
                args.data_dir,
                "roi_masks",
                hemisphere[0] + "h." + roi_class + "_fsaverage_space.npy",
            )
            roi_map_dir = os.path.join(
                args.data_dir, "roi_masks", "mapping_" + roi_class + ".npy"
            )
            challenge_roi_class = np.load(challenge_roi_class_dir)
            fsaverage_roi_class = np.load(fsaverage_roi_class_dir)
            roi_map = np.load(roi_map_dir, allow_pickle=True).item()

            # Select the vertices corresponding to the ROI of interest
            roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
            challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
            fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)

            print(f"roi: {roi}, shape: ", challenge_roi.sum())
            subj_dict[args.subj_][hemisphere[0]][roi] = challenge_roi

import pickle

print(subj_dict)
with open("subject_meta.pkl", "wb") as f:
    pickle.dump(subj_dict, f)
