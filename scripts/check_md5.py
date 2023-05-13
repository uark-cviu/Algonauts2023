import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
import glob
import hashlib

data_dir = "data"
parent_submission_dir = "predictions"

all_df = []

for subj in [1,2,3,4,5,6,7,8]:

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
    train_image_dir =  os.path.join(args.data_dir, 'training_split', 'training_images')
    test_image_dir =  os.path.join(args.data_dir, 'test_split', 'test_images')
    images = sorted(glob.glob(f"{train_image_dir}/*.png")) + sorted(glob.glob(f"{test_image_dir}/*.png"))
    all_hash = []
    for image in tqdm(images, total=len(images)):
        with open(image,"rb") as f:
            bytes = f.read() # read file as bytes
            readable_hash = hashlib.md5(bytes).hexdigest()
            all_hash.append(readable_hash)

    
    tmp_df = pd.DataFrame({
        'subj': [subj] * len(images),
        'images': images,
        'hash': all_hash
    })
    print(tmp_df['hash'].nunique())

    all_df.append(tmp_df)

all_df = pd.concat(all_df, axis=0)
all_df.to_csv('image_hash.csv', index=False)

