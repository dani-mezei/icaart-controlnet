import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _list_image_files(directory):
    return sorted(
        filename
        for filename in os.listdir(directory)
        if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
    )


class KittiDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        image_transform=None,
        label_transform=None,
        additional_image_dirs=None,
        additional_label_dirs=None,
        negative_image_dirs=None,
        negative_label_dirs=None,
        max_samples=None,
    ):
        super().__init__()

        self.samples = []

        image_filenames = _list_image_files(image_dir)
        label_filenames = _list_image_files(label_dir)

        for img, lbl in zip(image_filenames, label_filenames):
            label_path = os.path.join(label_dir, lbl)
            image_path = os.path.join(image_dir, img)

            self.samples.append((image_path, label_path))

        if additional_image_dirs and additional_label_dirs:
            for img_dir, lbl_dir in zip(additional_image_dirs, additional_label_dirs):
                imgs = _list_image_files(img_dir)
                lbls = _list_image_files(lbl_dir)
                for img, lbl in zip(imgs, lbls):
                    lbl_path = os.path.join(lbl_dir, lbl)
                    img_path = os.path.join(img_dir, img)

                    self.samples.append((img_path, lbl_path))

        if negative_image_dirs and negative_label_dirs:
            for neg_img_dir, neg_lbl_dir in zip(negative_image_dirs, negative_label_dirs):
                neg_imgs = _list_image_files(neg_img_dir)
                neg_lbls = _list_image_files(neg_lbl_dir)
                print(f"Length before removal: {len(self.samples)}")
                
                current_img_names = {os.path.basename(img_path) for img_path, _ in self.samples}
                current_lbl_names = {os.path.basename(lbl_path) for _, lbl_path in self.samples}
                
                img_to_remove = set(neg_imgs) & current_img_names
                lbl_to_remove = set(neg_lbls) & current_lbl_names
                
                self.samples = [
                    (img_path, lbl_path) 
                    for img_path, lbl_path in self.samples 
                    if os.path.basename(img_path) not in img_to_remove 
                    and os.path.basename(lbl_path) not in lbl_to_remove
                ]
                
                print(f"Length after removal: {len(self.samples)}")

        self.image_transform = image_transform
        self.label_transform = label_transform

        if max_samples:
            if len(self.samples) > max_samples:
                self.samples = random.sample(self.samples, max_samples)

        print(f"Number of samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Failed to read label: {label_path}")

        if self.image_transform:
            image = self.image_transform(image)

        if self.label_transform:
            label = self.label_transform(label)

        label = torch.tensor(np.array(label), dtype=torch.int64)

        return image, label
