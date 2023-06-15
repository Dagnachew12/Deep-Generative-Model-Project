from abc import ABC
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2


class CustomDataset(Dataset, ABC):
    def __init__(self, MRI_path, CT_path, transform=None):
        self.mri_path = MRI_path
        self.ct_path = CT_path
        self.transform = transform

        self.mri_images = os.listdir(MRI_path)
        self.ct_images = os.listdir(CT_path)
        self.dataset_size = max(len(self.mri_images), len(self.ct_images))
        self.mri_len = len(self.mri_images)
        self.ct_len = len(self.ct_images)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        Mri_image = self.mri_images[idx % self.mri_len]
        CT_image = self.ct_images[idx % self.ct_len]

        mri_path = os.path.join(self.mri_path, Mri_image)
        ct_path = os.path.join(self.ct_path, CT_image)
        mri_image = np.array(Image.open(mri_path).convert('RGB'))
        ct_image = np.array(Image.open(ct_path).convert('RGB'))
        if self.transform:
            mri_image = self.transform(mri_image)
            ct_image = self.transform(ct_image)

        return mri_image, ct_image
