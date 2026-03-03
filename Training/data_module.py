# datamodule.py

import math
import pytorch_lightning as pl
from torch.utils.data import Dataset,DataLoader
import torchvision
import numpy as np
import cv2
import torch

# CIFAR10 mean & std
mean = np.array([0.4914, 0.4822, 0.4465])
std  = np.array([0.2470, 0.2435, 0.2616])


class CIFAR10_OpenCV(Dataset):
    def __init__(self, train=True):
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=train,
            download=True
        )
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # PIL → numpy
        img = np.array(img)

        # OpenCV uses BGR, convert if needed
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.train:
            # Random crop with padding
            img = cv2.copyMakeBorder(img, 4, 4, 4, 4,
                                     cv2.BORDER_REFLECT)

            x = np.random.randint(0, 9)
            y = np.random.randint(0, 9)
            img = img[y:y+32, x:x+32]

            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)

        # Convert back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std

        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img, dtype=torch.float32), label


class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, batch_size=128, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Create datasets
        self.train_dataset = CIFAR10_OpenCV(train=True)
        self.val_dataset   = CIFAR10_OpenCV(train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
