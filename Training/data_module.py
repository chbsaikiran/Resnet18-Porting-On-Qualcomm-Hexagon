# datamodule.py

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cv2
import torch

# CIFAR10 mean & std
mean = np.array([0.4914, 0.4822, 0.4465])
std  = np.array([0.2470, 0.2435, 0.2616])


# =====================================================================
#  RandAugment operations – pure OpenCV / numpy, uint8 BGR input
# =====================================================================

def _auto_contrast(img, _m):
    result = np.empty_like(img)
    for c in range(3):
        ch = img[:, :, c]
        lo, hi = int(ch.min()), int(ch.max())
        if lo == hi:
            result[:, :, c] = ch
        else:
            scale = 255.0 / (hi - lo)
            result[:, :, c] = np.clip(
                (ch.astype(np.float32) - lo) * scale, 0, 255
            ).astype(np.uint8)
    return result


def _equalize(img, _m):
    return cv2.merge([cv2.equalizeHist(ch) for ch in cv2.split(img)])


def _invert(img, _m):
    return 255 - img


def _posterize(img, m):
    bits = max(1, 8 - int(round(m * 4.0 / 10.0)))
    shift = 8 - bits
    return (img >> shift) << shift


def _solarize(img, m):
    threshold = int(256 - m * 256.0 / 10.0)
    result = img.copy()
    mask = result >= threshold
    result[mask] = 255 - result[mask]
    return result


def _rotate(img, m):
    deg = m * 30.0 / 10.0 * np.random.choice([-1, 1])
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _shear_x(img, m):
    v = m * 0.3 / 10.0 * np.random.choice([-1, 1])
    h, w = img.shape[:2]
    M = np.float32([[1, v, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _shear_y(img, m):
    v = m * 0.3 / 10.0 * np.random.choice([-1, 1])
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 0], [v, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _translate_x(img, m):
    px = int(m * img.shape[1] * 0.33 / 10.0) * np.random.choice([-1, 1])
    h, w = img.shape[:2]
    M = np.float32([[1, 0, px], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _translate_y(img, m):
    px = int(m * img.shape[0] * 0.33 / 10.0) * np.random.choice([-1, 1])
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 0], [0, 1, px]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _brightness(img, m):
    factor = 1.0 + m * 0.9 / 10.0 * np.random.choice([-1, 1])
    factor = max(0.1, factor)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _contrast(img, m):
    factor = 1.0 + m * 0.9 / 10.0 * np.random.choice([-1, 1])
    factor = max(0.1, factor)
    gray_mean = np.mean(img, dtype=np.float32)
    return np.clip(
        gray_mean + factor * (img.astype(np.float32) - gray_mean), 0, 255
    ).astype(np.uint8)


def _color(img, m):
    factor = 1.0 + m * 0.9 / 10.0 * np.random.choice([-1, 1])
    factor = max(0.1, factor)
    gray = cv2.cvtColor(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
    ).astype(np.float32)
    return np.clip(
        gray + factor * (img.astype(np.float32) - gray), 0, 255
    ).astype(np.uint8)


def _sharpness(img, m):
    factor = 1.0 + m * 0.9 / 10.0 * np.random.choice([-1, 1])
    factor = max(0.1, factor)
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    blurred = cv2.filter2D(img, -1, kernel).astype(np.float32)
    return np.clip(
        blurred + factor * (img.astype(np.float32) - blurred), 0, 255
    ).astype(np.uint8)


def _identity(img, _m):
    return img


_RAND_AUGMENT_OPS = [
    _auto_contrast, _equalize, _invert, _posterize, _solarize,
    _rotate, _shear_x, _shear_y, _translate_x, _translate_y,
    _brightness, _contrast, _color, _sharpness, _identity,
]


def rand_augment(img, num_ops=2, magnitude=9):
    """RandAugment: randomly pick and apply num_ops transforms."""
    chosen = np.random.choice(len(_RAND_AUGMENT_OPS),
                              size=num_ops, replace=False)
    for i in chosen:
        img = _RAND_AUGMENT_OPS[i](img, magnitude)
    return img


# =====================================================================
#  Cutout – applied on the normalised CHW float32 tensor
# =====================================================================

def cutout(img_chw, length=16):
    """Zero out one random square patch of size length x length."""
    _, h, w = img_chw.shape
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(0, cy - length // 2), min(h, cy + length // 2)
    x1, x2 = max(0, cx - length // 2), min(w, cx + length // 2)
    img_chw[:, y1:y2, x1:x2] = 0.0
    return img_chw


# =====================================================================
#  Dataset & DataModule
# =====================================================================

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

        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.train:
            # Random crop with 4px padding
            img = cv2.copyMakeBorder(img, 4, 4, 4, 4,
                                     cv2.BORDER_REFLECT)
            x = np.random.randint(0, 9)
            y = np.random.randint(0, 9)
            img = img[y:y+32, x:x+32]

            # Random horizontal flip
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)

            # RandAugment (OpenCV-based)
            img = rand_augment(img, num_ops=2, magnitude=9)

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # Cutout (training only)
        # if self.train:
        #     img = cutout(img, length=16)

        return torch.tensor(img, dtype=torch.float32), label


class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, batch_size=128, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
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
