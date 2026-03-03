# model_module.py
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.models import resnet18


def _rand_bbox(size, lam):
    """Random bounding box for CutMix."""
    _, _, H, W = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    return y1, x1, y2, x2


class ResNet18Lightning(pl.LightningModule):

    def __init__(self, num_classes=10, lr=1e-3, len_train_loader=204,
                 epochs=10, mixup_alpha=1.0, cutmix_alpha=1.0, mix_prob=0.5):
        super().__init__()
        self.lr = lr
        self.len_train_loader = len_train_loader
        self.epochs = epochs
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob

        self.model = resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        if np.random.random() < self.mix_prob:
            rand_index = torch.randperm(images.size(0), device=images.device)

            if np.random.random() < 0.5:
                # ---- Mixup ----
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                images = lam * images + (1.0 - lam) * images[rand_index]
            else:
                # ---- CutMix ----
                lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                y1, x1, y2, x2 = _rand_bbox(images.size(), lam)
                images = images.clone()
                images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]
                lam = 1.0 - (y2 - y1) * (x2 - x1) / (images.size(-2) * images.size(-1))

            logits = self(images)
            loss = (lam * self.criterion(logits, labels)
                    + (1.0 - lam) * self.criterion(logits, labels[rand_index]))
        else:
            logits = self(images)
            loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=self.len_train_loader,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
