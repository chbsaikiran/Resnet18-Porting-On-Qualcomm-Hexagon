# train.py

import argparse
import glob
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from data_module import CIFAR10DataModule, mean as CIFAR_MEAN, std as CIFAR_STD
from model_module import ResNet18Lightning

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def preview_augmented_samples(data_module, n=10, save_path="augmented_preview.png"):
    """Grab n images from the train loader and save a visual grid."""
    data_module.setup()
    loader = data_module.train_dataloader()
    images, labels = next(iter(loader))

    images = images[:n].numpy()
    labels = labels[:n].numpy()

    mean = CIFAR_MEAN.reshape(3, 1, 1)
    std  = CIFAR_STD.reshape(3, 1, 1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        img = images[i] * std + mean
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))

        ax.imshow(img)
        ax.set_title(CLASS_NAMES[labels[i]], fontsize=12)
        ax.axis("off")

    fig.suptitle("Augmented training samples (after all transforms + cutout)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[preview] Saved {n} augmented samples → {save_path}")


def find_latest_checkpoint(search_dirs=("tb_logs", "lightning_logs")):
    """Find the most recent .ckpt file across logger/checkpoint directories."""
    ckpts = []
    for d in search_dirs:
        ckpts.extend(glob.glob(os.path.join(d, "**", "*.ckpt"), recursive=True))
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help="Resume from checkpoint. Pass a .ckpt path, or just --resume to auto-detect the latest one.",
    )
    args = parser.parse_args()

    ckpt_path = None
    if args.resume is not None:
        if args.resume == "auto":
            ckpt_path = find_latest_checkpoint()
            if ckpt_path is None:
                print("[resume] No checkpoint found, starting from scratch.")
            else:
                print(f"[resume] Auto-detected checkpoint → {ckpt_path}")
        else:
            ckpt_path = args.resume
            print(f"[resume] Using checkpoint → {ckpt_path}")

    batch_size = 512
    epochs = 20
    cifar10_train_size = 50000

    data_module = CIFAR10DataModule(
        batch_size=batch_size,
        num_workers=4
    )

    preview_augmented_samples(data_module, n=10)

    model = ResNet18Lightning(
        num_classes=10,
        lr=0.004,
        len_train_loader=cifar10_train_size // batch_size,
        epochs=epochs
    )

    tb_logger = TensorBoardLogger(
        save_dir=".",
        name="tb_logs",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     mode="min",
    #     patience=3,
    #     min_delta=0.001,
    #     verbose=True
    # )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    torch.save(model, "resnet18_cifar10_full.pth")
    torch.save(model.state_dict(), "resnet18_cifar10.pth")


if __name__ == "__main__":
    main()
