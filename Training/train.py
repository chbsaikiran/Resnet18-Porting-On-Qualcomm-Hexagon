# train.py

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data_module import CIFAR10DataModule
from model_module import ResNet18Lightning


def main():

    batch_size = 64
    epochs = 10
    cifar10_train_size = 50000

    data_module = CIFAR10DataModule(
        batch_size=batch_size,
        num_workers=4
    )

    model = ResNet18Lightning(
        num_classes=10,
        lr=0.004,
        len_train_loader=cifar10_train_size // batch_size,
        epochs=epochs
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        min_delta=0.001,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(model, datamodule=data_module)

    torch.save(model, "resnet18_cifar10_full.pth")
    torch.save(model.state_dict(), "resnet18_cifar10.pth")


if __name__ == "__main__":
    main()
