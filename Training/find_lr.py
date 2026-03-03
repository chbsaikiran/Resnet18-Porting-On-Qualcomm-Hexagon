
# import torch
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint

# from config import *
# from datamodule import ImageNetDataModule
# from lightning_module import PruningDistillationModule
# from model import BasicBlock,Bottleneck,ResNet
# import torch.nn as nn
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

from data_module import CIFAR10DataModule
from model_module import ResNet18Lightning
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim

data_module = CIFAR10DataModule(
        batch_size=512,
        num_workers=4
    )

def find_lr(model, train_loader, criterion, optimizer, device,
            start_lr=1e-7, end_lr=1, num_iters=100):

    model.train()
    
    lrs = []
    losses = []
    
    # Reset optimizer lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr

    lr_mult = (end_lr / start_lr) ** (1/num_iters)
    
    iterator = iter(train_loader)
    
    for i in range(num_iters):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, targets = next(iterator)

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        # Increase LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult

        # Stop if loss explodes
        if loss.item() > 10:
            break

    return lrs, losses



model = resnet18(weights=None)

# Modify for CIFAR
model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                        stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

datamodule = CIFAR10DataModule(
    batch_size=512,
    num_workers=4
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(  
    model.parameters(),
    lr=1e-07
)
data_module.setup()
train_loader = data_module.train_dataloader()
lrs, losses = find_lr(model, train_loader, criterion, optimizer, device)
plt.plot(lrs, losses)
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.show()