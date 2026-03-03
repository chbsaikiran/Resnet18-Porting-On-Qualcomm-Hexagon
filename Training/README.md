# Training — ResNet18 on CIFAR-10

This folder contains everything needed to train a ResNet18 classifier on CIFAR-10, find a good learning rate, export the trained model to ONNX, and visualise training metrics with TensorBoard.

## File Overview

| File | Purpose |
|---|---|
| `train.py` | Main training script. Trains the model with PyTorch Lightning, saves `.pth` checkpoints and logs metrics to TensorBoard. |
| `model_module.py` | Lightning module wrapping `torchvision.models.resnet18`, modified for CIFAR-10 (3x3 conv1, no maxpool, 10-class FC). Includes Mixup and CutMix augmentation in the training step. |
| `data_module.py` | Lightning data module with a custom `CIFAR10_OpenCV` dataset. All preprocessing and augmentation uses OpenCV/numpy to match the C++ inference pipeline. |
| `find_lr.py` | Learning rate range test — sweeps LR from 1e-7 to 1 and plots loss vs. LR to find the optimal starting rate. |
| `onnx_dump.py` | Exports the trained full model (`.pth`) to ONNX format for deployment on x86/ARM/DSP targets. |

## Data Augmentations

Applied during training only, all implemented with OpenCV:

**Per-image (in `data_module.py`):**

- Random crop with 4px reflect-padding
- Random horizontal flip
- **RandAugment** (2 random ops at magnitude 9) — includes rotate, shear, translate, brightness, contrast, color saturation, sharpness, auto-contrast, equalize, posterize, solarize, invert
- **Cutout** (16x16 zero-patch on the normalised tensor)

**Per-batch (in `model_module.py`):**

- **Mixup** — pixel-wise blend of two samples, with mixed cross-entropy loss
- **CutMix** — rectangular patch swap between two samples, loss weighted by area

## How to Train

```bash
cd Training
python train.py
```

Training parameters are set at the top of `train.py`:

- `batch_size = 512`
- `epochs = 10`
- `lr = 0.004` (with OneCycleLR scheduler)

The script saves two files on completion:

- `resnet18_cifar10_full.pth` — full pickled model (used by `onnx_dump.py`)
- `resnet18_cifar10.pth` — state dict only

## Viewing TensorBoard Plots

Training automatically logs the following metrics to TensorBoard:

- `train_loss` / `val_loss`
- `train_acc` / `val_acc`
- `lr-SGD` (learning rate at each step, via `LearningRateMonitor`)

### Start TensorBoard

After training has started (or finished), run:

```bash
cd Training
tensorboard --logdir tb_logs
```

Then open **http://localhost:6006** in your browser.

### TensorBoard Tabs

| Tab | What you'll see |
|---|---|
| **Scalars** | Loss curves, accuracy curves, and learning rate schedule |
| **Graphs** | The model computation graph (if exported) |
| **Hparams** | Hyperparameters saved by Lightning |

### Comparing Runs

Each training run creates a new versioned sub-directory under `tb_logs/`. TensorBoard automatically overlays all runs so you can compare experiments side by side. To filter runs, use the checkboxes in the left sidebar.

### Remote Server

If training on a remote machine, forward the port via SSH:

```bash
ssh -L 6006:localhost:6006 user@remote-host
```

Then open **http://localhost:6006** locally.

## Exporting to ONNX

After training completes:

```bash
python onnx_dump.py
```

This produces `resnet18_cifar10.onnx` (opset 13, input shape `1x3x32x32`) which can be consumed by the C++ inference code in `CPP_RELATED/`.

## Finding the Optimal Learning Rate

```bash
python find_lr.py
```

Runs a short sweep over 100 iterations, plotting loss vs. learning rate on a log scale. Pick the LR just before the loss starts to rise steeply.
