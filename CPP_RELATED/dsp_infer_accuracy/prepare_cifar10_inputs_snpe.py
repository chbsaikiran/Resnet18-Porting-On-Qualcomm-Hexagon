"""
prepare_cifar10_inputs_snpe.py

Reads CIFAR-10 test batch and writes each image as raw float32 NCHW files
for snpe-net-run. SNPE DLC converted from ONNX expects input [1,3,32,32]
NCHW format.

Output: float32 NCHW (3, 32, 32) = 3072 floats = 12288 bytes per file.
Normalised with CIFAR-10 mean/std.

Usage:
    python prepare_cifar10_inputs_snpe.py \
        --test_batch ../../Training/data/cifar-10-batches-py/test_batch \
        --output_dir ./snpe_inputs \
        --num_images 100
"""

import argparse
import os
import pickle
import numpy as np

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _normalise_chw(img_chw_uint8):
    """uint8 CHW → float32 CHW, normalised (keep NCHW for SNPE)."""
    img = img_chw_uint8.astype(np.float32) / 255.0
    for c in range(3):
        img[c] = (img[c] - CIFAR10_MEAN[c]) / CIFAR10_STD[c]
    return img  # CHW preserved


def load_cifar10_pickle(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    raw_labels = batch[b"labels"]
    raw_data = batch[b"data"]
    images = []
    labels = []
    for i in range(len(raw_labels)):
        pixels = raw_data[i].reshape(3, 32, 32)
        images.append(_normalise_chw(pixels))
        labels.append(raw_labels[i])
    return images, labels


def load_cifar10_binary(path):
    with open(path, "rb") as f:
        data = f.read()
    num_images = len(data) // 3073
    images, labels = [], []
    for i in range(num_images):
        offset = i * 3073
        label = data[offset]
        pixels = np.frombuffer(data[offset + 1 : offset + 3073], dtype=np.uint8)
        images.append(_normalise_chw(pixels.reshape(3, 32, 32)))
        labels.append(label)
    return images, labels


def load_cifar10(path):
    try:
        return load_cifar10_pickle(path)
    except (pickle.UnpicklingError, KeyError, EOFError):
        return load_cifar10_binary(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch", required=True)
    parser.add_argument("--output_dir", default="./snpe_inputs")
    parser.add_argument("--num_images", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images, labels = load_cifar10(args.test_batch)
    n = min(args.num_images, len(images))

    input_list_path = os.path.join(args.output_dir, "input_list.txt")
    labels_path = os.path.join(args.output_dir, "labels.txt")

    with open(input_list_path, "w") as f_list, open(labels_path, "w") as f_labels:
        for i in range(n):
            raw_path = os.path.join(args.output_dir, f"img_{i:05d}.raw")
            images[i].tofile(raw_path)
            f_list.write(f"/data/local/tmp/snpe_resnet18/inputs/img_{i:05d}.raw\n")
            f_labels.write(f"{labels[i]} {CLASSES[labels[i]]}\n")

    print(f"Prepared {n} images in {args.output_dir}/")
    print(f"  input_list.txt : {n} entries")
    print(f"  labels.txt     : ground truth")
    print(f"  img_XXXXX.raw  : float32 NCHW (3x32x32 = 12288 bytes each)")


if __name__ == "__main__":
    main()
