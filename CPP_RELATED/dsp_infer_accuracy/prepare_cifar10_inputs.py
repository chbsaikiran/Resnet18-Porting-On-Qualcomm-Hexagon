"""
prepare_cifar10_inputs.py

Reads a CIFAR-10 test batch (binary *or* Python-pickle format) and writes
each image as a raw float32 file that qnn-net-run can consume.

Output layout per file: HWC  (32, 32, 3) — QNN expects NHWC; the batch
dimension is implicit (one image per file).
Normalised with CIFAR-10 mean/std.

Usage:
    python prepare_cifar10_inputs.py \
        --test_batch ../../../Training/data/cifar-10-batches-py/test_batch \
        --output_dir ./qnn_inputs \
        --num_images 100
"""

import argparse
import os
import pickle
import numpy as np

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _normalise(img_chw_uint8):
    """uint8 CHW → float32 HWC, normalised."""
    img = img_chw_uint8.astype(np.float32) / 255.0
    for c in range(3):
        img[c] = (img[c] - CIFAR10_MEAN[c]) / CIFAR10_STD[c]
    return img.transpose(1, 2, 0)  # CHW → HWC


def load_cifar10_pickle(path):
    """Load the Python-pickle format (cifar-10-batches-py/test_batch)."""
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    raw_labels = batch[b"labels"]
    raw_data = batch[b"data"]  # (N, 3072) uint8

    images, labels = [], []
    for i in range(len(raw_labels)):
        pixels = raw_data[i].reshape(3, 32, 32)
        images.append(_normalise(pixels))
        labels.append(raw_labels[i])
    return images, labels


def load_cifar10_binary(path):
    """Load the raw-binary format (cifar-10-batches-bin/test_batch.bin)."""
    with open(path, "rb") as f:
        data = f.read()

    num_images = len(data) // 3073
    images, labels = [], []
    for i in range(num_images):
        offset = i * 3073
        label = data[offset]
        pixels = np.frombuffer(data[offset + 1 : offset + 3073], dtype=np.uint8)
        images.append(_normalise(pixels.reshape(3, 32, 32)))
        labels.append(label)
    return images, labels


def load_cifar10(path):
    """Auto-detect format: try pickle first, fall back to raw binary."""
    try:
        return load_cifar10_pickle(path)
    except (pickle.UnpicklingError, KeyError, EOFError):
        return load_cifar10_binary(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch", required=True)
    parser.add_argument("--output_dir", default="./qnn_inputs")
    parser.add_argument("--num_images", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    images, labels = load_cifar10(args.test_batch)
    n = min(args.num_images, len(images))

    input_list_path = os.path.join(args.output_dir, "input_list.txt")
    labels_path = os.path.join(args.output_dir, "labels.txt")

    with open(input_list_path, "w") as f_list, \
         open(labels_path, "w") as f_labels:

        for i in range(n):
            raw_path = os.path.join(args.output_dir, f"img_{i:05d}.raw")
            images[i].tofile(raw_path)

            f_list.write(f"img_{i:05d}.raw\n")
            f_labels.write(f"{labels[i]} {CLASSES[labels[i]]}\n")

    print(f"Prepared {n} images in {args.output_dir}/")
    print(f"  input_list.txt : {n} entries")
    print(f"  labels.txt     : ground truth")
    print(f"  img_XXXXX.raw  : float32 NHWC (1x32x32x3 = 12288 bytes each)")


if __name__ == "__main__":
    main()
