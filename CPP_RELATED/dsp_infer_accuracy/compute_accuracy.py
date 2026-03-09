"""
compute_accuracy.py

Reads qnn-net-run output (raw float32 logits) and compares against
ground-truth labels to compute CIFAR-10 top-1 accuracy.

Usage:
    python compute_accuracy.py \
        --output_dir ./qnn_outputs \
        --labels     ./qnn_inputs/labels.txt
"""

import argparse
import os
import glob
import numpy as np

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./qnn_outputs")
    parser.add_argument("--labels", default="./qnn_inputs/labels.txt")
    args = parser.parse_args()

    with open(args.labels, "r") as f:
        gt_labels = [int(line.strip().split()[0]) for line in f.readlines()]

    result_dirs = sorted(glob.glob(os.path.join(args.output_dir, "Result_*")))
    if not result_dirs:
        result_dirs = sorted(glob.glob(os.path.join(args.output_dir, "outputs", "Result_*")))

    if not result_dirs:
        print("No Result_* directories found. Checking for raw files...")
        raw_files = sorted(glob.glob(os.path.join(args.output_dir, "*.raw")))
        if not raw_files:
            print(f"ERROR: No output files found in {args.output_dir}")
            return
        result_dirs = None
        output_files = raw_files
    else:
        output_files = None

    correct = 0
    total = 0
    per_class_correct = [0] * 10
    per_class_total = [0] * 10

    n = len(gt_labels)

    for i in range(n):
        if result_dirs:
            if i >= len(result_dirs):
                break
            raw_files = glob.glob(os.path.join(result_dirs[i], "*.raw"))
            if not raw_files:
                continue
            raw_path = raw_files[0]
        else:
            if i >= len(output_files):
                break
            raw_path = output_files[i]

        logits = np.fromfile(raw_path, dtype=np.float32)
        pred = int(np.argmax(logits))
        gt = gt_labels[i]

        per_class_total[gt] += 1
        if pred == gt:
            correct += 1
            per_class_correct[gt] += 1
        total += 1

    print(f"\n====== CIFAR-10 QNN DSP Inference Results ======")
    print(f"Total images : {total}")
    print(f"Correct      : {correct}")
    print(f"Top-1 Accuracy: {100.0 * correct / total:.2f}%")
    print(f"================================================\n")
    print("Per-class accuracy:")
    for c in range(10):
        if per_class_total[c] > 0:
            acc = 100.0 * per_class_correct[c] / per_class_total[c]
            print(f"  [{c}] {CLASSES[c]:15s} {per_class_correct[c]:4d} / {per_class_total[c]:4d}  ({acc:.1f}%)")


if __name__ == "__main__":
    main()
