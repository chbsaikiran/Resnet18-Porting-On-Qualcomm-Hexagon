#!/bin/bash
# ======================================================================
# run_qnn_infer.sh  –  Run ResNet18 inference on SDM710 DSP via QNN
#
# Prerequisites:
#   1. Run deploy_qnn.sh (pushes binaries to device)
#   2. Run prepare_cifar10_inputs.py (creates raw input files)
#   3. Push qnn_inputs/ to device
#
# SDM710 → Hexagon 685 (v66) → QnnDsp backend
# ======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR=/data/local/tmp/qnn_resnet18
INPUT_DIR=${SCRIPT_DIR}/qnn_inputs
NUM_IMAGES=${1:-100}

# --- Step 1: Prepare inputs locally ------------------------------------
echo "=== Step 1: Preparing CIFAR-10 inputs ==="
TEST_BATCH="${SCRIPT_DIR}/../../Training/data/cifar-10-batches-bin/test_batch.bin"
if [ ! -f "${TEST_BATCH}" ]; then
    TEST_BATCH="${SCRIPT_DIR}/../../../Training/data/cifar-10-batches-bin/test_batch.bin"
fi

conda run -n mlpy310env python "${SCRIPT_DIR}/prepare_cifar10_inputs.py" \
    --test_batch "${TEST_BATCH}" \
    --output_dir "${INPUT_DIR}" \
    --num_images "${NUM_IMAGES}"

# --- Step 2: Push inputs to device -------------------------------------
echo ""
echo "=== Step 2: Pushing inputs to device ==="
adb shell "mkdir -p ${DEVICE_DIR}/inputs"
adb push "${INPUT_DIR}/input_list.txt" "${DEVICE_DIR}/inputs/"
for f in "${INPUT_DIR}"/img_*.raw; do
    adb push "$f" "${DEVICE_DIR}/inputs/" 2>/dev/null
done
echo "  Pushed ${NUM_IMAGES} input files"

# --- Step 3: Run inference on device ------------------------------------
echo ""
echo "=== Step 3: Running QNN inference on DSP ==="
echo "  Backend: QnnDsp (Hexagon v66)"
echo "  Model:   libresnet18_cifar10.so"
echo ""

# SDM710 Hexagon 685 = DSP V65 — QAIRT 2.31.0 only ships V66+.
# Use GPU (best perf on this device) or CPU as backend.
BACKEND=${BACKEND:-libQnnGpu.so}

adb shell "cd ${DEVICE_DIR}/inputs && \
    LD_LIBRARY_PATH=${DEVICE_DIR} \
    ADSP_LIBRARY_PATH=${DEVICE_DIR} \
    ${DEVICE_DIR}/qnn-net-run \
        --model ${DEVICE_DIR}/libresnet18_cifar10.so \
        --backend ${DEVICE_DIR}/${BACKEND} \
        --input_list ${DEVICE_DIR}/inputs/input_list.txt \
        --output_dir ${DEVICE_DIR}/outputs"

# --- Step 4: Pull results and compute accuracy --------------------------
echo ""
echo "=== Step 4: Pulling results ==="
mkdir -p "${SCRIPT_DIR}/qnn_outputs"
adb pull "${DEVICE_DIR}/outputs/" "${SCRIPT_DIR}/qnn_outputs/" 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "Results pulled to ${SCRIPT_DIR}/qnn_outputs/"
echo "Run: python compute_accuracy.py to calculate top-1 accuracy"
