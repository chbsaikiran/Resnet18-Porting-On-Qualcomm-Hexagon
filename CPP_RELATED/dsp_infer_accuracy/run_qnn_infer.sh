#!/bin/bash
# ======================================================================
# run_qnn_infer.sh  –  Run ResNet18 inference on device via QNN
#
# Prerequisites:
#   1. Run deploy_qnn.sh first (pushes qnn-net-run + model + libs)
#   2. CIFAR-10 test_batch.bin available under Training/data/
#
# Usage:
#   ./run_qnn_infer.sh [NUM_IMAGES] [BACKEND]
#
#   NUM_IMAGES  – how many test images (default 100)
#   BACKEND     – libQnnDsp.so (default, runs on Hexagon DSP)
#                 libQnnCpu.so | libQnnGpu.so
# ======================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR=/data/local/tmp/qnn_resnet18
INPUT_DIR="${SCRIPT_DIR}/qnn_inputs"
OUTPUT_LOCAL="${SCRIPT_DIR}/qnn_outputs"
NUM_IMAGES=${1:-100}
BACKEND=${2:-libQnnDsp.so}

# --- Step 1: Prepare inputs locally ------------------------------------
echo "=== Step 1: Preparing CIFAR-10 inputs (${NUM_IMAGES} images) ==="
TEST_BATCH=""
for candidate in \
    "${SCRIPT_DIR}/../../Training/data/cifar-10-batches-py/test_batch" \
    "${SCRIPT_DIR}/../../Training/data/cifar-10-batches-bin/test_batch.bin" \
    "${SCRIPT_DIR}/../../../Training/data/cifar-10-batches-py/test_batch" \
    "${SCRIPT_DIR}/../../../Training/data/cifar-10-batches-bin/test_batch.bin"; do
    if [ -f "$candidate" ]; then
        TEST_BATCH="$candidate"
        break
    fi
done
if [ -z "${TEST_BATCH}" ]; then
    echo "ERROR: CIFAR-10 test batch not found. Searched for:"
    echo "  .../cifar-10-batches-py/test_batch  (pickle format)"
    echo "  .../cifar-10-batches-bin/test_batch.bin  (binary format)"
    exit 1
fi
echo "  Using: ${TEST_BATCH}"

rm -rf "${INPUT_DIR}"
conda run -n mlpy310env python "${SCRIPT_DIR}/prepare_cifar10_inputs.py" \
    --test_batch "${TEST_BATCH}" \
    --output_dir "${INPUT_DIR}" \
    --num_images "${NUM_IMAGES}"

# --- Step 2: Push inputs to device -------------------------------------
echo ""
echo "=== Step 2: Pushing inputs to device ==="
adb shell "rm -rf ${DEVICE_DIR}/inputs ${DEVICE_DIR}/outputs"
adb shell "mkdir -p ${DEVICE_DIR}/inputs"
adb push "${INPUT_DIR}/input_list.txt" "${DEVICE_DIR}/inputs/"
for f in "${INPUT_DIR}"/img_*.raw; do
    [ -f "$f" ] && adb push "$f" "${DEVICE_DIR}/inputs/" 2>/dev/null
done
echo "  Pushed ${NUM_IMAGES} input files"

# --- Step 3: Run inference on device ------------------------------------
echo ""
echo "=== Step 3: Running QNN inference ==="
echo "  Backend: ${BACKEND}"
echo "  Model:   libresnet18_cifar10.so"
echo "  Images:  ${NUM_IMAGES}"
echo ""

# ADSP_LIBRARY_PATH / CDSP_LIBRARY_PATH tell the FastRPC loader where to
# find the DSP skel libraries (libQnnDspV66Skel.so, libQnnDspV66.so).
# SDM710 cDSP (Hexagon 685) — set both paths to be safe.
DSP_LIB_PATH="${DEVICE_DIR};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"

set +e
adb shell "cd ${DEVICE_DIR}/inputs && \
    export LD_LIBRARY_PATH=${DEVICE_DIR} && \
    export ADSP_LIBRARY_PATH='${DSP_LIB_PATH}' && \
    export CDSP_LIBRARY_PATH='${DSP_LIB_PATH}' && \
    ${DEVICE_DIR}/qnn-net-run \
        --model ${DEVICE_DIR}/libresnet18_cifar10.so \
        --backend ${DEVICE_DIR}/${BACKEND} \
        --input_list ${DEVICE_DIR}/inputs/input_list.txt \
        --output_dir ${DEVICE_DIR}/outputs \
        --log_level verbose"
RUN_EXIT=$?
set -e
if [ $RUN_EXIT -ne 0 ]; then
    echo ""
    echo "WARNING: qnn-net-run exited with code ${RUN_EXIT} (segfault during teardown is expected — results may still be valid)"
fi

# --- Step 4: Pull results -----------------------------------------------
echo ""
echo "=== Step 4: Pulling results ==="
rm -rf "${OUTPUT_LOCAL}"
mkdir -p "${OUTPUT_LOCAL}"
adb pull "${DEVICE_DIR}/outputs/" "${OUTPUT_LOCAL}/" 2>/dev/null || true

# --- Step 5: Compute accuracy -------------------------------------------
echo ""
echo "=== Step 5: Computing accuracy ==="
conda run -n mlpy310env python "${SCRIPT_DIR}/compute_accuracy.py" \
    --output_dir "${OUTPUT_LOCAL}" \
    --labels "${INPUT_DIR}/labels.txt"
