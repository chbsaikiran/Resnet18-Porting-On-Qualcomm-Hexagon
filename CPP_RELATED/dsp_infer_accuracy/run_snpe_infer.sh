#!/bin/bash
# ======================================================================
# run_snpe_infer.sh  –  Run ResNet18 inference on device via SNPE
#
# Prerequisites:
#   1. Run deploy_snpe.sh first (pushes snpe-net-run + DLC + libs)
#   2. CIFAR-10 test batch under Training/data/
#
# Usage:
#   ./run_snpe_infer.sh [NUM_IMAGES] [RUNTIME]
#
#   NUM_IMAGES  – how many test images (default 100)
#   RUNTIME     – dsp (default) | cpu | gpu
#
# Note: SNPE expects NCHW input (unlike QNN NHWC). prepare_cifar10_inputs_snpe.py
# outputs NCHW layout.
#
# DSP runtime: Float DLC may run on DSP; if it fails, use cpu or gpu. For optimal
# DSP performance, quantize the DLC with snpe-dlc-quantize first.
# ======================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR=/data/local/tmp/snpe_resnet18
INPUT_DIR="${SCRIPT_DIR}/snpe_inputs"
OUTPUT_LOCAL="${SCRIPT_DIR}/snpe_outputs"
NUM_IMAGES=${1:-1000}
RUNTIME=${2:-dsp}

# --- Step 1: Prepare inputs (NCHW for SNPE) ---------------------------
echo "=== Step 1: Preparing CIFAR-10 inputs (${NUM_IMAGES} images, NCHW) ==="
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
    echo "ERROR: CIFAR-10 test batch not found."
    exit 1
fi
echo "  Using: ${TEST_BATCH}"

rm -rf "${INPUT_DIR}"
conda run -n mlpy310env python "${SCRIPT_DIR}/prepare_cifar10_inputs_snpe.py" \
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
echo "=== Step 3: Running SNPE inference ==="
echo "  Runtime: ${RUNTIME}"
echo "  Model:   resnet18_cifar10_quantized.dlc"
echo "  Images:  ${NUM_IMAGES}"
echo ""

# Quick sanity check that snpe-net-run exists on device
if ! adb shell "[ -f ${DEVICE_DIR}/snpe-net-run ]" >/dev/null 2>&1; then
    echo "ERROR: ${DEVICE_DIR}/snpe-net-run not found on device."
    echo "       Run ./deploy_snpe.sh first (and ensure it succeeds)."
    exit 1
fi

# DSP library path for Hexagon skel loading
DSP_LIB_PATH="${DEVICE_DIR};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"

# Build snpe-net-run arguments
SNPE_ARGS="--container ${DEVICE_DIR}/resnet18_cifar10_quantized.dlc"
SNPE_ARGS="${SNPE_ARGS} --input_list ${DEVICE_DIR}/inputs/input_list.txt"
SNPE_ARGS="${SNPE_ARGS} --userlogs verbose"
SNPE_ARGS="${SNPE_ARGS} --output_dir ${DEVICE_DIR}/outputs"
#SNPE_ARGS="${SNPE_ARGS} --userbuffer_float"
SNPE_ARGS="${SNPE_ARGS} --platform_options=\"unsignedPD:OFF\""

case "$RUNTIME" in
    dsp) SNPE_ARGS="${SNPE_ARGS} --use_dsp" ;;
    gpu) SNPE_ARGS="${SNPE_ARGS} --use_gpu" ;;
    cpu) ;;
    *)   echo "ERROR: Unknown runtime '${RUNTIME}'. Use: cpu | gpu | dsp"; exit 1 ;;
esac

set +e
adb shell "export LD_LIBRARY_PATH=${DEVICE_DIR} && \
    export ADSP_LIBRARY_PATH='${DSP_LIB_PATH}' && \
    export CDSP_LIBRARY_PATH='${DSP_LIB_PATH}' && \
    ${DEVICE_DIR}/snpe-net-run ${SNPE_ARGS}"
RUN_EXIT=$?
set -e
if [ $RUN_EXIT -ne 0 ]; then
    echo ""
    echo "WARNING: snpe-net-run exited with code ${RUN_EXIT}. Results may still be valid."
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
