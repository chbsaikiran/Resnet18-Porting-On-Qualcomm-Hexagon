#!/bin/bash
# ======================================================================
# run_demo_snpe.sh — Prepare one CIFAR-10 SNPE input, push, run cifar10_snpe_demo
#
# Prerequisites:
#   - ./deploy_demo_snpe.sh (binary + DLC + libs on device)
#   - CIFAR-10 test batch under Training/data/ (read-only; uses sibling script)
#   - conda env mlpy310env (same as dsp_infer_accuracy)
#
# Usage:
#   ./run_demo_snpe.sh [dsp|gpu|cpu]
#
# Does not modify dsp_infer_accuracy/ — only invokes its Python script.
# ======================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR=/data/local/tmp/demo_snpe_app
LOCAL_INPUT_DIR="${SCRIPT_DIR}/demo_snpe_inputs"
RUNTIME="${1:-dsp}"

PREP_PY="${SCRIPT_DIR}/../dsp_infer_accuracy/prepare_cifar10_inputs_snpe.py"
if [[ ! -f "${PREP_PY}" ]]; then
    echo "ERROR: Expected ${PREP_PY}"
    exit 1
fi

TEST_BATCH=""
for candidate in \
    "${SCRIPT_DIR}/../../Training/data/cifar-10-batches-py/test_batch" \
    "${SCRIPT_DIR}/../../Training/data/cifar-10-batches-bin/test_batch.bin" \
    "${SCRIPT_DIR}/../../../Training/data/cifar-10-batches-py/test_batch" \
    "${SCRIPT_DIR}/../../../Training/data/cifar-10-batches-bin/test_batch.bin"; do
    if [[ -f "${candidate}" ]]; then
        TEST_BATCH="${candidate}"
        break
    fi
done
if [[ -z "${TEST_BATCH}" ]]; then
    echo "ERROR: CIFAR-10 test batch not found under Training/data/"
    exit 1
fi

rm -rf "${LOCAL_INPUT_DIR}"
mkdir -p "${LOCAL_INPUT_DIR}"

echo "=== Preparing 1× SNPE NCHW input (via dsp_infer_accuracy script) ==="
conda run -n mlpy310env python "${PREP_PY}" \
    --test_batch "${TEST_BATCH}" \
    --output_dir "${LOCAL_INPUT_DIR}" \
    --num_images 1

RAW_LOCAL="${LOCAL_INPUT_DIR}/img_00000.raw"
if [[ ! -f "${RAW_LOCAL}" ]]; then
    echo "ERROR: ${RAW_LOCAL} not created"
    exit 1
fi

# Rewrite input_list for our device path (prepare script hardcodes snpe_resnet18 paths)
DEVICE_INPUT="${DEVICE_DIR}/inputs/img_00000.raw"
adb shell "rm -rf ${DEVICE_DIR}/inputs && mkdir -p ${DEVICE_DIR}/inputs"
adb push "${RAW_LOCAL}" "${DEVICE_INPUT}"

if ! adb shell "test -f ${DEVICE_DIR}/cifar10_snpe_demo"; then
    echo "ERROR: ${DEVICE_DIR}/cifar10_snpe_demo missing. Run ./deploy_demo_snpe.sh"
    exit 1
fi

DSP_LIB_PATH="${DEVICE_DIR};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"

echo ""
echo "=== Running cifar10_snpe_demo on device (runtime=${RUNTIME}) ==="
set +e
adb shell "export LD_LIBRARY_PATH=${DEVICE_DIR} && \
    export ADSP_LIBRARY_PATH='${DSP_LIB_PATH}' && \
    export CDSP_LIBRARY_PATH='${DSP_LIB_PATH}' && \
    ${DEVICE_DIR}/cifar10_snpe_demo ${DEVICE_INPUT} ${RUNTIME}"
RC=$?
set -e

echo ""
if [[ ${RC} -ne 0 ]]; then
    echo "WARNING: remote exit code ${RC}"
fi
exit "${RC}"
