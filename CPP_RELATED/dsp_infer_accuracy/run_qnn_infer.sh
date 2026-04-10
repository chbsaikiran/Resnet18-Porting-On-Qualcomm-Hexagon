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
NUM_IMAGES=${1:-10}
RAW_BACKEND=${2:-libQnnDsp.so}

# Allow friendly aliases: cpu|gpu|dsp in addition to full lib names.
case "${RAW_BACKEND}" in
    cpu|CPU) BACKEND="libQnnCpu.so" ;;
    gpu|GPU) BACKEND="libQnnGpu.so" ;;
    dsp|DSP) BACKEND="libQnnDsp.so" ;;
    *)       BACKEND="${RAW_BACKEND}" ;;
esac

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
DSP_LIB_PATH="${DEVICE_DIR};/system/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/dsp"

set +e
if [ "${BACKEND}" = "libQnnDsp.so" ]; then
    # DSP backend can segfault when processing large input lists on some devices/SDK combos.
    # Workaround: run per-image (or small batches) so we still get outputs.
    DSP_BATCH=${DSP_BATCH:-1}
    echo "  DSP_BATCH: ${DSP_BATCH} (workaround for DSP segfault)"
    adb shell "rm -rf ${DEVICE_DIR}/outputs && mkdir -p ${DEVICE_DIR}/outputs"

    i=0
    while [ $i -lt ${NUM_IMAGES} ]; do
        # Build a small input_list chunk on-device
        adb shell "rm -f ${DEVICE_DIR}/inputs/input_list_chunk.txt"
        j=0
        while [ $j -lt ${DSP_BATCH} ] && [ $i -lt ${NUM_IMAGES} ]; do
            fname=$(printf "img_%05d.raw" $i)
            adb shell "echo ${DEVICE_DIR}/inputs/${fname} >> ${DEVICE_DIR}/inputs/input_list_chunk.txt"
            i=$((i+1))
            j=$((j+1))
        done

        adb shell "cd ${DEVICE_DIR}/inputs && \
            export LD_LIBRARY_PATH=${DEVICE_DIR} && \
            export ADSP_LIBRARY_PATH='${DSP_LIB_PATH}' && \
            export CDSP_LIBRARY_PATH='${DSP_LIB_PATH}' && \
            ${DEVICE_DIR}/qnn-net-run \
                --model ${DEVICE_DIR}/libresnet18_cifar10.so \
                --backend ${DEVICE_DIR}/${BACKEND} \
                --input_list ${DEVICE_DIR}/inputs/input_list_chunk.txt \
                --output_dir ${DEVICE_DIR}/outputs \
                --log_level verbose"
        CHUNK_RC=$?
        if [ $CHUNK_RC -ne 0 ]; then
            echo ""
            echo "WARNING: DSP chunk run failed with code ${CHUNK_RC}"
            RUN_EXIT=$CHUNK_RC
            break
        fi
    done
    RUN_EXIT=${RUN_EXIT:-0}
else
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
fi
set -e
if [ $RUN_EXIT -ne 0 ]; then
    echo ""
    if [ "${BACKEND}" = "libQnnDsp.so" ]; then
        echo "WARNING: qnn-net-run exited with code ${RUN_EXIT} on DSP."
        echo "         If no outputs are produced, this is NOT a teardown-only segfault."
        echo "         Common cause on SDM710: QAIRT 2.31 ships Hexagon v66 DSP libs but SDM710 is v65."
        echo "         In that case, use SNPE DSP flow (v66 skel) or run QNN on CPU/GPU."
    else
        echo "WARNING: qnn-net-run exited with code ${RUN_EXIT} (segfault during teardown can be expected — results may still be valid)"
    fi
fi

# --- Step 4: Pull results -----------------------------------------------
echo ""
echo "=== Step 4: Pulling results ==="
rm -rf "${OUTPUT_LOCAL}"
mkdir -p "${OUTPUT_LOCAL}"
adb pull "${DEVICE_DIR}/outputs/" "${OUTPUT_LOCAL}/" 2>/dev/null || true

# If nothing was produced, stop early with a clear error (avoid misleading accuracy step).
if ! ls -1 "${OUTPUT_LOCAL}"/* >/dev/null 2>&1; then
    echo ""
    echo "ERROR: No output files were produced/pulled."
    if [ "${BACKEND}" = "libQnnDsp.so" ]; then
        echo "       DSP backend likely crashed early (see log above)."
        echo "       Try:  ./run_qnn_infer.sh ${NUM_IMAGES} cpu"
        echo "       Or use SNPE DSP scripts (SNPE 2.10 has v66 skel for SDM710)."
    fi
    exit 2
fi

# --- Step 5: Compute accuracy -------------------------------------------
echo ""
echo "=== Step 5: Computing accuracy ==="
conda run -n mlpy310env python "${SCRIPT_DIR}/compute_accuracy.py" \
    --output_dir "${OUTPUT_LOCAL}" \
    --labels "${INPUT_DIR}/labels.txt"
