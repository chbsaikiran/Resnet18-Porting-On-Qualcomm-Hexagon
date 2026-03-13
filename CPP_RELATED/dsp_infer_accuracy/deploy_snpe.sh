#!/bin/bash
# ======================================================================
# deploy_snpe.sh  –  Deploy SNPE ResNet18 model to SDM710 device
#
# Pushes the DLC model, SNPE runtime (snpe-net-run), and libs to
# /data/local/tmp on the Android device.
#
# Prerequisites:
#   1. Convert ONNX to DLC first:  ./convert_onnx_to_dlc.sh
#   2. DLC file at: resnet18_cifar10.dlc
#
# For DSP: SNPE 2.10 has V65 skel (libSnpeDspV65Skel.so) — matches
# SDM710 Hexagon 685. Also pushes V66 for compatibility.
# ======================================================================

set -euo pipefail

SNPE_ROOT=${SNPE_ROOT:-/opt/qcom/aistack/snpe/2.10.40.4}
HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/home/saikiran/Qualcomm/Hexagon_SDK/6.5.0.0}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR=/data/local/tmp/snpe_resnet18

# --- Paths to SNPE components -----------------------------------------
SNPE_NET_RUN="${SNPE_ROOT}/bin/aarch64-android/snpe-net-run"
MODEL_DLC="${SCRIPT_DIR}/resnet18_cifar10.dlc"
ARM_LIBS="${SNPE_ROOT}/lib/aarch64-android"
DSP_V65="${SNPE_ROOT}/lib/hexagon-v65/unsigned"
DSP_V66="${SNPE_ROOT}/lib/hexagon-v66/unsigned"

# --- Verify DLC exists ------------------------------------------------
if [ ! -f "$MODEL_DLC" ]; then
    echo "ERROR: DLC model not found at $MODEL_DLC"
    echo ""
    echo "Convert ONNX to DLC first:"
    echo "  ./convert_onnx_to_dlc.sh"
    echo ""
    echo "Or run manually:"
    echo "  snpe-onnx-to-dlc -i <path/to/resnet18_cifar10.onnx> -o resnet18_cifar10.dlc"
    exit 1
fi

for f in "$SNPE_NET_RUN" "$MODEL_DLC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

echo "=== Deploying SNPE ResNet18 to device ==="

adb shell "mkdir -p ${DEVICE_DIR}"

echo ">>> Pushing snpe-net-run ..."
adb push "$SNPE_NET_RUN" "${DEVICE_DIR}/"
# Ensure executable bit is set on device (some pushes lose x permission)
adb shell "chmod 755 ${DEVICE_DIR}/snpe-net-run" || true

echo ">>> Pushing DLC model ..."
adb push "$MODEL_DLC" "${DEVICE_DIR}/"

echo ">>> Pushing SNPE runtime libs (CPU/GPU/DSP stubs) ..."
adb push "${ARM_LIBS}/libSNPE.so" "${DEVICE_DIR}/"
if [ -f "${ARM_LIBS}/libc++_shared.so" ]; then
    adb push "${ARM_LIBS}/libc++_shared.so" "${DEVICE_DIR}/"
else
    echo "  WARNING: libc++_shared.so not found under ${ARM_LIBS};"
    echo "           will rely on device/system copy instead."
fi
adb push "${ARM_LIBS}/libcalculator.so" "${DEVICE_DIR}/"
# DSP stubs (ARM-side) — V65 for SDM710 Hexagon 685, V66 fallback
[ -f "${ARM_LIBS}/libSnpeDspV65Stub.so" ] && adb push "${ARM_LIBS}/libSnpeDspV65Stub.so" "${DEVICE_DIR}/"
[ -f "${ARM_LIBS}/libSnpeDspV66Stub.so" ] && adb push "${ARM_LIBS}/libSnpeDspV66Stub.so" "${DEVICE_DIR}/"
# GPU
[ -f "${ARM_LIBS}/libSnpeHtpPrepare.so" ] && adb push "${ARM_LIBS}/libSnpeHtpPrepare.so" "${DEVICE_DIR}/"

echo ">>> Pushing SNPE DSP skel libs (DSP-side) ..."
if [ -d "$DSP_V65" ]; then
    adb push "${DSP_V65}/libSnpeDspV65Skel.so" "${DEVICE_DIR}/" 2>/dev/null || true
    adb push "${DSP_V65}/libcalculator_skel.so" "${DEVICE_DIR}/" 2>/dev/null || true
fi
if [ -d "$DSP_V66" ]; then
    adb push "${DSP_V66}/libSnpeDspV66Skel.so" "${DEVICE_DIR}/" 2>/dev/null || true
fi

# --- Test signature for unsigned DSP loading --------------------------
echo ">>> Generating & pushing DSP test signature ..."
SIGNER="${HEXAGON_SDK_ROOT}/utils/scripts/signer.py"
TESTSIG_DIR="${SCRIPT_DIR}/snpe_testsig"
if [ -f "$SIGNER" ]; then
    mkdir -p "$TESTSIG_DIR"
    SERIAL=$(adb shell getprop ro.serialno 2>/dev/null | tr -d '\r\n')
    if [ -n "$SERIAL" ]; then
        python3 "$SIGNER" --serial "$SERIAL" --output "$TESTSIG_DIR" 2>/dev/null || \
        python  "$SIGNER" --serial "$SERIAL" --output "$TESTSIG_DIR" 2>/dev/null || true
    fi
    for sig in "$TESTSIG_DIR"/testsig-*.so; do
        [ -f "$sig" ] && adb push "$sig" "${DEVICE_DIR}/" && echo "  pushed $(basename $sig)"
    done
else
    echo "  WARNING: signer.py not found. If DSP fails, generate test signature manually."
fi

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Files on device at ${DEVICE_DIR}/:"
adb shell "ls -la ${DEVICE_DIR}/"
echo ""
echo "To run inference:  ./run_snpe_infer.sh [NUM_IMAGES] [RUNTIME]"
echo "  Runtimes: dsp (default) | cpu | gpu"
