#!/bin/bash
# ======================================================================
# deploy_qnn.sh  –  Deploy QNN ResNet18 model to SDM710 device
#
# Pushes the model library, QNN DSP runtime, and qnn-net-run to
# /data/local/tmp on the Android device.
#
# SDM710 has Hexagon 685 (v66 arch) → uses QnnDsp backend.
# ======================================================================

set -euo pipefail

QAIRT_ROOT=${QAIRT_ROOT:-/opt/qcom/aistack/qairt/2.31.0.250130}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR=/data/local/tmp/qnn_resnet18

# --- Paths to QNN components -------------------------------------------
QNN_NET_RUN="${QAIRT_ROOT}/bin/aarch64-android/qnn-net-run"
MODEL_SO="${SCRIPT_DIR}/model_libs/aarch64-android/libresnet18_cifar10.so"

ARM_LIBS="${QAIRT_ROOT}/lib/aarch64-android"
DSP_LIBS="${QAIRT_ROOT}/lib/hexagon-v66/unsigned"

# --- Verify files exist ------------------------------------------------
for f in "$QNN_NET_RUN" "$MODEL_SO"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

echo "=== Deploying QNN ResNet18 to device ==="

adb shell "mkdir -p ${DEVICE_DIR}"

echo ">>> Pushing qnn-net-run ..."
adb push "$QNN_NET_RUN" "${DEVICE_DIR}/"

echo ">>> Pushing model library ..."
adb push "$MODEL_SO" "${DEVICE_DIR}/"

echo ">>> Pushing QNN backend libs ..."
adb push "${ARM_LIBS}/libQnnCpu.so"                    "${DEVICE_DIR}/"
adb push "${ARM_LIBS}/libQnnGpu.so"                    "${DEVICE_DIR}/"
adb push "${ARM_LIBS}/libQnnSystem.so"                 "${DEVICE_DIR}/"

# NOTE: SDM710 Hexagon 685 = DSP V65, but QAIRT 2.31.0 only ships V66+.
#       DSP backend is NOT supported on this device.
#       Use --backend libQnnGpu.so (GPU) or libQnnCpu.so (CPU) instead.

echo ">>> Pushing libc++_shared.so (NDK) ..."
NDK_ROOT=${NDK_ROOT:-/home/saikiran/Android/ndk/android-ndk-r27d}
LIBCPP="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -f "$LIBCPP" ]; then
    adb push "$LIBCPP" "${DEVICE_DIR}/"
fi

echo ">>> Pushing test signature ..."
TESTSIG="/tmp/testsig/testsig-0x791ff979.so"
if [ -f "$TESTSIG" ]; then
    adb push "$TESTSIG" "${DEVICE_DIR}/"
else
    echo "WARNING: test signature not found at $TESTSIG"
    echo "  DSP may refuse to load unsigned skel libraries."
fi

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Files on device at ${DEVICE_DIR}/:"
adb shell "ls -la ${DEVICE_DIR}/"
echo ""
echo "To run inference, use run_qnn_infer.sh"
