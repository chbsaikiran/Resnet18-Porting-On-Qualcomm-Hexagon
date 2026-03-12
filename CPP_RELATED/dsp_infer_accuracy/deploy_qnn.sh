#!/bin/bash
# ======================================================================
# deploy_qnn.sh  –  Deploy QNN ResNet18 model to SDM710 device
#
# Pushes the model library, QNN runtime (CPU/GPU/DSP backends), and
# qnn-net-run to /data/local/tmp on the Android device.
#
# For DSP: pushes the V66 DSP skel + stub libs.  SDM710 has Hexagon 685
# (V65); the closest available backend in QAIRT 2.31.0 is the V66 DSP
# backend — it may work via backward compatibility.
# ======================================================================

set -euo pipefail

QAIRT_ROOT=${QAIRT_ROOT:-/opt/qcom/aistack/qairt/2.31.0.250130}
HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/home/saikiran/Qualcomm/Hexagon_SDK/6.5.0.0}
NDK_ROOT=${NDK_ROOT:-/home/saikiran/NDK/android-ndk-r27d}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR=/data/local/tmp/qnn_resnet18

# --- Paths to QNN components -------------------------------------------
QNN_NET_RUN="${QAIRT_ROOT}/bin/aarch64-android/qnn-net-run"
MODEL_SO="${SCRIPT_DIR}/lib/aarch64-android/libresnet18_cifar10.so"

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

echo ">>> Pushing QNN backend libs (CPU / GPU) ..."
adb push "${ARM_LIBS}/libQnnCpu.so"    "${DEVICE_DIR}/"
adb push "${ARM_LIBS}/libQnnGpu.so"    "${DEVICE_DIR}/"
adb push "${ARM_LIBS}/libQnnSystem.so" "${DEVICE_DIR}/"

echo ">>> Pushing QNN DSP backend libs (ARM-side) ..."
adb push "${ARM_LIBS}/libQnnDsp.so"                  "${DEVICE_DIR}/"
adb push "${ARM_LIBS}/libQnnDspV66Stub.so"            "${DEVICE_DIR}/"
adb push "${ARM_LIBS}/libQnnDspNetRunExtensions.so"   "${DEVICE_DIR}/"

echo ">>> Pushing QNN DSP skel libs (DSP-side, hexagon-v66) ..."
adb push "${DSP_LIBS}/libQnnDspV66Skel.so" "${DEVICE_DIR}/"
adb push "${DSP_LIBS}/libQnnDspV66.so"     "${DEVICE_DIR}/"

echo ">>> Pushing libc++_shared.so (NDK) ..."
LIBCPP="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [ -f "$LIBCPP" ]; then
    adb push "$LIBCPP" "${DEVICE_DIR}/"
else
    echo "WARNING: libc++_shared.so not found at $LIBCPP"
fi

# --- Test signature for unsigned DSP loading ---------------------------
echo ">>> Generating & pushing DSP test signature ..."
SIGNER="${HEXAGON_SDK_ROOT}/utils/scripts/signer.py"
TESTSIG_DIR="${SCRIPT_DIR}/testsig"
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
    echo "  WARNING: signer.py not found at $SIGNER"
    echo "  If DSP loading fails, generate a test signature manually."
fi

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Files on device at ${DEVICE_DIR}/:"
adb shell "ls -la ${DEVICE_DIR}/"
echo ""
echo "To run inference:  ./run_qnn_infer.sh [NUM_IMAGES] [BACKEND]"
echo "  Backends: libQnnDsp.so (DSP) | libQnnCpu.so (CPU) | libQnnGpu.so (GPU)"
