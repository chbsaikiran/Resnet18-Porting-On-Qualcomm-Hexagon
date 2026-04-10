#!/bin/bash
# ======================================================================
# deploy_qnn.sh  –  Deploy QNN ResNet18 model to SDM710 device
#
# Pushes the model library, QNN runtime (CPU/GPU/DSP backends), and
# qnn-net-run to /data/local/tmp on the Android device.
#
# For DSP: pushes the V66 DSP skel + stub libs.  SDM710 has Hexagon 685
# (V66); the closest available backend in QAIRT 2.31.0 is the V66 DSP
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

# --- Optional: install DSP libs into /vendor/lib/rfsa/* -----------------
# Many devices load DSP modules from RFSA domain dirs (adsp/cdsp). Writing to
# /vendor usually needs adb root+remount; on user builds this will fail, so we
# fall back to pushing into DEVICE_DIR and using ADSP_LIBRARY_PATH/CDSP_LIBRARY_PATH.
RFSA_BASE=${RFSA_BASE:-/vendor/lib/rfsa}
RFSA_DOMAIN=${RFSA_DOMAIN:-}   # optional override: adsp | cdsp
RFSA_TARGET_DIR=${RFSA_TARGET_DIR:-}  # optional override: full path

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
SKEL_LOCAL="${DSP_LIBS}/libQnnDspV66Skel.so"
DSPCORE_LOCAL="${DSP_LIBS}/libQnnDspV66.so"

# Decide RFSA install dir if not forced by env
if [ -z "${RFSA_TARGET_DIR}" ]; then
    if [ -n "${RFSA_DOMAIN}" ]; then
        RFSA_TARGET_DIR="${RFSA_BASE}/${RFSA_DOMAIN}"
    else
        # Prefer cdsp if present; otherwise adsp (common on SDM710 images)
        if adb shell "ls -ld '${RFSA_BASE}/cdsp' >/dev/null 2>&1"; then
            RFSA_DOMAIN="cdsp"
            RFSA_TARGET_DIR="${RFSA_BASE}/cdsp"
        elif adb shell "ls -ld '${RFSA_BASE}/adsp' >/dev/null 2>&1"; then
            RFSA_DOMAIN="adsp"
            RFSA_TARGET_DIR="${RFSA_BASE}/adsp"
        else
            RFSA_DOMAIN="adsp"
            RFSA_TARGET_DIR=""
        fi
    fi
fi

PUSHED_TO_RFSA=0
if [ -n "${RFSA_TARGET_DIR}" ]; then
    echo ">>> Attempting RFSA install to ${RFSA_TARGET_DIR} (domain=${RFSA_DOMAIN}) ..."
    set +e
    adb root >/dev/null 2>&1
    adb remount >/dev/null 2>&1
    set -e
    set +e
    adb push "${SKEL_LOCAL}"    "${RFSA_TARGET_DIR}/libQnnDspV66Skel.so" >/dev/null 2>&1
    RC1=$?
    adb push "${DSPCORE_LOCAL}" "${RFSA_TARGET_DIR}/libQnnDspV66.so"     >/dev/null 2>&1
    RC2=$?
    set -e
    if [ $RC1 -eq 0 ] && [ $RC2 -eq 0 ]; then
        echo "  RFSA install succeeded."
        PUSHED_TO_RFSA=1
    else
        echo "  WARNING: RFSA install failed (need root/remount or writable rfsa path)."
    fi
else
    echo ">>> RFSA domain dirs not found under ${RFSA_BASE}; skipping RFSA install."
fi

# Always push to DEVICE_DIR as a fallback (and for non-root devices).
echo ">>> Pushing DSP skel libs to ${DEVICE_DIR}/ (fallback) ..."
adb push "${SKEL_LOCAL}"    "${DEVICE_DIR}/"
adb push "${DSPCORE_LOCAL}" "${DEVICE_DIR}/"
#adb push "${DSP_LIBS}/unsigned/libSnpeDspV66Skel.so" "${DEVICE_DIR}/"

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
