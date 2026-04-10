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
MODEL_DLC="${SCRIPT_DIR}/resnet18_cifar10_quantized.dlc"
ARM_LIBS="${SNPE_ROOT}/lib/aarch64-android"
DSP_V65="${SNPE_ROOT}/lib/hexagon-v65/unsigned"
DSP_V66="${SNPE_ROOT}/lib/hexagon-v66/unsigned"

# --- Optional: install DSP skels into /vendor/lib/rfsa/* ----------------
# Writing to /vendor typically needs adb root+remount; on user builds this may
# fail. We attempt it and always fall back to DEVICE_DIR.
RFSA_BASE=${RFSA_BASE:-/vendor/lib/rfsa}
RFSA_DOMAIN=${RFSA_DOMAIN:-}          # optional override: adsp | cdsp
RFSA_TARGET_DIR=${RFSA_TARGET_DIR:-}  # optional override: full path

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
SKEL_V65_LOCAL="${DSP_V65}/libSnpeDspV65Skel.so"
SKEL_V66_LOCAL="${DSP_V66}/libSnpeDspV66Skel.so"
CALC_SKEL_LOCAL="${DSP_V65}/libcalculator_skel.so"

# Decide RFSA install dir if not forced by env
if [ -z "${RFSA_TARGET_DIR}" ]; then
    if [ -n "${RFSA_DOMAIN}" ]; then
        RFSA_TARGET_DIR="${RFSA_BASE}/${RFSA_DOMAIN}"
    else
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

if [ -n "${RFSA_TARGET_DIR}" ]; then
    echo ">>> Attempting RFSA install to ${RFSA_TARGET_DIR} (domain=${RFSA_DOMAIN}) ..."
    set +e
    adb root >/dev/null 2>&1
    adb remount >/dev/null 2>&1
    set -e
    set +e
    [ -f "${SKEL_V65_LOCAL}" ] && adb push "${SKEL_V65_LOCAL}" "${RFSA_TARGET_DIR}/libSnpeDspV65Skel.so" >/dev/null 2>&1
    RC1=$?
    [ -f "${CALC_SKEL_LOCAL}" ] && adb push "${CALC_SKEL_LOCAL}" "${RFSA_TARGET_DIR}/libcalculator_skel.so" >/dev/null 2>&1
    RC2=$?
    [ -f "${SKEL_V66_LOCAL}" ] && adb push "${SKEL_V66_LOCAL}" "${RFSA_TARGET_DIR}/libSnpeDspV66Skel.so" >/dev/null 2>&1
    RC3=$?
    set -e
    if [ $RC1 -eq 0 ] || [ $RC2 -eq 0 ] || [ $RC3 -eq 0 ]; then
        echo "  RFSA install attempted (one or more pushes succeeded)."
    else
        echo "  WARNING: RFSA install failed (need root/remount or writable rfsa path)."
    fi
else
    echo ">>> RFSA domain dirs not found under ${RFSA_BASE}; skipping RFSA install."
fi

# Always push to DEVICE_DIR as a fallback
if [ -f "${SKEL_V65_LOCAL}" ]; then
    adb push "${SKEL_V65_LOCAL}" "${DEVICE_DIR}/" 2>/dev/null || true
fi
if [ -f "${CALC_SKEL_LOCAL}" ]; then
    adb push "${CALC_SKEL_LOCAL}" "${DEVICE_DIR}/" 2>/dev/null || true
fi
if [ -f "${SKEL_V66_LOCAL}" ]; then
    adb push "${SKEL_V66_LOCAL}" "${DEVICE_DIR}/" 2>/dev/null || true
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

# Also attempt to place the test signature into RFSA domain dir (if available),
# since some DSP loaders look there first for unsigned PD enablement.
if [ -n "${RFSA_TARGET_DIR:-}" ]; then
    for sig in "${TESTSIG_DIR}"/testsig-*.so; do
        if [ -f "$sig" ]; then
            set +e
            adb root >/dev/null 2>&1
            adb remount >/dev/null 2>&1
            adb push "$sig" "${RFSA_TARGET_DIR}/" >/dev/null 2>&1
            RC_SIG=$?
            set -e
            if [ $RC_SIG -eq 0 ]; then
                echo "  pushed $(basename "$sig") to ${RFSA_TARGET_DIR}/"
            fi
        fi
    done
fi

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Files on device at ${DEVICE_DIR}/:"
adb shell "ls -la ${DEVICE_DIR}/"
echo ""
echo "To run inference:  ./run_snpe_infer.sh [NUM_IMAGES] [RUNTIME]"
echo "  Runtimes: dsp (default) | cpu | gpu"
