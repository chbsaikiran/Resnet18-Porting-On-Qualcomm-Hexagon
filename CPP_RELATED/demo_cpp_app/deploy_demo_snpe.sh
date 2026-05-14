#!/bin/bash
# ======================================================================
# deploy_demo_snpe.sh — Push SNPE demo binary + runtime to the device
#
# Does not modify dsp_infer_accuracy. Expects:
#   - DLC at ../dsp_infer_accuracy/resnet18_cifar10_quantized.dlc (same as SNPE flow)
#   - Binary from ./build_android_snpe.sh → build_android/cifar10_snpe_demo
#
# Environment (same defaults as dsp_infer_accuracy/deploy_snpe.sh):
#   SNPE_ROOT, HEXAGON_SDK_ROOT
# ======================================================================

set -euo pipefail

SNPE_ROOT="${SNPE_ROOT:-/opt/qcom/aistack/qairt/2.31.0.250130}"
HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/home/saikiran/Qualcomm/Hexagon_SDK/6.5.0.0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE_DIR=/data/local/tmp/demo_snpe_app

DSP_INF="${SCRIPT_DIR}/../dsp_infer_accuracy/resnet18_cifar10_quantized.dlc"
BIN_LOCAL="${SCRIPT_DIR}/build_android/cifar10_snpe_demo"

ARM_LIBS="${SNPE_ROOT}/lib/aarch64-android"
NDK_ROOT="${NDK_ROOT:-/home/saikiran/NDK/android-ndk-r27d}"
NDK_CXX_SHARED="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
DSP_V65="${SNPE_ROOT}/lib/hexagon-v65/unsigned"
DSP_V66="${SNPE_ROOT}/lib/hexagon-v66/unsigned"

RFSA_BASE="${RFSA_BASE:-/vendor/lib/rfsa}"
RFSA_DOMAIN="${RFSA_DOMAIN:-}"
RFSA_TARGET_DIR="${RFSA_TARGET_DIR:-}"

if [[ ! -f "${BIN_LOCAL}" ]]; then
    echo "ERROR: ${BIN_LOCAL} not found."
    echo "Build first:  ./build_android_snpe.sh"
    exit 1
fi

if [[ ! -f "${DSP_INF}" ]]; then
    echo "ERROR: DLC not found at ${DSP_INF}"
    echo "Use the quantized DLC from your SNPE pipeline (same path as dsp_infer_accuracy)."
    exit 1
fi

echo "=== Deploying SNPE C++ demo to device ==="

adb shell "mkdir -p ${DEVICE_DIR}"

echo ">>> Pushing cifar10_snpe_demo ..."
adb push "${BIN_LOCAL}" "${DEVICE_DIR}/"
adb shell "chmod 755 ${DEVICE_DIR}/cifar10_snpe_demo" || true

echo ">>> Pushing DLC ..."
adb push "${DSP_INF}" "${DEVICE_DIR}/resnet18_cifar10_quantized.dlc"

echo ">>> Pushing SNPE runtime libs ..."
adb push "${ARM_LIBS}/libSNPE.so" "${DEVICE_DIR}/"
if [[ -f "${ARM_LIBS}/libc++_shared.so" ]]; then
    adb push "${ARM_LIBS}/libc++_shared.so" "${DEVICE_DIR}/"
elif [[ -f "${NDK_CXX_SHARED}" ]]; then
    echo "  libc++_shared.so not in SNPE libs; using NDK copy."
    adb push "${NDK_CXX_SHARED}" "${DEVICE_DIR}/"
else
    echo "  WARNING: libc++_shared.so not found in SNPE libs or NDK; may fail at runtime."
fi
# QNN backend libs — SNPE 2.31 wraps QNN and dlopen's these at runtime
for lib in \
    libQnnModelDlc.so \
    libQnnSystem.so   \
    libQnnCpu.so      \
    libQnnGpu.so      \
    libQnnDsp.so      \
    libQnnHtp.so      \
    libSnpeDspV66Stub.so \
    libSnpeHtpPrepare.so \
    libcalculator.so; do
    [[ -f "${ARM_LIBS}/${lib}" ]] && adb push "${ARM_LIBS}/${lib}" "${DEVICE_DIR}/"
done

SKEL_V65_LOCAL="${DSP_V65}/libSnpeDspV65Skel.so"
SKEL_V66_LOCAL="${DSP_V66}/libSnpeDspV66Skel.so"
CALC_SKEL_LOCAL="${DSP_V65}/libcalculator_skel.so"

if [[ -z "${RFSA_TARGET_DIR}" ]]; then
    if [[ -n "${RFSA_DOMAIN}" ]]; then
        RFSA_TARGET_DIR="${RFSA_BASE}/${RFSA_DOMAIN}"
    elif adb shell "ls -ld '${RFSA_BASE}/cdsp' >/dev/null 2>&1"; then
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

if [[ -n "${RFSA_TARGET_DIR}" ]]; then
    echo ">>> Attempting RFSA install to ${RFSA_TARGET_DIR} ..."
    set +e
    adb root >/dev/null 2>&1
    adb remount >/dev/null 2>&1
    set -e
    set +e
    [[ -f "${SKEL_V65_LOCAL}" ]] && adb push "${SKEL_V65_LOCAL}" "${RFSA_TARGET_DIR}/libSnpeDspV65Skel.so" >/dev/null 2>&1
    [[ -f "${CALC_SKEL_LOCAL}" ]] && adb push "${CALC_SKEL_LOCAL}" "${RFSA_TARGET_DIR}/libcalculator_skel.so" >/dev/null 2>&1
    [[ -f "${SKEL_V66_LOCAL}" ]] && adb push "${SKEL_V66_LOCAL}" "${RFSA_TARGET_DIR}/libSnpeDspV66Skel.so" >/dev/null 2>&1
    set -e
    echo "  (RFSA push may require root; device dir always has skel copies below.)"
else
    echo ">>> RFSA dirs not found under ${RFSA_BASE}; skipping RFSA install."
fi

[[ -f "${SKEL_V65_LOCAL}" ]] && adb push "${SKEL_V65_LOCAL}" "${DEVICE_DIR}/" 2>/dev/null || true
[[ -f "${CALC_SKEL_LOCAL}" ]] && adb push "${CALC_SKEL_LOCAL}" "${DEVICE_DIR}/" 2>/dev/null || true
[[ -f "${SKEL_V66_LOCAL}" ]] && adb push "${SKEL_V66_LOCAL}" "${DEVICE_DIR}/" 2>/dev/null || true

echo ">>> Generating & pushing DSP test signature (if Hexagon SDK present) ..."
SIGNER="${HEXAGON_SDK_ROOT}/utils/scripts/signer.py"
TESTSIG_DIR="${SCRIPT_DIR}/snpe_testsig"
if [[ -f "${SIGNER}" ]]; then
    mkdir -p "${TESTSIG_DIR}"
    SERIAL=$(adb shell getprop ro.serialno 2>/dev/null | tr -d '\r\n')
    if [[ -n "${SERIAL}" ]]; then
        python3 "${SIGNER}" --serial "${SERIAL}" --output "${TESTSIG_DIR}" 2>/dev/null || \
        python  "${SIGNER}" --serial "${SERIAL}" --output "${TESTSIG_DIR}" 2>/dev/null || true
    fi
    for sig in "${TESTSIG_DIR}"/testsig-*.so; do
        [[ -f "${sig}" ]] && adb push "${sig}" "${DEVICE_DIR}/" && echo "  pushed $(basename "${sig}")"
    done
    if [[ -n "${RFSA_TARGET_DIR:-}" ]]; then
        for sig in "${TESTSIG_DIR}"/testsig-*.so; do
            if [[ -f "${sig}" ]]; then
                set +e
                adb root >/dev/null 2>&1
                adb remount >/dev/null 2>&1
                adb push "${sig}" "${RFSA_TARGET_DIR}/" >/dev/null 2>&1
                set -e
            fi
        done
    fi
else
    echo "  WARNING: signer.py not found at ${SIGNER}"
fi

echo ""
echo "=== Deploy complete ==="
adb shell "ls -la ${DEVICE_DIR}/"
echo ""
echo "Run:  ./run_demo_snpe.sh [dsp|gpu|cpu]"
