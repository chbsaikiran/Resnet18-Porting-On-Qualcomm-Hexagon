#!/bin/bash
# Cross-compile cifar10_snpe_demo for aarch64 Android (SNPE C++ API).
#
# Prerequisites:
#   - Android NDK (r27+ recommended)
#   - SNPE SDK with aarch64-android prebuilts (e.g. 2.10.x)
#
# Environment overrides:
#   NDK_ROOT   — Android NDK root (default: /home/saikiran/NDK/android-ndk-r27d)
#   SNPE_ROOT  — SNPE install (default: /opt/qcom/aistack/snpe/2.10.40.4)
#   API_LEVEL  — min Android API for STL (default: 24)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NDK_ROOT="${NDK_ROOT:-/home/saikiran/NDK/android-ndk-r27d}"
SNPE_ROOT="${SNPE_ROOT:-/opt/qcom/aistack/snpe/2.10.40.4}"
API_LEVEL="${API_LEVEL:-24}"

HOST_TAG="linux-x86_64"
if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        HOST_TAG="darwin-arm64"
    else
        HOST_TAG="darwin-x86_64"
    fi
fi

CLANGXX="${NDK_ROOT}/toolchains/llvm/prebuilt/${HOST_TAG}/bin/aarch64-linux-android${API_LEVEL}-clang++"
# SNPE 2.x ships C++ API headers under include/SNPE (not include/zdl).
SNPE_INC="${SNPE_ROOT}/include/SNPE"
SNPE_LIB="${SNPE_ROOT}/lib/aarch64-android"

if [[ ! -x "${CLANGXX}" ]]; then
    echo "ERROR: NDK clang++ not found: ${CLANGXX}"
    echo "Set NDK_ROOT to your Android NDK."
    exit 1
fi

if [[ ! -f "${SNPE_INC}/SNPE/SNPEBuilder.hpp" ]]; then
    echo "ERROR: SNPE C++ headers not found under ${SNPE_INC}"
    echo "Set SNPE_ROOT to your SNPE SDK (expected: .../include/SNPE/SNPE/SNPEBuilder.hpp)."
    exit 1
fi

if [[ ! -f "${SNPE_LIB}/libSNPE.so" ]]; then
    echo "ERROR: libSNPE.so not found under ${SNPE_LIB}"
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/build_android"

echo "=== Building cifar10_snpe_demo (aarch64-android) ==="
echo "  CXX:       ${CLANGXX}"
echo "  SNPE_ROOT: ${SNPE_ROOT}"
echo ""

# SNPE headers use dynamic_cast; RTTI must be enabled.
"${CLANGXX}" -std=c++17 -O2 -fPIC -pie -frtti -DANDROID \
    -I"${SNPE_INC}" \
    -L"${SNPE_LIB}" \
    -o "${SCRIPT_DIR}/build_android/cifar10_snpe_demo" \
    "${SCRIPT_DIR}/main_snpe.cpp" \
    -lSNPE -llog -landroid -ldl -lc++_shared

echo ""
echo "  Output: ${SCRIPT_DIR}/build_android/cifar10_snpe_demo"
echo "=== Build complete ==="
echo ""
echo "Deploy:  ./deploy_demo_snpe.sh"
echo "Run:     ./run_demo_snpe.sh [dsp|gpu|cpu]"
