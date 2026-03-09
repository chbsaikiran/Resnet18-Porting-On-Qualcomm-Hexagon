#!/bin/bash
# ======================================================================
# build_dsp.sh  –  CIFAR-10 inference DSP skeleton library
#
# Builds libcifar10_infer_skel.so for the Hexagon DSP.
#
# Requirements:
#   - Hexagon SDK installed (set HEXAGON_SDK_ROOT)
#   - qaic on PATH
#
# Backends (optional, mutually exclusive):
#   -DUSE_SNPE   → link against SNPE for DSP inference
#   -DUSE_QNN    → link against QNN  for HTP inference
#   (default)    → mock backend for testing the FastRPC pipeline
#
# Deploy:
#   adb push build_dsp/libcifar10_infer_skel.so /vendor/lib/rfsa/dsp/
# ======================================================================

set -euo pipefail

HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/home/saikiran/Qualcomm/Hexagon_SDK/5.5.6.0/5.5.6.0}
HEXAGON_TOOLS_ROOT=${HEXAGON_TOOLS_ROOT:-${HEXAGON_SDK_ROOT}/tools}
HEXAGON_ARCH=${HEXAGON_ARCH:-v68}

HEXAGON_CC=${HEXAGON_TOOLS_ROOT}/bin/hexagon-clang

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IDL_DIR="${SCRIPT_DIR}/idl"
INC_DIR="${SCRIPT_DIR}/inc"
DSP_DIR="${SCRIPT_DIR}/dsp"
BUILD_DIR="${SCRIPT_DIR}/build_dsp"

mkdir -p "${BUILD_DIR}"

# ---- Step 1: Generate stub/skel from IDL ----------------------------

echo ">>> Running qaic on cifar10_infer.idl ..."
qaic -mdll \
     -I "${HEXAGON_SDK_ROOT}/incs/stddef" \
     -o "${BUILD_DIR}" \
     "${IDL_DIR}/cifar10_infer.idl"
echo ">>> qaic generated: cifar10_infer.h, cifar10_infer_stub.c, cifar10_infer_skel.c"

# ---- Step 2: Compile ------------------------------------------------

CFLAGS=(
    -m${HEXAGON_ARCH}
    -O2
    -fPIC
    -I "${INC_DIR}"
    -I "${BUILD_DIR}"
    -I "${HEXAGON_SDK_ROOT}/incs"
    -I "${HEXAGON_SDK_ROOT}/incs/stddef"
    -D__HEXAGON__
)

EXTRA_DEFINES=()
EXTRA_LIBS=()

# Uncomment ONE of the following for a real NN backend:
# EXTRA_DEFINES+=(-DUSE_SNPE -I "${SNPE_ROOT}/include/zdl")
# EXTRA_LIBS+=(-L "${SNPE_ROOT}/lib/hexagon-v68" -lSNPE)
#
# EXTRA_DEFINES+=(-DUSE_QNN -I "${QNN_SDK_ROOT}/include/QNN")
# EXTRA_LIBS+=(-L "${QNN_SDK_ROOT}/lib/hexagon-v68" -lQnnHtp)

echo ">>> Compiling skel ..."
${HEXAGON_CC} "${CFLAGS[@]}" "${EXTRA_DEFINES[@]+"${EXTRA_DEFINES[@]}"}" -c \
    "${BUILD_DIR}/cifar10_infer_skel.c" \
    -o "${BUILD_DIR}/cifar10_infer_skel.o"

echo ">>> Compiling implementation ..."
${HEXAGON_CC} "${CFLAGS[@]}" "${EXTRA_DEFINES[@]+"${EXTRA_DEFINES[@]}"}" -c \
    "${DSP_DIR}/cifar10_infer_imp.c" \
    -o "${BUILD_DIR}/cifar10_infer_imp.o"

# ---- Step 3: Link ---------------------------------------------------

echo ">>> Linking libcifar10_infer_skel.so ..."
${HEXAGON_CC} -m${HEXAGON_ARCH} -shared -fPIC \
    -o "${BUILD_DIR}/libcifar10_infer_skel.so" \
    "${BUILD_DIR}/cifar10_infer_skel.o" \
    "${BUILD_DIR}/cifar10_infer_imp.o" \
    -L "${HEXAGON_SDK_ROOT}/libs/hexagon/${HEXAGON_ARCH}" \
    "${EXTRA_LIBS[@]+"${EXTRA_LIBS[@]}"}" \
    -lc

echo ""
echo ">>> Build complete: ${BUILD_DIR}/libcifar10_infer_skel.so"
echo ""
echo "Deploy:"
echo "  adb push ${BUILD_DIR}/libcifar10_infer_skel.so /vendor/lib/rfsa/dsp/"
