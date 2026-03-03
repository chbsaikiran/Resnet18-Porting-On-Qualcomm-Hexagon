#!/bin/bash
# ======================================================================
# build_dsp.sh  –  Vector dot-product DSP library
#
# Builds libvec_dot_skel.so for the Hexagon DSP.
# No external SDK dependencies (no QNN) – pure C implementation.
#
# Requirements:
#   - Hexagon SDK installed (set HEXAGON_SDK_ROOT)
#   - qaic on PATH
#
# Deploy:
#   adb push build_dsp/libvec_dot_skel.so /vendor/lib/rfsa/dsp/
# ======================================================================

set -euo pipefail

HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/opt/hexagon/sdk}
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

echo ">>> Running qaic on vec_dot.idl ..."
qaic -mdll \
     -I "${HEXAGON_SDK_ROOT}/incs/stddef" \
     -o "${BUILD_DIR}" \
     "${IDL_DIR}/vec_dot.idl"
echo ">>> qaic generated: vec_dot.h, vec_dot_stub.c, vec_dot_skel.c"

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

echo ">>> Compiling skel ..."
${HEXAGON_CC} "${CFLAGS[@]}" -c \
    "${BUILD_DIR}/vec_dot_skel.c" \
    -o "${BUILD_DIR}/vec_dot_skel.o"

echo ">>> Compiling implementation ..."
${HEXAGON_CC} "${CFLAGS[@]}" -c \
    "${DSP_DIR}/vec_dot_imp.c" \
    -o "${BUILD_DIR}/vec_dot_imp.o"

# ---- Step 3: Link ---------------------------------------------------

echo ">>> Linking libvec_dot_skel.so ..."
${HEXAGON_CC} -m${HEXAGON_ARCH} -shared -fPIC \
    -o "${BUILD_DIR}/libvec_dot_skel.so" \
    "${BUILD_DIR}/vec_dot_skel.o" \
    "${BUILD_DIR}/vec_dot_imp.o" \
    -L "${HEXAGON_SDK_ROOT}/libs/hexagon/${HEXAGON_ARCH}" \
    -lc

echo ""
echo ">>> Build complete: ${BUILD_DIR}/libvec_dot_skel.so"
echo ""
echo "Deploy:"
echo "  adb push ${BUILD_DIR}/libvec_dot_skel.so /vendor/lib/rfsa/dsp/"
