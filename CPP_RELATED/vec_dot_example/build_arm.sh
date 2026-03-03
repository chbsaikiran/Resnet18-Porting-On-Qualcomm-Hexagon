#!/bin/bash
# ======================================================================
# build_arm.sh  –  Vector dot-product ARM host
#
# Builds the ARM binary that calls into the DSP via FastRPC.
# No OpenCV or other heavy dependencies – just FastRPC + rpcmem.
#
# Requirements:
#   - Android NDK installed   (set NDK_ROOT)
#   - Hexagon SDK installed   (set HEXAGON_SDK_ROOT)
#   - qaic-generated stub from build_dsp.sh must exist
#
# Deploy & run:
#   adb push build_arm/vec_dot_arm /data/local/tmp/
#   adb shell '/data/local/tmp/vec_dot_arm 1024'
# ======================================================================

set -euo pipefail

NDK_ROOT=${NDK_ROOT:-/opt/android-ndk}
HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/opt/hexagon/sdk}

API_LEVEL=30
TARGET=aarch64-linux-android

CC="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/${TARGET}${API_LEVEL}-clang++"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INC_DIR="${SCRIPT_DIR}/inc"
ARM_DIR="${SCRIPT_DIR}/arm"
BUILD_DSP_DIR="${SCRIPT_DIR}/build_dsp"
BUILD_DIR="${SCRIPT_DIR}/build_arm"

mkdir -p "${BUILD_DIR}"

# ---- Check stub exists -----------------------------------------------

STUB_C="${BUILD_DSP_DIR}/vec_dot_stub.c"
if [ ! -f "${STUB_C}" ]; then
    echo "ERROR: ${STUB_C} not found.  Run build_dsp.sh first."
    exit 1
fi

# ---- Compile ---------------------------------------------------------

CFLAGS=(
    -O2
    -std=c++17
    -I "${INC_DIR}"
    -I "${BUILD_DSP_DIR}"
    -I "${HEXAGON_SDK_ROOT}/incs"
    -I "${HEXAGON_SDK_ROOT}/incs/stddef"
    -I "${HEXAGON_SDK_ROOT}/libs/common/rpcmem/inc"
)

echo ">>> Compiling main.cpp ..."
${CC} "${CFLAGS[@]}" -c \
    "${ARM_DIR}/main.cpp" \
    -o "${BUILD_DIR}/main.o"

echo ">>> Compiling stub ..."
${CC} "${CFLAGS[@]}" -c \
    "${STUB_C}" \
    -o "${BUILD_DIR}/vec_dot_stub.o"

# ---- Link ------------------------------------------------------------

FASTRPC_LIBS=(
    -L "${HEXAGON_SDK_ROOT}/libs/common/remote/ship/android_aarch64"
    -ladsprpc
    -L "${HEXAGON_SDK_ROOT}/libs/common/rpcmem/android_aarch64"
    -lrpcmem
)

echo ">>> Linking vec_dot_arm ..."
${CC} -o "${BUILD_DIR}/vec_dot_arm" \
    "${BUILD_DIR}/main.o" \
    "${BUILD_DIR}/vec_dot_stub.o" \
    "${FASTRPC_LIBS[@]}" \
    -llog -ldl

echo ""
echo ">>> Build complete: ${BUILD_DIR}/vec_dot_arm"
echo ""
echo "Deploy & run:"
echo "  adb push ${BUILD_DIR}/vec_dot_arm /data/local/tmp/"
echo "  adb shell '/data/local/tmp/vec_dot_arm 1024'"
