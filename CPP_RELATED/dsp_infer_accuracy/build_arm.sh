#!/bin/bash
# ======================================================================
# build_arm.sh  –  CIFAR-10 DSP accuracy ARM host
#
# Builds the ARM binary that loads images, preprocesses on ARM,
# and sends each tensor to the Hexagon DSP for inference via FastRPC.
#
# Requirements:
#   - Android NDK installed   (set NDK_ROOT)
#   - Hexagon SDK installed   (set HEXAGON_SDK_ROOT)
#   - qaic-generated stub from build_dsp.sh must exist
#
# Deploy & run:
#   adb push build_arm/cifar10_dsp_accuracy /data/local/tmp/
#   adb push <model.dlc>                    /data/local/tmp/
#   adb push <test_batch.bin>               /data/local/tmp/
#   adb shell '/data/local/tmp/cifar10_dsp_accuracy \
#       /data/local/tmp/resnet18_cifar10.dlc \
#       /data/local/tmp/test_batch.bin'
# ======================================================================

set -euo pipefail

NDK_ROOT=${NDK_ROOT:-/home/saikiran/Android/ndk/android-ndk-r27d}
HEXAGON_SDK_ROOT=${HEXAGON_SDK_ROOT:-/home/saikiran/Qualcomm/Hexagon_SDK/5.5.6.0/5.5.6.0}

API_LEVEL=30
TARGET=aarch64-linux-android

CXX="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/${TARGET}${API_LEVEL}-clang++"
CC="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/${TARGET}${API_LEVEL}-clang"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INC_DIR="${SCRIPT_DIR}/inc"
ARM_DIR="${SCRIPT_DIR}/arm"
BUILD_DSP_DIR="${SCRIPT_DIR}/build_dsp"
BUILD_DIR="${SCRIPT_DIR}/build_arm"

mkdir -p "${BUILD_DIR}"

# ---- Check stub exists -----------------------------------------------

STUB_C="${BUILD_DSP_DIR}/cifar10_infer_stub.c"
if [ ! -f "${STUB_C}" ]; then
    echo "ERROR: ${STUB_C} not found.  Run build_dsp.sh first."
    exit 1
fi

# ---- Compile ---------------------------------------------------------

COMMON_INCLUDES=(
    -I "${INC_DIR}"
    -I "${BUILD_DSP_DIR}"
    -I "${HEXAGON_SDK_ROOT}/incs"
    -I "${HEXAGON_SDK_ROOT}/incs/stddef"
    -I "${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc"
)

echo ">>> Compiling main.cpp ..."
${CXX} -O2 -std=c++17 "${COMMON_INCLUDES[@]}" -c \
    "${ARM_DIR}/main.cpp" \
    -o "${BUILD_DIR}/main.o"

echo ">>> Compiling stub (as C) ..."
${CC} -O2 -std=c11 "${COMMON_INCLUDES[@]}" -c \
    "${STUB_C}" \
    -o "${BUILD_DIR}/cifar10_infer_stub.o"

# ---- Link ------------------------------------------------------------

FASTRPC_LIBS=(
    -L "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64"
    -ladsprpc
    "${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a"
)

echo ">>> Linking cifar10_dsp_accuracy ..."
${CXX} -o "${BUILD_DIR}/cifar10_dsp_accuracy" \
    "${BUILD_DIR}/main.o" \
    "${BUILD_DIR}/cifar10_infer_stub.o" \
    "${FASTRPC_LIBS[@]}" \
    -llog -ldl

echo ""
echo ">>> Build complete: ${BUILD_DIR}/cifar10_dsp_accuracy"
echo ""
echo "Deploy & run:"
echo "  adb push ${BUILD_DIR}/cifar10_dsp_accuracy /data/local/tmp/"
echo "  adb shell '/data/local/tmp/cifar10_dsp_accuracy \\"
echo "      /data/local/tmp/resnet18_cifar10.dlc \\"
echo "      /data/local/tmp/test_batch.bin'"
