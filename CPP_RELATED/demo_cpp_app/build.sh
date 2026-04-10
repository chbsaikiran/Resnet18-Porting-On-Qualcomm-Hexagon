#!/bin/bash
# Build the host (x86_64) QNN demo that dlopen()s libQnnCpu.so + model .so.
# For SNPE on Android, use ./build_android_snpe.sh instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QAIRT_ROOT=${QAIRT_ROOT:-/opt/qcom/aistack/qairt/2.31.0.250130}
QNN_INC="${QAIRT_ROOT}/include/QNN"

mkdir -p "${SCRIPT_DIR}/build"

echo "=== Building cifar10_qnn_app ==="
echo "  Compiler: $(which g++)"
echo "  QNN headers: ${QNN_INC}"

g++ -std=c++17 -O2 \
    -I "${QNN_INC}" \
    -o "${SCRIPT_DIR}/build/cifar10_qnn_app" \
    "${SCRIPT_DIR}/main.cpp" \
    -ldl

echo "  Output: build/cifar10_qnn_app"
echo ""
echo "=== Build complete ==="
echo ""
echo "Run with:"
echo "  cd ${SCRIPT_DIR}"
echo "  ./build/cifar10_qnn_app <input.raw>"
