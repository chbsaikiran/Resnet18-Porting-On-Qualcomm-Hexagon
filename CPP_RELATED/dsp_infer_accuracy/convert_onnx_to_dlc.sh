#!/bin/bash
# ======================================================================
# convert_onnx_to_dlc.sh  –  Convert ResNet18 ONNX to SNPE DLC format
#
# Uses QAIRT SDK's snpe-onnx-to-dlc (Python 3.10 compatible).
# Requires: conda env 'mlpy310env' with numpy and onnx.
#
# Usage: ./convert_onnx_to_dlc.sh
#
# Output: resnet18_cifar10.dlc in this directory
# ======================================================================

set -euo pipefail

QAIRT_ROOT=${QAIRT_ROOT:-/opt/qcom/aistack/qairt/2.31.0.250130}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ONNX_PATH=""
DLC_OUT="${SCRIPT_DIR}/resnet18_cifar10.dlc"

for candidate in \
    "${SCRIPT_DIR}/../../Training/resnet18_cifar10.onnx" \
    "${SCRIPT_DIR}/../Training/resnet18_cifar10.onnx" \
    "${SCRIPT_DIR}/resnet18_cifar10.onnx"; do
    if [ -f "$candidate" ]; then
        ONNX_PATH="$candidate"
        break
    fi
done

if [ -z "$ONNX_PATH" ]; then
    echo "ERROR: ONNX model not found. Expected:"
    echo "  .../Training/resnet18_cifar10.onnx"
    exit 1
fi

# Use QAIRT SDK's snpe-onnx-to-dlc (works with Python 3.10).
# The standalone SNPE 2.10 converter needs Python 3.6 which is EOL.
CONVERTER="${QAIRT_ROOT}/bin/x86_64-linux-clang/snpe-onnx-to-dlc"
if [ ! -f "$CONVERTER" ]; then
    echo "ERROR: snpe-onnx-to-dlc not found at $CONVERTER"
    echo "  Set QAIRT_ROOT if the SDK is installed elsewhere."
    exit 1
fi

echo "=== Converting ONNX to DLC ==="
echo "  Input:    ${ONNX_PATH}"
echo "  Output:   ${DLC_OUT}"
echo "  Converter: ${CONVERTER}"
echo ""

conda run -n mlpy310env python "$CONVERTER" -i "$ONNX_PATH" -o "$DLC_OUT"

echo ""
echo "=== DLC created: ${DLC_OUT} ==="
