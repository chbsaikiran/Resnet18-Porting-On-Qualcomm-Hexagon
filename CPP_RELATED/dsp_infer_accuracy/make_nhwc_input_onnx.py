#!/usr/bin/env python3
"""
Create an ONNX model variant that accepts NHWC input.

Why:
  QAIRT 2.31 qnn-onnx-converter can hit a shape-prop bug on some NCHW models
  (residual Add broadcast failure with trailing 0 dims). A reliable workaround
  is to make the model input explicitly NHWC and insert a Transpose (NHWC->NCHW)
  as the very first node, so the converter doesn't need to "guess" axis order.

This script:
  - Loads an existing ONNX model (default: model.onnx)
  - Replaces the first graph input with a new input named "input_nhwc"
    shaped [1,32,32,3]
  - Inserts a Transpose node that produces the original input tensor name,
    shaped [1,3,32,32]
  - Saves the new model (default: model_nhwc_input.onnx)
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import onnx
from onnx import TensorProto, helper


def _get_dim_value(dim) -> Optional[int]:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="model.onnx")
    ap.add_argument("--out", dest="out_path", default="model_nhwc_input.onnx")
    ap.add_argument("--input_name", default="input")
    ap.add_argument("--output_name", default=None)
    args = ap.parse_args()

    m = onnx.load(args.in_path)
    g = m.graph

    if len(g.input) < 1:
        raise RuntimeError("ONNX graph has no inputs")

    # Find the input to replace by name (default: "input")
    input_vi = None
    for vi in g.input:
        if vi.name == args.input_name:
            input_vi = vi
            break
    if input_vi is None:
        raise RuntimeError(f"Input '{args.input_name}' not found. Inputs: {[i.name for i in g.input]}")

    # Validate existing input looks like NCHW (1,3,32,32) but don't hard fail.
    try:
        dims = [d for d in input_vi.type.tensor_type.shape.dim]
        old_shape = [_get_dim_value(d) for d in dims]
    except Exception:
        old_shape = None

    # New NHWC input
    new_input_name = f"{args.input_name}_nhwc"
    nhwc_vi = helper.make_tensor_value_info(new_input_name, TensorProto.FLOAT, [1, 32, 32, 3])

    # Insert Transpose: NHWC -> NCHW
    # Output name must match the original input tensor name so existing graph stays intact.
    tnode = helper.make_node(
        "Transpose",
        inputs=[new_input_name],
        outputs=[args.input_name],
        perm=[0, 3, 1, 2],
        name="Transpose_NHWC_to_NCHW",
    )

    # Replace the graph input ValueInfo entry with NHWC input
    for i in range(len(g.input)):
        if g.input[i].name == args.input_name:
            g.input.remove(g.input[i])
            break
    g.input.insert(0, nhwc_vi)

    # Prepend transpose node
    g.node.insert(0, tnode)

    # Optional: keep the same output name if user provided it (sanity check)
    if args.output_name is not None:
        out_names = [o.name for o in g.output]
        if args.output_name not in out_names:
            raise RuntimeError(f"Output '{args.output_name}' not found. Outputs: {out_names}")

    onnx.checker.check_model(m)
    onnx.save(m, args.out_path)

    print(f"Wrote {args.out_path}")
    if old_shape is not None:
        print(f"Original input '{args.input_name}' shape in source model: {old_shape}")
    print(f"New input '{new_input_name}' shape: [1,32,32,3] (NHWC)")
    print("Inserted Transpose perm=[0,3,1,2] (NHWC->NCHW)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

