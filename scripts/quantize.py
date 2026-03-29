#!/usr/bin/env python
"""Quantize an ONNX model to INT8 using dynamic quantization.

Usage:
    python scripts/quantize.py <model.onnx> [--output model_int8.onnx]
"""

import argparse
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantization for ONNX models")
    parser.add_argument("model", type=Path, help="Path to the FP32 ONNX model")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: model_int8.onnx next to input)")
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_path = args.output or model_path.parent / "model_int8.onnx"

    print(f"Quantizing {model_path} -> {output_path}")
    quantize_dynamic(
        str(model_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )
    print(f"Done. Quantized model: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
