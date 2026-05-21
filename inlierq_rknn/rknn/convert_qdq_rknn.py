#!/usr/bin/env python3
"""Convert an ONNX Q/DQ model to RKNN without RKNN re-quantization."""

from __future__ import annotations

import argparse
import types
from pathlib import Path


def stage(index: int, total: int, message: str) -> None:
    print(f"\n[{index}/{total}] {message}", flush=True)


def patch_onnx_mapping_for_rknn() -> None:
    """Restore old onnx.mapping symbols expected by rknn-toolkit2 2.3.2.

    ONNX >= 1.16 moved mapping helpers under onnx._mapping and removed the
    public onnx.mapping module. RKNN 2.3.2 still imports onnx.mapping directly
    while loading ONNX, so we provide the small compatibility surface it uses.
    """
    import onnx

    if hasattr(onnx, "mapping"):
        return
    if not hasattr(onnx, "_mapping") or not hasattr(onnx._mapping, "TENSOR_TYPE_MAP"):
        raise RuntimeError(
            f"Unsupported ONNX mapping API in onnx {getattr(onnx, '__version__', 'unknown')}. "
            "Try installing onnx==1.14.1 in the rknn environment."
        )

    tensor_type_to_np_type = {
        tensor_type: dtype_map.np_dtype
        for tensor_type, dtype_map in onnx._mapping.TENSOR_TYPE_MAP.items()
    }
    np_type_to_tensor_type = {
        dtype_map.np_dtype: tensor_type
        for tensor_type, dtype_map in onnx._mapping.TENSOR_TYPE_MAP.items()
    }
    onnx.mapping = types.SimpleNamespace(
        TENSOR_TYPE_TO_NP_TYPE=tensor_type_to_np_type,
        NP_TYPE_TO_TENSOR_TYPE=np_type_to_tensor_type,
    )
    print(
        f"Patched onnx.mapping compatibility for onnx {onnx.__version__} "
        f"({len(tensor_type_to_np_type)} tensor types)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="input ONNX Q/DQ model")
    parser.add_argument("--output", default="outputs/rknn/spacer_640_inlierq.rknn")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--mean", nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument("--std", nargs=3, type=float, default=[255, 255, 255])
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--do-quantization", action="store_true", help="debug only: allow RKNN to quantize again")
    parser.add_argument("--dataset", default=None, help="optional dataset txt, only used with --do-quantization")
    parser.add_argument("--output-optimize", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    patch_onnx_mapping_for_rknn()
    from rknn.api import RKNN

    onnx_path = Path(args.onnx).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 5
    stage(1, total, "Create RKNN object")
    rknn = RKNN(verbose=args.verbose)

    try:
        stage(2, total, "Config RKNN")
        rknn.config(
            mean_values=[args.mean],
            std_values=[args.std],
            target_platform=args.target,
            optimization_level=args.optimization_level,
            output_optimize=args.output_optimize,
        )

        stage(3, total, f"Load ONNX: {onnx_path}")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"Load ONNX failed: {ret}")

        stage(4, total, "Build RKNN")
        print(
            f"do_quantization={args.do_quantization}. For Q/DQ ONNX this should normally be False.",
            flush=True,
        )
        build_kwargs = {"do_quantization": bool(args.do_quantization)}
        if args.do_quantization:
            if not args.dataset:
                raise ValueError("--dataset is required when --do-quantization is enabled")
            build_kwargs["dataset"] = str(Path(args.dataset).resolve())
        ret = rknn.build(**build_kwargs)
        if ret != 0:
            raise RuntimeError(f"Build RKNN failed: {ret}")

        stage(5, total, f"Export RKNN: {output_path}")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"Export RKNN failed: {ret}")
    finally:
        rknn.release()

    print(f"\nDone: {output_path}", flush=True)


if __name__ == "__main__":
    main()
