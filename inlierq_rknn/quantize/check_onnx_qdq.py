#!/usr/bin/env python3
"""Check ONNX Q/DQ nodes and optionally run ONNX checker."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm as progress_iter
except ModuleNotFoundError:  # pragma: no cover
    def progress_iter(items, **kwargs):
        items = list(items)
        total = len(items)
        desc = kwargs.get("desc", "progress")
        for index, item in enumerate(items, start=1):
            if index == 1 or index == total or index % max(total // 10, 1) == 0:
                print(f"{desc}: {index}/{total}", flush=True)
            yield item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()

    import onnx

    path = Path(args.onnx).resolve()
    print(f"==> Load ONNX: {path}", flush=True)
    model = onnx.load(str(path))

    if args.check_model:
        print("==> Run onnx.checker.check_model", flush=True)
        onnx.checker.check_model(model)

    print("==> Count nodes", flush=True)
    op_counts = Counter()
    q_nodes = []
    dq_nodes = []
    for node in progress_iter(model.graph.node, desc="scan ONNX nodes"):
        op_counts[node.op_type] += 1
        if node.op_type == "QuantizeLinear":
            q_nodes.append(node.name or node.output[0])
        elif node.op_type == "DequantizeLinear":
            dq_nodes.append(node.name or node.output[0])

    summary = {
        "onnx": str(path),
        "node_count": len(model.graph.node),
        "op_counts": dict(sorted(op_counts.items())),
        "quantize_linear_count": len(q_nodes),
        "dequantize_linear_count": len(dq_nodes),
        "graph_inputs": [value.name for value in model.graph.input],
        "graph_outputs": [value.name for value in model.graph.output],
        "sample_quantize_nodes": q_nodes[:20],
        "sample_dequantize_nodes": dq_nodes[:20],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.summary:
        out = Path(args.summary).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Summary written: {out}", flush=True)

    if len(q_nodes) == 0 or len(dq_nodes) == 0:
        raise RuntimeError("No QuantizeLinear/DequantizeLinear nodes found. This ONNX is not a Q/DQ model yet.")


if __name__ == "__main__":
    main()
