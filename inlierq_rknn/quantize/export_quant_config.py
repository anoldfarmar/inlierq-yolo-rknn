#!/usr/bin/env python3
"""Export a unified InlierQ quantization config for later Q/DQ insertion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_score_logit_layer(layer_name: str) -> bool:
    return layer_name.startswith("model.23.cv3.")


def build_layer_config(
    layer_name: str,
    qparams: dict[str, Any],
    refined: dict[str, Any] | None,
    fisher: dict[str, Any] | None,
) -> dict[str, Any]:
    i8 = qparams["activation_i8_symmetric"]
    u8 = qparams["activation_u8_asymmetric"]
    per_channel = qparams.get("activation_i8_per_channel_symmetric")

    selected_scale = float(i8["scale"])
    scale_source = "inlier_minmax_i8_symmetric"
    refinement: dict[str, Any] | None = None
    if refined is not None:
        selected_scale = float(refined["refined_scale"])
        scale_source = "inlier_reconstruction_grid_search_i8_symmetric"
        refinement = {
            "init_scale": float(refined["init_scale"]),
            "refined_scale": selected_scale,
            "init_mse": float(refined["init_mse"]),
            "refined_mse": float(refined["refined_mse"]),
            "improvement_ratio": float(refined["improvement_ratio"]),
            "grid_factor": float(refined["grid_factor"]),
            "sample_count": int(refined["sample_count"]),
        }
    fisher_refinement: dict[str, Any] | None = None
    if fisher is not None:
        selected_scale = float(fisher["refined_scale"])
        scale_source = "diagonal_fisher_weighted_inlier_reconstruction_i8_symmetric"
        fisher_refinement = {
            "init_scale": float(fisher["init_scale"]),
            "refined_scale": selected_scale,
            "init_weighted_mse": float(fisher["init_weighted_mse"]),
            "refined_weighted_mse": float(fisher["refined_weighted_mse"]),
            "improvement_ratio": float(fisher["improvement_ratio"]),
            "steps": int(fisher["steps"]),
            "lr": float(fisher["lr"]),
            "sample_count": int(fisher["sample_count"]),
        }

    notes: list[str] = []
    activation_scheme = "symmetric_i8_per_tensor"
    if is_score_logit_layer(layer_name):
        notes.append(
            "score-logit layer: symmetric int8 is selected because u8 asymmetric "
            "zero-point saturates at 255 for negative logits"
        )

    layer_cfg: dict[str, Any] = {
        "layer": layer_name,
        "activation": {
            "scheme": activation_scheme,
            "scale": selected_scale,
            "zero_point": 0,
            "quant_min": -128,
            "quant_max": 127,
            "scale_source": scale_source,
        },
        "inlier_activation_stats": {
            "count": int(qparams["count"]),
            "min": float(qparams["min"]),
            "max": float(qparams["max"]),
            "mean": float(qparams["mean"]),
            "std": float(qparams["std"]),
        },
        "references": {
            "activation_i8_symmetric": {
                "scale": float(i8["scale"]),
                "zero_point": int(i8["zero_point"]),
                "quant_min": int(i8["quant_min"]),
                "quant_max": int(i8["quant_max"]),
            },
            "activation_u8_asymmetric": {
                "scale": float(u8["scale"]),
                "zero_point": int(u8["zero_point"]),
                "quant_min": int(u8["quant_min"]),
                "quant_max": int(u8["quant_max"]),
            },
        },
    }

    if per_channel is not None:
        layer_cfg["references"]["activation_i8_per_channel_symmetric"] = {
            "axis": int(per_channel["axis"]),
            "scale": [float(x) for x in per_channel["scale"]],
            "zero_point": int(per_channel["zero_point"]),
            "quant_min": int(per_channel["quant_min"]),
            "quant_max": int(per_channel["quant_max"]),
        }

    if refinement is not None:
        layer_cfg["grid_search_refinement"] = refinement
    if fisher_refinement is not None:
        layer_cfg["fisher_refinement"] = fisher_refinement
    if notes:
        layer_cfg["notes"] = notes

    return layer_cfg


def export_config(args: argparse.Namespace) -> dict[str, Any]:
    qparams_path = Path(args.qparams).resolve()
    refined_path = Path(args.refined).resolve()
    fisher_path = Path(args.fisher).resolve() if args.fisher else None
    output_path = Path(args.output).resolve()

    qparams = load_json(qparams_path)
    refined = load_json(refined_path)
    fisher_data = load_json(fisher_path) if fisher_path is not None else {"layers": {}}
    q_layers = qparams["layers"]
    r_layers = refined.get("layers", {})
    f_layers = fisher_data.get("layers", {})

    missing_refined = sorted(set(q_layers) - set(r_layers))
    extra_refined = sorted(set(r_layers) - set(q_layers))
    missing_fisher = sorted(set(q_layers) - set(f_layers)) if fisher_path is not None else []
    extra_fisher = sorted(set(f_layers) - set(q_layers))

    layers = {
        layer_name: build_layer_config(layer_name, layer_qparams, r_layers.get(layer_name), f_layers.get(layer_name))
        for layer_name, layer_qparams in q_layers.items()
    }

    config = {
        "format": "inlierq_quant_config.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": qparams.get("model"),
        "calib": qparams.get("calib"),
        "masks": qparams.get("masks"),
        "images": qparams.get("images"),
        "layer_count": len(layers),
        "default_weight": {
            "scheme": "symmetric_i8_per_channel",
            "axis": 0,
            "zero_point": 0,
            "quant_min": -128,
            "quant_max": 127,
            "scale_source": "phase3_weight_observer_or_exporter",
        },
        "default_activation": {
            "scheme": "symmetric_i8_per_tensor",
            "zero_point": 0,
            "quant_min": -128,
            "quant_max": 127,
            "scale_source": "diagonal_fisher_weighted_scale_when_available_else_grid_search",
        },
        "layers": layers,
        "source_files": {
            "qparams": str(qparams_path),
            "refined": str(refined_path),
            "fisher": str(fisher_path) if fisher_path is not None else None,
        },
        "limitations": [
            "The selected activation scales prefer a diagonal-Fisher proxy weighted by squared GVSS saliency.",
            "This is a diagonal Fisher approximation, not a full Hessian implementation.",
            "Layer names are PyTorch hook names; Phase 3 must map them to fake-quant or ONNX Q/DQ insertion points.",
        ],
        "validation": {
            "missing_refined_layers": missing_refined,
            "extra_refined_layers": extra_refined,
            "missing_fisher_layers": missing_fisher,
            "extra_fisher_layers": extra_fisher,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qparams",
        default="outputs/quant/inlier_activation_qparams.json",
        help="Inlier activation qparams JSON from inlier_optimizer.py",
    )
    parser.add_argument(
        "--refined",
        default="outputs/quant/inlier_activation_scale_refined.json",
        help="Refined scale JSON from scale_refine.py",
    )
    parser.add_argument(
        "--fisher",
        default="outputs/quant/inlier_activation_fisher_refined.json",
        help="Optional diagonal-Fisher refined scale JSON from fisher_scale_refine.py",
    )
    parser.add_argument(
        "--output",
        default="outputs/quant/inlierq_quant_config.json",
        help="Output unified quantization config JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = export_config(args)
    validation = config["validation"]
    print(f"Exported {config['layer_count']} layers to {Path(args.output).resolve()}")
    if validation["missing_refined_layers"]:
        print(f"Missing refined layers: {validation['missing_refined_layers']}")
    if validation["extra_refined_layers"]:
        print(f"Extra refined layers: {validation['extra_refined_layers']}")


if __name__ == "__main__":
    main()
