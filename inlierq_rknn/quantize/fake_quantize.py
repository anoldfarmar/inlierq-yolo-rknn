#!/usr/bin/env python3
"""Apply fixed InlierQ fake-quant wrappers and optionally export ONNX.

This script is intentionally verbose because ONNX export can be slow on WSL.
Use ``--dry-run`` first to validate module mapping without running export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

try:
    from tqdm import tqdm as progress_iter
except ModuleNotFoundError:  # pragma: no cover - tiny fallback for minimal envs
    def progress_iter(items, **kwargs):
        items = list(items)
        total = len(items)
        desc = kwargs.get("desc", "progress")
        for index, item in enumerate(items, start=1):
            if index == 1 or index == total or index % max(total // 10, 1) == 0:
                print(f"{desc}: {index}/{total}", flush=True)
            yield item


def stage(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def load_quant_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_parent_and_child(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def get_child(parent: nn.Module, child_name: str) -> nn.Module:
    return parent[int(child_name)] if child_name.isdigit() else getattr(parent, child_name)


def set_child(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def fake_quant_tensor(x: torch.Tensor, scale: float, zero_point: int, qmin: int, qmax: int) -> torch.Tensor:
    # torch.fake_quantize_per_tensor_affine is export-friendly and usually lowers to Q/DQ.
    return torch.fake_quantize_per_tensor_affine(x, float(scale), int(zero_point), int(qmin), int(qmax))


def fake_quant_output(value: Any, scale: float, zero_point: int, qmin: int, qmax: int) -> Any:
    if isinstance(value, torch.Tensor):
        return fake_quant_tensor(value, scale, zero_point, qmin, qmax)
    if isinstance(value, tuple):
        return tuple(fake_quant_output(item, scale, zero_point, qmin, qmax) for item in value)
    if isinstance(value, list):
        return [fake_quant_output(item, scale, zero_point, qmin, qmax) for item in value]
    return value


class ActivationFakeQuantWrapper(nn.Module):
    def __init__(self, module: nn.Module, layer_name: str, activation_cfg: dict[str, Any]):
        super().__init__()
        self.module = module
        self.layer_name = layer_name
        self.scale = float(activation_cfg["scale"])
        self.zero_point = int(activation_cfg["zero_point"])
        self.quant_min = int(activation_cfg["quant_min"])
        self.quant_max = int(activation_cfg["quant_max"])

        # Ultralytics' graph runner reads these attributes from modules in self.model.
        for attr in ("i", "f", "type", "np"):
            if hasattr(module, attr):
                setattr(self, attr, getattr(module, attr))

    def forward(self, x):
        y = self.module(x)
        return fake_quant_output(y, self.scale, self.zero_point, self.quant_min, self.quant_max)


class FirstTensorAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return first_tensor(self.model(x))


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"No tensor found in output of type {type(value).__name__}")


def apply_activation_fake_quant(model: nn.Module, config: dict[str, Any]) -> list[str]:
    wrapped = []
    modules = dict(model.named_modules())
    for layer_name, layer_cfg in progress_iter(config["layers"].items(), desc="wrap activation layers"):
        if layer_name not in modules:
            print(f"[warn] layer not found, skip: {layer_name}", flush=True)
            continue
        parent, child_name = get_parent_and_child(model, layer_name)
        child = get_child(parent, child_name)
        if isinstance(child, ActivationFakeQuantWrapper):
            continue
        set_child(parent, child_name, ActivationFakeQuantWrapper(child, layer_name, layer_cfg["activation"]))
        wrapped.append(layer_name)
    return wrapped


def fake_quantize_conv_weights(model: nn.Module) -> int:
    count = 0
    for module in progress_iter(list(model.modules()), desc="fake-quant conv weights"):
        if not isinstance(module, nn.Conv2d):
            continue
        with torch.no_grad():
            weight = module.weight.detach()
            flat = weight.flatten(1)
            absmax = flat.abs().amax(dim=1).clamp_min(1e-12)
            scales = absmax / 127.0
            q = torch.round(flat / scales[:, None]).clamp(-128, 127)
            module.weight.copy_((q * scales[:, None]).reshape_as(weight))
        count += 1
    return count


def smoke_forward(model: nn.Module, imgsz: int, device: str) -> None:
    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(dummy)
    print(f"Smoke forward output type: {type(output).__name__}", flush=True)


def export_onnx_torch(model: nn.Module, output: Path, imgsz: int, opset: int, device: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32, device=device)
    torch.onnx.export(
        FirstTensorAdapter(model),
        dummy,
        str(output),
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
        do_constant_folding=True,
    )


def export_onnx_ultralytics(yolo: YOLO, output: Path, imgsz: int, opset: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    exported = yolo.export(format="onnx", imgsz=imgsz, opset=opset, simplify=False, dynamic=False)
    exported_path = Path(exported).resolve()
    if exported_path != output.resolve():
        output.write_bytes(exported_path.read_bytes())
    print(f"Ultralytics export source: {exported_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="spacer_640.pt")
    parser.add_argument("--quant-config", default="outputs/quant/inlierq_quant_config.json")
    parser.add_argument("--output", default="outputs/onnx/spacer_640_inlierq_qdq.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--export-backend",
        choices=["ultralytics", "torch"],
        default="ultralytics",
        help="ultralytics keeps YOLO export behavior; torch exports first tensor from model(dummy)",
    )
    parser.add_argument("--no-weight-fake-quant", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate wrapping and smoke forward; do not export ONNX")
    args = parser.parse_args()

    stage("Load quant config")
    config = load_quant_config(Path(args.quant_config).resolve())
    print(f"Quant config layers: {config['layer_count']}", flush=True)

    stage("Load YOLO model")
    yolo = YOLO(str(Path(args.model).resolve()))
    model = yolo.model.to(args.device).eval()

    stage("Insert fixed activation fake-quant wrappers")
    wrapped = apply_activation_fake_quant(model, config)
    print(f"Wrapped activation layers: {len(wrapped)} / {config['layer_count']}", flush=True)
    if len(wrapped) != config["layer_count"]:
        print("[warn] Not all configured layers were wrapped. Inspect layer names before heavy export.", flush=True)

    if not args.no_weight_fake_quant:
        stage("Apply per-channel symmetric fake quant to Conv2d weights")
        conv_count = fake_quantize_conv_weights(model)
        print(f"Fake-quantized Conv2d weights: {conv_count}", flush=True)

    stage("Smoke forward")
    smoke_forward(model, args.imgsz, args.device)

    if args.dry_run:
        stage("Dry run complete; ONNX export skipped")
        return

    stage("Export ONNX Q/DQ candidate")
    print(
        "This is the CPU-heavy step. If the log stays here for a while, ONNX export is running.",
        flush=True,
    )
    if args.export_backend == "ultralytics":
        export_onnx_ultralytics(yolo, Path(args.output).resolve(), args.imgsz, args.opset)
    else:
        export_onnx_torch(model, Path(args.output).resolve(), args.imgsz, args.opset, args.device)
    print(f"Done: {Path(args.output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
