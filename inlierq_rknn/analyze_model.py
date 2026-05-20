import argparse
import json
from pathlib import Path


def module_param_shape(module):
    if hasattr(module, "weight") and module.weight is not None:
        return list(module.weight.shape)
    return None


def module_param_count(module):
    return sum(p.numel() for p in module.parameters(recurse=False))


def tensor_summary(value):
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.detach().min().cpu()) if value.numel() else None,
            "max": float(value.detach().max().cpu()) if value.numel() else None,
        }
    if isinstance(value, (list, tuple)):
        return [tensor_summary(item) for item in value]
    if isinstance(value, dict):
        return {str(key): tensor_summary(item) for key, item in value.items()}
    return {"type": type(value).__name__}


def pick_target_layers(named_modules):
    target_suffixes = ("Conv", "C2f", "SPPF", "Detect")
    targets = []
    for name, module in named_modules:
        class_name = module.__class__.__name__
        if class_name.endswith(target_suffixes) or class_name in {"Conv2d"}:
            targets.append(
                {
                    "name": name,
                    "type": class_name,
                    "weight_shape": module_param_shape(module),
                    "param_count": module_param_count(module),
                }
            )
    return targets


def build_markdown(report):
    lines = [
        "# YOLO26 Model Analysis",
        "",
        f"- model_path: `{report['model_path']}`",
        f"- torch: `{report['environment'].get('torch')}`",
        f"- ultralytics: `{report['environment'].get('ultralytics')}`",
        f"- cuda_available: `{report['environment'].get('cuda_available')}`",
        f"- total_modules: `{report['summary']['total_modules']}`",
        f"- target_layers: `{len(report['target_layers'])}`",
        "",
        "## Forward Output",
        "",
        "```json",
        json.dumps(report["forward_output"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Target Hook Layers",
        "",
        "| idx | name | type | weight_shape | params |",
        "|---:|---|---|---|---:|",
    ]
    for idx, layer in enumerate(report["target_layers"]):
        lines.append(
            f"| {idx} | `{layer['name']}` | `{layer['type']}` | "
            f"`{layer['weight_shape']}` | {layer['param_count']} |"
        )
    lines.extend(["", "## All Modules", "", "| name | type | params |", "|---|---|---:|"])
    for module in report["modules"]:
        lines.append(f"| `{module['name']}` | `{module['type']}` | {module['param_count']} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="spacer_640.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/model_analysis")
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(model_path))
    net = yolo.model.to(args.device).eval()
    named_modules = list(net.named_modules())

    modules = [
        {
            "name": name or "<root>",
            "type": module.__class__.__name__,
            "param_count": module_param_count(module),
            "weight_shape": module_param_shape(module),
        }
        for name, module in named_modules
    ]

    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, device=args.device)
        output = net(dummy)

    report = {
        "model_path": str(model_path),
        "environment": {
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
        },
        "summary": {"total_modules": len(modules)},
        "forward_output": tensor_summary(output),
        "target_layers": pick_target_layers(named_modules),
        "modules": modules,
    }

    json_path = output_dir / "spacer_640_model_analysis.json"
    md_path = output_dir / "spacer_640_model_analysis.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
