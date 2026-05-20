import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


DEFAULT_HOOKS = [
    "model.2",
    "model.4",
    "model.6",
    "model.8",
    "model.9",
    "model.10",
    "model.13",
    "model.16",
    "model.19",
    "model.22",
    "model.23.cv3.0",
    "model.23.cv3.1",
    "model.23.cv3.2",
]


def load_image(path, imgsz, device):
    image = Image.open(path).convert("RGB").resize((imgsz, imgsz))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def read_image_list(path, max_images):
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    images = [Path(line) for line in lines if line.strip()]
    return images[:max_images] if max_images > 0 else images


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def get_module(root, name):
    modules = dict(root.named_modules())
    if name not in modules:
        raise KeyError(f"Hook module not found: {name}")
    return modules[name]


class ActivationCollector:
    def __init__(self, model, hook_names):
        self.activations = {}
        self.handles = []
        for name in hook_names:
            module = get_module(model, name)
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is not None:
                self.activations[name] = tensor.detach().float().cpu()

        return hook

    def clear(self):
        self.activations.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def init_layer_state(channels):
    return {
        "count": 0,
        "min": np.inf,
        "max": -np.inf,
        "sum": 0.0,
        "sum_sq": 0.0,
        "channel_min": np.full(channels, np.inf, dtype=np.float64),
        "channel_max": np.full(channels, -np.inf, dtype=np.float64),
    }


def update_state(state, values):
    if values.size == 0:
        return
    state["count"] += int(values.size)
    state["min"] = min(state["min"], float(values.min()))
    state["max"] = max(state["max"], float(values.max()))
    state["sum"] += float(values.sum(dtype=np.float64))
    state["sum_sq"] += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
    state["channel_min"] = np.minimum(state["channel_min"], values.min(axis=0))
    state["channel_max"] = np.maximum(state["channel_max"], values.max(axis=0))


def asymmetric_u8_params(min_value, max_value, eps=1e-12):
    min_value = float(min_value)
    max_value = float(max_value)
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return None, None
    if max_value <= min_value:
        max_value = min_value + eps
    scale = (max_value - min_value) / 255.0
    scale = max(scale, eps)
    zero_point = int(np.clip(np.round(-min_value / scale), 0, 255))
    return float(scale), zero_point


def symmetric_i8_scale(abs_max, eps=1e-12):
    abs_max = float(abs_max)
    if not np.isfinite(abs_max):
        return None
    return float(max(abs_max / 127.0, eps))


def finalize_state(state):
    if state["count"] == 0:
        return {"count": 0, "error": "no inlier activations"}

    mean = state["sum"] / state["count"]
    variance = max(state["sum_sq"] / state["count"] - mean * mean, 0.0)
    scale_u8, zp_u8 = asymmetric_u8_params(state["min"], state["max"])
    abs_max = max(abs(state["min"]), abs(state["max"]))
    scale_i8 = symmetric_i8_scale(abs_max)

    channel_abs = np.maximum(np.abs(state["channel_min"]), np.abs(state["channel_max"]))
    channel_scale_i8 = np.maximum(channel_abs / 127.0, 1e-12)

    return {
        "count": int(state["count"]),
        "min": float(state["min"]),
        "max": float(state["max"]),
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "activation_u8_asymmetric": {
            "scale": scale_u8,
            "zero_point": zp_u8,
            "quant_min": 0,
            "quant_max": 255,
        },
        "activation_i8_symmetric": {
            "scale": scale_i8,
            "zero_point": 0,
            "quant_min": -128,
            "quant_max": 127,
        },
        "activation_i8_per_channel_symmetric": {
            "axis": 1,
            "scale": channel_scale_i8.astype(float).tolist(),
            "zero_point": 0,
            "quant_min": -128,
            "quant_max": 127,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="spacer_640.pt")
    parser.add_argument("--calib", default="outputs/calib/spacer_train64.txt")
    parser.add_argument("--masks", default="outputs/gvss/em_gmm_train64_masks.npz")
    parser.add_argument("--output", default="outputs/quant/inlier_activation_qparams.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--hooks", nargs="*", default=DEFAULT_HOOKS)
    args = parser.parse_args()

    images = read_image_list(args.calib, args.max_images)
    if not images:
        raise FileNotFoundError(f"No calibration images in {args.calib}")

    masks = np.load(args.masks)
    yolo = YOLO(str(Path(args.model).resolve()))
    model = yolo.model.to(args.device).eval()
    collector = ActivationCollector(model, args.hooks)

    states = {}
    offsets = {name: 0 for name in args.hooks}
    try:
        with torch.no_grad():
            for image_index, image_path in enumerate(images, start=1):
                collector.clear()
                image = load_image(image_path, args.imgsz, args.device)
                _ = model(image)

                for name, tensor in collector.activations.items():
                    key = name.replace(".", "__")
                    if key not in masks:
                        continue
                    if tensor.ndim != 4 or tensor.shape[0] != 1:
                        continue

                    _, channels, height, width = tensor.shape
                    spatial_count = height * width
                    start = offsets.get(name, 0)
                    end = start + spatial_count
                    layer_mask = masks[key][start:end].astype(bool)
                    offsets[name] = end

                    if layer_mask.size != spatial_count:
                        raise ValueError(f"Mask size mismatch for {name}: got {layer_mask.size}, expected {spatial_count}")
                    if not layer_mask.any():
                        continue

                    values = tensor[0].permute(1, 2, 0).reshape(spatial_count, channels).numpy()
                    selected = values[layer_mask]
                    states.setdefault(name, init_layer_state(channels))
                    update_state(states[name], selected)

                if image_index == 1 or image_index == len(images) or image_index % 8 == 0:
                    print(f"Processed {image_index}/{len(images)}: {image_path}")
    finally:
        collector.close()

    summary = {
        "model": str(Path(args.model).resolve()),
        "calib": str(Path(args.calib).resolve()),
        "masks": str(Path(args.masks).resolve()),
        "images": len(images),
        "layers": {name: finalize_state(state) for name, state in states.items()},
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Layers with qparams: {len(summary['layers'])}")


if __name__ == "__main__":
    main()
