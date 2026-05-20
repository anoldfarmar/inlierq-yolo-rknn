import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from inlier_optimizer import ActivationCollector, DEFAULT_HOOKS, load_image, read_image_list


def collect_inlier_samples(model, images, masks, hook_names, imgsz, device, max_samples_per_layer, seed):
    rng = np.random.default_rng(seed)
    collector = ActivationCollector(model, hook_names)
    offsets = {name: 0 for name in hook_names}
    samples = {name: np.empty((0,), dtype=np.float32) for name in hook_names}

    try:
        with torch.no_grad():
            for image_index, image_path in enumerate(images, start=1):
                collector.clear()
                image = load_image(image_path, imgsz, device)
                _ = model(image)

                for name, tensor in collector.activations.items():
                    key = name.replace(".", "__")
                    if key not in masks or tensor.ndim != 4 or tensor.shape[0] != 1:
                        continue
                    _, channels, height, width = tensor.shape
                    spatial_count = height * width
                    start = offsets.get(name, 0)
                    end = start + spatial_count
                    layer_mask = masks[key][start:end].astype(bool)
                    offsets[name] = end
                    if not layer_mask.any():
                        continue

                    values = tensor[0].permute(1, 2, 0).reshape(spatial_count, channels).numpy()
                    selected = values[layer_mask].reshape(-1).astype(np.float32, copy=False)
                    if selected.size == 0:
                        continue

                    merged = np.concatenate([samples.get(name, np.empty((0,), dtype=np.float32)), selected])
                    if merged.size > max_samples_per_layer:
                        keep = rng.choice(merged.size, size=max_samples_per_layer, replace=False)
                        merged = merged[keep]
                    samples[name] = merged

                if image_index == 1 or image_index == len(images) or image_index % 8 == 0:
                    print(f"Collected {image_index}/{len(images)}: {image_path}")
    finally:
        collector.close()

    return {name: values for name, values in samples.items() if values.size > 0}


def fake_quant_symmetric(values, scale):
    q = np.clip(np.round(values / scale), -128, 127)
    return q * scale


def mse(values, scale):
    restored = fake_quant_symmetric(values, scale)
    diff = restored - values
    return float(np.mean(diff * diff))


def refine_scale(values, init_scale, grid_points, low_factor, high_factor):
    if values.size == 0 or init_scale <= 0:
        return None
    factors = np.geomspace(low_factor, high_factor, grid_points)
    candidates = init_scale * factors
    losses = np.array([mse(values, scale) for scale in candidates], dtype=np.float64)
    best_index = int(np.argmin(losses))
    return {
        "init_scale": float(init_scale),
        "refined_scale": float(candidates[best_index]),
        "init_mse": float(mse(values, init_scale)),
        "refined_mse": float(losses[best_index]),
        "improvement_ratio": float((mse(values, init_scale) - losses[best_index]) / max(mse(values, init_scale), 1e-18)),
        "grid_factor": float(factors[best_index]),
        "sample_count": int(values.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="spacer_640.pt")
    parser.add_argument("--calib", default="outputs/calib/spacer_train64.txt")
    parser.add_argument("--masks", default="outputs/gvss/em_gmm_train64_masks.npz")
    parser.add_argument("--qparams", default="outputs/quant/inlier_activation_qparams.json")
    parser.add_argument("--output", default="outputs/quant/inlier_activation_scale_refined.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--max-samples-per-layer", type=int, default=200000)
    parser.add_argument("--grid-points", type=int, default=81)
    parser.add_argument("--low-factor", type=float, default=0.5)
    parser.add_argument("--high-factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hooks", nargs="*", default=DEFAULT_HOOKS)
    args = parser.parse_args()

    images = read_image_list(args.calib, args.max_images)
    masks = np.load(args.masks)
    qparams = json.loads(Path(args.qparams).read_text(encoding="utf-8"))

    yolo = YOLO(str(Path(args.model).resolve()))
    model = yolo.model.to(args.device).eval()
    samples = collect_inlier_samples(
        model,
        images,
        masks,
        args.hooks,
        args.imgsz,
        args.device,
        args.max_samples_per_layer,
        args.seed,
    )

    layers = {}
    for name, values in samples.items():
        layer_params = qparams.get("layers", {}).get(name)
        if not layer_params:
            continue
        init_scale = layer_params["activation_i8_symmetric"]["scale"]
        refined = refine_scale(values, init_scale, args.grid_points, args.low_factor, args.high_factor)
        if refined is not None:
            layers[name] = refined

    output_data = {
        "qparams": str(Path(args.qparams).resolve()),
        "masks": str(Path(args.masks).resolve()),
        "images": len(images),
        "objective": "uniform_inlier_reconstruction_mse_symmetric_i8",
        "layers": layers,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Layers refined: {len(layers)}")


if __name__ == "__main__":
    main()
