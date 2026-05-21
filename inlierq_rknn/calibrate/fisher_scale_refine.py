#!/usr/bin/env python3
"""Refine Inlier activation scales with a diagonal-Fisher style weighting.

The Fisher proxy is the squared GVSS saliency at each spatial position. This is
not the paper's full Hessian implementation, but it completes a practical
diagonal approximation that is consistent with the available GVSS signals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from inlier_optimizer import ActivationCollector, DEFAULT_HOOKS, load_image, read_image_list


def collect_weighted_inlier_samples(
    model,
    images,
    masks,
    saliency,
    hook_names,
    imgsz,
    device,
    max_samples_per_layer,
    seed,
):
    rng = np.random.default_rng(seed)
    collector = ActivationCollector(model, hook_names)
    offsets = {name: 0 for name in hook_names}
    samples = {name: np.empty((0,), dtype=np.float32) for name in hook_names}
    weights = {name: np.empty((0,), dtype=np.float32) for name in hook_names}

    try:
        with torch.no_grad():
            for image_index, image_path in enumerate(images, start=1):
                collector.clear()
                image = load_image(image_path, imgsz, device)
                _ = model(image)

                for name, tensor in collector.activations.items():
                    key = name.replace(".", "__")
                    if key not in masks or key not in saliency or tensor.ndim != 4 or tensor.shape[0] != 1:
                        continue

                    _, channels, height, width = tensor.shape
                    spatial_count = height * width
                    start = offsets.get(name, 0)
                    end = start + spatial_count
                    layer_mask = masks[key][start:end].astype(bool)
                    layer_saliency = saliency[key][start:end].astype(np.float32, copy=False)
                    offsets[name] = end

                    if layer_mask.size != spatial_count or layer_saliency.size != spatial_count:
                        raise ValueError(f"Mask/saliency size mismatch for {name}")
                    if not layer_mask.any():
                        continue

                    values = tensor[0].permute(1, 2, 0).reshape(spatial_count, channels).numpy()
                    selected_values = values[layer_mask].reshape(-1).astype(np.float32, copy=False)
                    selected_weights = np.repeat(np.square(layer_saliency[layer_mask]), channels)
                    selected_weights = selected_weights.astype(np.float32, copy=False)
                    if selected_values.size == 0:
                        continue

                    merged_values = np.concatenate([samples[name], selected_values])
                    merged_weights = np.concatenate([weights[name], selected_weights])
                    if merged_values.size > max_samples_per_layer:
                        keep = rng.choice(merged_values.size, size=max_samples_per_layer, replace=False)
                        merged_values = merged_values[keep]
                        merged_weights = merged_weights[keep]
                    samples[name] = merged_values
                    weights[name] = merged_weights

                if image_index == 1 or image_index == len(images) or image_index % 8 == 0:
                    print(f"Collected {image_index}/{len(images)}: {image_path}")
    finally:
        collector.close()

    return {
        name: (samples[name], normalize_weights(weights[name]))
        for name in hook_names
        if samples[name].size > 0
    }


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    if weights.size == 0 or not np.isfinite(weights).all() or float(weights.max()) <= 0.0:
        return np.ones_like(weights, dtype=np.float32)
    weights = weights + 1e-12
    return weights / max(float(weights.mean()), 1e-12)


def weighted_mse_np(values: np.ndarray, weights: np.ndarray, scale: float) -> float:
    q = np.clip(np.round(values / scale), -128, 127)
    diff = q * scale - values
    return float(np.mean(weights * diff * diff))


def optimize_scale(
    values: np.ndarray,
    weights: np.ndarray,
    init_scale: float,
    lr: float,
    steps: int,
    device: str,
) -> dict:
    if values.size == 0 or init_scale <= 0:
        raise ValueError("empty values or invalid init scale")

    x = torch.from_numpy(values.astype(np.float32, copy=False)).to(device)
    w = torch.from_numpy(weights.astype(np.float32, copy=False)).to(device)
    log_scale = torch.tensor(np.log(init_scale), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_scale], lr=lr)

    best_scale = float(init_scale)
    best_loss = weighted_mse_np(values, weights, best_scale)
    init_loss = best_loss

    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(log_scale).clamp_min(1e-12)
        q = torch.clamp(torch.round(x / scale), -128, 127)
        restored = q * scale
        loss = torch.mean(w * torch.square(restored - x))
        loss.backward()
        optimizer.step()

        current_scale = float(torch.exp(log_scale).detach().cpu())
        current_loss = weighted_mse_np(values, weights, current_scale)
        if current_loss < best_loss:
            best_loss = current_loss
            best_scale = current_scale

    return {
        "init_scale": float(init_scale),
        "refined_scale": float(best_scale),
        "init_weighted_mse": float(init_loss),
        "refined_weighted_mse": float(best_loss),
        "improvement_ratio": float((init_loss - best_loss) / max(init_loss, 1e-18)),
        "steps": int(steps),
        "lr": float(lr),
        "sample_count": int(values.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="spacer_640.pt")
    parser.add_argument("--calib", default="outputs/calib/spacer_train64.txt")
    parser.add_argument("--masks", default="outputs/gvss/em_gmm_train64_masks.npz")
    parser.add_argument("--saliency", default="outputs/gvss/gvss_train64_saliency.npz")
    parser.add_argument("--qparams", default="outputs/quant/inlier_activation_qparams.json")
    parser.add_argument("--start-scales", default="outputs/quant/inlier_activation_scale_refined.json")
    parser.add_argument("--output", default="outputs/quant/inlier_activation_fisher_refined.json")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--optim-device", default="cpu")
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--max-samples-per-layer", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hooks", nargs="*", default=DEFAULT_HOOKS)
    args = parser.parse_args()

    images = read_image_list(args.calib, args.max_images)
    masks = np.load(args.masks)
    saliency = np.load(args.saliency)
    qparams = json.loads(Path(args.qparams).read_text(encoding="utf-8"))
    start_scales = json.loads(Path(args.start_scales).read_text(encoding="utf-8")).get("layers", {})

    yolo = YOLO(str(Path(args.model).resolve()))
    model = yolo.model.to(args.device).eval()
    samples = collect_weighted_inlier_samples(
        model,
        images,
        masks,
        saliency,
        args.hooks,
        args.imgsz,
        args.device,
        args.max_samples_per_layer,
        args.seed,
    )

    layers = {}
    for name, (values, layer_weights) in samples.items():
        layer_params = qparams.get("layers", {}).get(name)
        if not layer_params:
            continue
        init_scale = start_scales.get(name, {}).get(
            "refined_scale",
            layer_params["activation_i8_symmetric"]["scale"],
        )
        layers[name] = optimize_scale(values, layer_weights, init_scale, args.lr, args.steps, args.optim_device)

    output_data = {
        "qparams": str(Path(args.qparams).resolve()),
        "masks": str(Path(args.masks).resolve()),
        "saliency": str(Path(args.saliency).resolve()),
        "start_scales": str(Path(args.start_scales).resolve()),
        "images": len(images),
        "objective": "diagonal_fisher_weighted_inlier_reconstruction_mse_symmetric_i8",
        "fisher_proxy": "gvss_saliency_squared_per_spatial_volume",
        "layers": layers,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Layers refined: {len(layers)}")


if __name__ == "__main__":
    main()
