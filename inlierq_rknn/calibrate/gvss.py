import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    "model.23.cv2.0",
    "model.23.cv2.1",
    "model.23.cv2.2",
    "model.23.cv3.0",
    "model.23.cv3.1",
    "model.23.cv3.2",
    "model.23.one2one_cv2.0",
    "model.23.one2one_cv2.1",
    "model.23.one2one_cv2.2",
    "model.23.one2one_cv3.0",
    "model.23.one2one_cv3.1",
    "model.23.one2one_cv3.2",
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


def get_module(root, name):
    modules = dict(root.named_modules())
    if name not in modules:
        raise KeyError(f"Hook module not found: {name}")
    return modules[name]


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


class ActivationCatcher:
    def __init__(self, model, hook_names):
        self.activations = {}
        self.handles = []
        for name in hook_names:
            module = get_module(model, name)
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is None or not tensor.requires_grad:
                return
            tensor.retain_grad()
            self.activations[name] = tensor

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def extract_raw_scores(output):
    if isinstance(output, (list, tuple)) and len(output) >= 2 and isinstance(output[1], dict):
        raw = output[1]
        if "one2many" in raw and "scores" in raw["one2many"]:
            return raw["one2many"]["scores"]
        if "one2one" in raw and "scores" in raw["one2one"]:
            return raw["one2one"]["scores"]
    if isinstance(output, dict):
        if "one2many" in output and "scores" in output["one2many"]:
            return output["one2many"]["scores"]
        if "scores" in output:
            return output["scores"]
    if isinstance(output, torch.Tensor) and output.ndim == 3:
        return output[:, 4:5, :]
    raise TypeError("Could not find raw detection scores in model output")


def saliency_stats(tensor):
    score = saliency_score(tensor)
    if score is None:
        return None
    flat = score.float().flatten()
    return {
        "activation_shape": list(tensor.shape),
        "saliency_shape": list(score.shape),
        "min": float(flat.min().cpu()),
        "max": float(flat.max().cpu()),
        "mean": float(flat.mean().cpu()),
        "nonzero": int((flat > 0).sum().cpu()),
        "count": int(flat.numel()),
    }


def saliency_score(tensor):
    if tensor.grad is None:
        return None
    grad = tensor.grad.detach()
    if grad.ndim == 4:
        return grad.pow(2).sum(dim=1).sqrt()
    return grad.abs()


def compute_one(model, image_path, hook_names, imgsz, device, topk, return_scores=False):
    model.zero_grad(set_to_none=True)
    catcher = ActivationCatcher(model, hook_names)
    try:
        image = load_image(image_path, imgsz, device)
        image.requires_grad_(True)
        output = model(image)
        scores = extract_raw_scores(output)
        flat_scores = scores.reshape(scores.shape[0], -1)
        k = min(topk, flat_scores.shape[1])
        topk_scores = flat_scores.topk(k, dim=1).values
        loss = F.binary_cross_entropy_with_logits(topk_scores, torch.ones_like(topk_scores))
        loss.backward()

        layer_stats = {}
        layer_scores = {}
        for name, tensor in catcher.activations.items():
            stats = saliency_stats(tensor)
            if stats is not None:
                layer_stats[name] = stats
            if return_scores:
                score = saliency_score(tensor)
                if score is not None:
                    layer_scores[name] = score.float().cpu().numpy().reshape(-1)

        result = {
            "image": str(image_path),
            "loss": float(loss.detach().cpu()),
            "scores_shape": list(scores.shape),
            "topk": k,
            "layers": layer_stats,
        }
        if return_scores:
            result["saliency_scores"] = layer_scores
        return result
    finally:
        catcher.close()


def summarize_arrays(arrays):
    summary = {}
    for name, parts in arrays.items():
        if not parts:
            continue
        values = np.concatenate(parts).astype(np.float32, copy=False)
        quantiles = np.quantile(values, [0.5, 0.9, 0.95, 0.99])
        summary[name] = {
            "count": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "nonzero": int((values > 0).sum()),
            "p50": float(quantiles[0]),
            "p90": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "p99": float(quantiles[3]),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="spacer_640.pt")
    parser.add_argument("--calib", default="outputs/calib/spacer_train64.txt")
    parser.add_argument("--output", default="outputs/gvss/gvss_smoke.json")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--saliency-output", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--hooks", nargs="*", default=DEFAULT_HOOKS)
    args = parser.parse_args()

    yolo = YOLO(str(Path(args.model).resolve()))
    model = yolo.model.to(args.device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    images = read_image_list(args.calib, args.max_images)
    if not images:
        raise FileNotFoundError(f"No calibration images in {args.calib}")

    collect_scores = args.saliency_output is not None or args.summary_output is not None
    results = []
    saliency_arrays = {name: [] for name in args.hooks}
    for index, image in enumerate(images, start=1):
        result = compute_one(model, image, args.hooks, args.imgsz, args.device, args.topk, collect_scores)
        scores_by_layer = result.pop("saliency_scores", {})
        for name, values in scores_by_layer.items():
            saliency_arrays.setdefault(name, []).append(values)
        results.append(result)
        if index == 1 or index == len(images) or index % 8 == 0:
            print(f"Processed {index}/{len(images)}: {image}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.saliency_output:
        saliency_output = Path(args.saliency_output).resolve()
        saliency_output.parent.mkdir(parents=True, exist_ok=True)
        packed = {
            name.replace(".", "__"): np.concatenate(parts).astype(np.float32, copy=False)
            for name, parts in saliency_arrays.items()
            if parts
        }
        np.savez_compressed(saliency_output, **packed)
        print(f"Wrote {saliency_output}")

    if args.summary_output:
        summary_output = Path(args.summary_output).resolve()
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "images": len(results),
            "topk": args.topk,
            "layers": summarize_arrays(saliency_arrays),
        }
        summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {summary_output}")

    print(f"Wrote {output}")
    print(f"Images: {len(results)}")
    print(f"Layers with saliency: {len(results[0]['layers'])}")


if __name__ == "__main__":
    main()
