import argparse
import json
from pathlib import Path

import numpy as np


def fit_two_component_gmm(values, max_iter=100, tol=1e-4, eps=1e-8):
    x = values.astype(np.float64, copy=False).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot fit GMM on empty array")

    if np.allclose(x.min(), x.max()):
        gamma = np.ones((x.size, 2), dtype=np.float64) * 0.5
        return {
            "mu": np.array([x.min(), x.max()], dtype=np.float64),
            "var": np.array([eps, eps], dtype=np.float64),
            "pi": np.array([0.5, 0.5], dtype=np.float64),
            "gamma": gamma,
            "iterations": 0,
            "converged": True,
        }

    scale = max(float(np.std(x)), float(np.max(np.abs(x))), eps)
    y = x / scale

    mu = np.quantile(y, [0.25, 0.90]).astype(np.float64)
    var = np.array([np.var(y) + eps, np.var(y) + eps], dtype=np.float64)
    pi = np.array([0.7, 0.3], dtype=np.float64)
    prev = np.concatenate([mu, var, pi])
    converged = False

    for iteration in range(1, max_iter + 1):
        prob = []
        for k in range(2):
            coef = 1.0 / np.sqrt(2.0 * np.pi * var[k])
            prob.append(pi[k] * coef * np.exp(-0.5 * (y - mu[k]) ** 2 / var[k]))
        prob = np.stack(prob, axis=1)
        denom = prob.sum(axis=1, keepdims=True) + eps
        gamma = prob / denom

        nk = gamma.sum(axis=0) + eps
        pi = nk / y.size
        mu = (gamma * y[:, None]).sum(axis=0) / nk
        var = (gamma * (y[:, None] - mu) ** 2).sum(axis=0) / nk
        var = np.maximum(var, eps)

        current = np.concatenate([mu, var, pi])
        delta = np.max(np.abs(current - prev))
        prev = current
        if delta < tol:
            converged = True
            break
    else:
        iteration = max_iter

    return {
        "mu": mu * scale,
        "var": var * scale * scale,
        "pi": pi,
        "gamma": gamma,
        "iterations": iteration,
        "converged": converged,
    }


def component_summary(values, posterior, mask):
    return {
        "count": int(values.size),
        "inlier_count": int(mask.sum()),
        "inlier_ratio": float(mask.mean()) if mask.size else 0.0,
        "saliency_min": float(values.min()) if values.size else None,
        "saliency_max": float(values.max()) if values.size else None,
        "saliency_mean": float(values.mean()) if values.size else None,
        "posterior_min": float(posterior.min()) if posterior.size else None,
        "posterior_max": float(posterior.max()) if posterior.size else None,
        "posterior_mean": float(posterior.mean()) if posterior.size else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/gvss/gvss_train64_saliency.npz")
    parser.add_argument("--summary-output", default="outputs/gvss/em_gmm_train64_summary.json")
    parser.add_argument("--mask-output", default="outputs/gvss/em_gmm_train64_masks.npz")
    parser.add_argument("--posterior-output", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sparse-nonzero-threshold", type=float, default=0.10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    saliency = np.load(args.input)
    masks = {}
    posteriors = {}
    summary = {
        "input": str(Path(args.input).resolve()),
        "threshold": args.threshold,
        "sparse_nonzero_threshold": args.sparse_nonzero_threshold,
        "inlier_rule": "component_with_larger_saliency_mean",
        "layers": {},
    }

    for key in saliency.files:
        values = saliency[key].astype(np.float32, copy=False).reshape(-1)
        nonzero_mask = values > 0
        nonzero_ratio = float(nonzero_mask.mean()) if values.size else 0.0
        fit_nonzero_only = 0.0 < nonzero_ratio < args.sparse_nonzero_threshold
        fit_values = values[nonzero_mask] if fit_nonzero_only else values

        fit = fit_two_component_gmm(fit_values, args.max_iter, args.tol)
        inlier_component = int(np.argmax(fit["mu"]))
        posterior = np.zeros(values.shape, dtype=np.float32)
        if fit_nonzero_only:
            posterior[nonzero_mask] = fit["gamma"][:, inlier_component].astype(np.float32)
        else:
            posterior = fit["gamma"][:, inlier_component].astype(np.float32)
        mask = posterior > args.threshold

        layer_name = key.replace("__", ".")
        masks[key] = mask.astype(np.uint8)
        posteriors[key] = posterior
        summary["layers"][layer_name] = {
            "key": key,
            "mu": [float(v) for v in fit["mu"]],
            "var": [float(v) for v in fit["var"]],
            "pi": [float(v) for v in fit["pi"]],
            "inlier_component": inlier_component,
            "fit_nonzero_only": bool(fit_nonzero_only),
            "nonzero_ratio": nonzero_ratio,
            "iterations": int(fit["iterations"]),
            "converged": bool(fit["converged"]),
            **component_summary(values, posterior, mask),
        }

    summary_output = Path(args.summary_output).resolve()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mask_output = Path(args.mask_output).resolve()
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_output, **masks)

    if args.posterior_output:
        posterior_output = Path(args.posterior_output).resolve()
        posterior_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(posterior_output, **posteriors)
        print(f"Wrote {posterior_output}")

    print(f"Wrote {summary_output}")
    print(f"Wrote {mask_output}")
    print(f"Layers: {len(summary['layers'])}")


if __name__ == "__main__":
    main()
