# YOLO26 InlierQ Hook Plan

## Inputs

- Model: `spacer_640.pt`
- Analysis report: `outputs/model_analysis/spacer_640_model_analysis.md`
- Data yaml: `/home/paipaiqi01/workspace/data/spacer-merged_processed/data.yaml`
- Calibration list: `outputs/calib/spacer_train64.txt`

## Real Model Structure

High-level modules:

```text
model.0  Conv
model.1  Conv
model.2  C3k2
model.3  Conv
model.4  C3k2
model.5  Conv
model.6  C3k2
model.7  Conv
model.8  C3k2
model.9  SPPF
model.10 C2PSA
model.11 Upsample
model.12 Concat
model.13 C3k2
model.14 Upsample
model.15 Concat
model.16 C3k2
model.17 Conv
model.18 Concat
model.19 C3k2
model.20 Conv
model.21 Concat
model.22 C3k2
model.23 Detect
```

## Forward Output

The PyTorch model forward returns:

- postprocessed detections: `[1, 300, 6]`
- raw `one2many` dict:
  - `boxes`: `[1, 4, 8400]`
  - `scores`: `[1, 1, 8400]`
  - `feats`: `[1, 64, 80, 80]`, `[1, 128, 40, 40]`, `[1, 256, 20, 20]`
- raw `one2one` dict:
  - `boxes`: `[1, 4, 8400]`
  - `scores`: `[1, 1, 8400]`
  - `feats`: `[1, 64, 80, 80]`, `[1, 128, 40, 40]`, `[1, 256, 20, 20]`

GVSS should use the raw `one2many["scores"]` tensor as the first heatmap choice. It preserves all 8400 grid positions and maps naturally to the 80x80, 40x40, and 20x20 feature levels. The postprocessed `[1, 300, 6]` output is not suitable for GVSS because dense spatial correspondence has already been reduced.

## Recommended Hook Scope

Start with block-level output hooks to keep memory manageable:

- Backbone/neck semantic blocks:
  - `model.2`, `model.4`, `model.6`, `model.8`
  - `model.9`
  - `model.10`
  - `model.13`, `model.16`, `model.19`, `model.22`
- Detect branch modules:
  - `model.23.cv2.0`, `model.23.cv2.1`, `model.23.cv2.2`
  - `model.23.cv3.0`, `model.23.cv3.1`, `model.23.cv3.2`
  - `model.23.one2one_cv2.0`, `model.23.one2one_cv2.1`, `model.23.one2one_cv2.2`
  - `model.23.one2one_cv3.0`, `model.23.one2one_cv3.1`, `model.23.one2one_cv3.2`

For per-convolution quantization parameter collection, use the 242 candidate layers in `spacer_640_model_analysis.md`, but GVSS saliency should first be computed at the block-level hooks above.

## Phase 2 Implementation Notes

- Convert raw `scores` logits with `sigmoid()` before Top-K selection.
- Default Top-K: 50 grid positions over flattened `[8400]`.
- Auxiliary loss candidate: `binary_cross_entropy_with_logits(scores_topk, ones)`.
- Keep `retain_grad()` on hooked activation tensors; compute per-location L2 norm over channel dimension.
- Map flattened score indices back to levels using 80x80, 40x40, and 20x20 boundaries when level-specific heatmaps are needed.
