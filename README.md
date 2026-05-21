# InlierQ YOLO26 RKNN Quantization for Spacer Detection

本项目尝试将 InlierQ（Inlier-Centric Post-Training Quantization）思想适配到 YOLO26 spacer 单类别检测模型，并完成从 PyTorch `.pt` 到 ONNX Q/DQ、RKNN、RK3588 实测对比的完整实验链路。

参考论文：INLIER-CENTRIC POST-TRAINING QUANTIZATION  
https://arxiv.org/pdf/2602.03472

## 结论

当前实验结论比较直接：这版 InlierQ 在 spacer 单类别检测任务上带来的精度收益很有限，且速度损失明显，因此不建议替换现有 `int8_split_baseline` 作为实际部署方案。

四模型 RK3588 test split 结果见 `eval/ModelComparisonResults.xlsx`：

| 模型 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference(ms) | Total(ms) | FPS |
|------|---------|---------------|-----------|--------|---------------|-----------|-----|
| baseline（pt直接转RKNN，未量化） | 0.9443 | 0.6483 | 0.9715 | 0.9583 | 48.53 | 53.22 | 18.79 |
| int8_split_baseline（RKNN内置INT8 + output split） | 0.9593 | 0.5648 | 0.8545 | 0.9789 | 23.21 | 27.66 | 36.15 |
| inlierq_opt2（Q/DQ, optimization-level=2） | 0.9936 | 0.6556 | 0.9860 | 0.9774 | 63.93 | 68.02 | 14.70 |
| inlierq（Q/DQ, optimization-level=3） | 0.9937 | 0.6596 | 1.0000 | 0.9685 | 54.66 | 58.70 | 17.04 |

要点：

- InlierQ 的 `mAP@0.5:0.95=0.6596`，相比未量化 baseline 的 `0.6483` 只有约 `+1.74%` 相对提升。
- InlierQ 的 `FPS=17.04`，相比 `int8_split_baseline` 的 `36.15` 下降约 `52.86%`。
- 对当前单目标 spacer 检测，部署优先级应是 `int8_split_baseline`，不是 InlierQ Q/DQ 版本。

可能原因：

- spacer 是单类别任务，模型原始精度已经较高，InlierQ 的异常值过滤优势不明显。
- 当前 Q/DQ 只保留了 backbone/neck 的 10 个 block-level 节点，Detect score 分支没有被 Ultralytics 导出路径保留下来。
- InlierQ RKNN 输出为 postprocessed `[1,300,6]`，而 int8 baseline 使用 raw split 输出，后处理路径不完全一致。
- RKNN 对 QAT/Q/DQ 模型的图优化可能额外影响性能收益。

## 项目结构

```text
.
├── inlierq_rknn/
│   ├── analyze_model.py
│   ├── calibrate/
│   │   ├── gvss.py
│   │   ├── em_gmm.py
│   │   ├── inlier_optimizer.py
│   │   ├── scale_refine.py
│   │   └── fisher_scale_refine.py
│   ├── data/
│   │   └── prepare_calib.py
│   ├── quantize/
│   │   ├── export_quant_config.py
│   │   ├── fake_quantize.py
│   │   └── check_onnx_qdq.py
│   └── rknn/
│       └── convert_qdq_rknn.py
├── org-rknn-quantization/
│   ├── convert_to_rknn_int8.py
│   ├── split_yolo_output_onnx.py
│   └── pt2rknn.py
├── eval/
│   ├── rknn-infer-test.py
│   └── ModelComparisonResults.xlsx
├── outputs/
│   ├── calib/
│   ├── gvss/
│   ├── model_analysis/
│   ├── onnx/
│   ├── quant/
│   └── rknn/
├── paper/
└── plan/
```

## 环境

已使用的主要环境：

- `yolo26`: PyTorch / Ultralytics / ONNX 导出
- `rknn`: rknn-toolkit2 2.3.2 / RKNN 转换
- RK3588: RKNN/RKNNLite 实测推理

数据集路径：

```text
/home/paipaiqi01/workspace/data/spacer-merged_processed/data.yaml
```

模型：

```text
spacer_640.pt
```

## 实验流程

### Phase 1: 模型分析与校准集

生成模型结构报告：

```bash
conda run -n yolo26 python inlierq_rknn/analyze_model.py \
  --model spacer_640.pt \
  --output outputs/model_analysis/spacer_640_model_analysis.json \
  --markdown outputs/model_analysis/spacer_640_model_analysis.md
```

准备 64 张校准图：

```bash
conda run -n yolo26 python inlierq_rknn/data/prepare_calib.py \
  --data /home/paipaiqi01/workspace/data/spacer-merged_processed/data.yaml \
  --split train \
  --count 64 \
  --output outputs/calib/spacer_train64.txt
```

### Phase 2: InlierQ 校准

计算 GVSS：

```bash
conda run -n yolo26 python inlierq_rknn/calibrate/gvss.py \
  --model spacer_640.pt \
  --calib outputs/calib/spacer_train64.txt \
  --summary-output outputs/gvss/gvss_train64_summary.json \
  --saliency-output outputs/gvss/gvss_train64_saliency.npz
```

EM-GMM 生成 Inlier mask：

```bash
conda run -n yolo26 python inlierq_rknn/calibrate/em_gmm.py \
  --saliency outputs/gvss/gvss_train64_saliency.npz \
  --summary-output outputs/gvss/em_gmm_train64_summary.json \
  --masks-output outputs/gvss/em_gmm_train64_masks.npz \
  --posteriors-output outputs/gvss/em_gmm_train64_posteriors.npz
```

生成 Inlier-aware 量化参数：

```bash
conda run -n yolo26 python inlierq_rknn/calibrate/inlier_optimizer.py \
  --model spacer_640.pt \
  --calib outputs/calib/spacer_train64.txt \
  --masks outputs/gvss/em_gmm_train64_masks.npz \
  --output outputs/quant/inlier_activation_qparams.json
```

scale 微调：

```bash
conda run -n yolo26 python inlierq_rknn/calibrate/scale_refine.py \
  --model spacer_640.pt \
  --calib outputs/calib/spacer_train64.txt \
  --masks outputs/gvss/em_gmm_train64_masks.npz \
  --qparams outputs/quant/inlier_activation_qparams.json \
  --output outputs/quant/inlier_activation_scale_refined.json
```

diagonal-Fisher proxy 微调：

```bash
conda run -n yolo26 python inlierq_rknn/calibrate/fisher_scale_refine.py \
  --model spacer_640.pt \
  --calib outputs/calib/spacer_train64.txt \
  --masks outputs/gvss/em_gmm_train64_masks.npz \
  --saliency outputs/gvss/gvss_train64_saliency.npz \
  --qparams outputs/quant/inlier_activation_qparams.json \
  --start-scales outputs/quant/inlier_activation_scale_refined.json \
  --output outputs/quant/inlier_activation_fisher_refined.json
```

导出统一量化配置：

```bash
conda run -n yolo26 python inlierq_rknn/quantize/export_quant_config.py \
  --qparams outputs/quant/inlier_activation_qparams.json \
  --refined outputs/quant/inlier_activation_scale_refined.json \
  --fisher outputs/quant/inlier_activation_fisher_refined.json \
  --output outputs/quant/inlierq_quant_config.json
```

### Phase 3: ONNX Q/DQ 与 RKNN

导出 ONNX Q/DQ 候选模型：

```bash
conda run -n yolo26 python inlierq_rknn/quantize/fake_quantize.py \
  --model spacer_640.pt \
  --quant-config outputs/quant/inlierq_quant_config.json \
  --output outputs/onnx/spacer_640_inlierq_qdq.onnx \
  --imgsz 640 \
  --opset 13 \
  --device cpu \
  --export-backend ultralytics
```

检查 Q/DQ 节点：

```bash
conda run -n yolo26 python inlierq_rknn/quantize/check_onnx_qdq.py \
  --onnx outputs/onnx/spacer_640_inlierq_qdq.onnx \
  --summary outputs/onnx/spacer_640_inlierq_qdq_summary.json \
  --check-model
```

转换 RKNN：

```bash
conda run -n rknn python inlierq_rknn/rknn/convert_qdq_rknn.py \
  --onnx outputs/onnx/spacer_640_inlierq_qdq.onnx \
  --output outputs/rknn/spacer_640_inlierq.rknn \
  --target rk3588 \
  --optimization-level 3 \
  --verbose
```

已生成：

```text
outputs/rknn/spacer_640_inlierq_opt2.rknn
outputs/rknn/spacer_640_inlierq.rknn
```

### Phase 4/5: 评测

统一评测脚本：

```bash
python eval/rknn-infer-test.py \
  --rknn outputs/rknn/spacer_640_inlierq.rknn \
  --data /home/paipaiqi01/workspace/data/spacer-merged_processed/data.yaml \
  --split test \
  --imgsz 640 \
  --output-format postprocessed
```

baseline split 模型使用：

```bash
python eval/rknn-infer-test.py \
  --rknn /path/to/baseline_int8_split.rknn \
  --data /home/paipaiqi01/workspace/data/spacer-merged_processed/data.yaml \
  --split test \
  --imgsz 640 \
  --output-format raw
```

## 重要产物

```text
outputs/model_analysis/spacer_640_model_analysis.md
outputs/model_analysis/spacer_640_hook_plan.md
outputs/calib/spacer_train64.txt
outputs/gvss/gvss_train64_summary.json
outputs/gvss/em_gmm_train64_summary.json
outputs/quant/inlierq_quant_config.json
outputs/onnx/spacer_640_inlierq_qdq.onnx
outputs/onnx/spacer_640_inlierq_qdq_summary.json
outputs/rknn/spacer_640_inlierq_opt2.rknn
outputs/rknn/spacer_640_inlierq.rknn
eval/ModelComparisonResults.xlsx
```

## 局限与后续方向

这次实现已经打通 InlierQ 思想到 RKNN 的端到端流程，但不代表它适合当前部署任务。

可继续研究的方向：

- 在 ONNX 图上对 Detect score 分支显式插入 Q/DQ，避免 Ultralytics export 丢失 wrapper。
- 让 InlierQ 与 baseline 使用一致的 raw split 输出，减少后处理路径差异。
- 尝试更多类别、更复杂背景或量化更敏感的检测任务。
- 对 RKNN `optimization_level=2/3` 做更细的精度和速度剖析。
- 实现更接近论文的完整 Hessian/Fisher 优化，而不是当前 diagonal-Fisher proxy。

当前部署建议：继续使用 `int8_split_baseline`。
