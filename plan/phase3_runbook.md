# Phase 3 Runbook: InlierQ ONNX Q/DQ 与 RKNN 转换

## 说明
ONNX导出和RKNN build是当前阶段最吃CPU/内存的步骤，建议由用户在终端中执行。脚本已加入阶段日志和进度输出；如果日志停在“Export ONNX”或“Build RKNN”，通常表示底层导出/图优化仍在运行。

## 0. 依赖检查
当前已确认：
- `yolo26`: 有 torch/ultralytics，但缺少 `onnx`、`onnxruntime`、`tqdm`
- `rknn`: 有 rknn-toolkit2、onnx、tqdm

建议先在 `yolo26` 中补齐ONNX检查依赖：

```bash
conda run -n yolo26 python -m pip install onnx onnxruntime tqdm
```

## 1. 轻量dry-run验证
不会导出ONNX，只验证量化配置、层映射、权重fake quant和一次dummy forward：

```bash
cd /home/paipaiqi01/workspace/rknnquantizationLearning
conda run -n yolo26 python inlierq_rknn/quantize/fake_quantize.py \
  --model spacer_640.pt \
  --quant-config outputs/quant/inlierq_quant_config.json \
  --imgsz 640 \
  --device cpu \
  --dry-run
```

期望看到：
- `Wrapped activation layers: 13 / 13`
- `Fake-quantized Conv2d weights: 126`
- `Smoke forward output type: tuple`

## 2. 导出ONNX Q/DQ候选模型
默认使用Ultralytics导出后端，尽量保持YOLO导出行为：

```bash
cd /home/paipaiqi01/workspace/rknnquantizationLearning
conda run -n yolo26 python inlierq_rknn/quantize/fake_quantize.py \
  --model spacer_640.pt \
  --quant-config outputs/quant/inlierq_quant_config.json \
  --output outputs/onnx/spacer_640_inlierq_qdq.onnx \
  --imgsz 640 \
  --opset 13 \
  --device cpu \
  --export-backend ultralytics
```

如果Ultralytics导出后检查不到Q/DQ节点，可再尝试torch后端：

```bash
conda run -n yolo26 python inlierq_rknn/quantize/fake_quantize.py \
  --model spacer_640.pt \
  --quant-config outputs/quant/inlierq_quant_config.json \
  --output outputs/onnx/spacer_640_inlierq_qdq_torch.onnx \
  --imgsz 640 \
  --opset 13 \
  --device cpu \
  --export-backend torch
```

## 3. 检查ONNX是否包含Q/DQ节点

```bash
cd /home/paipaiqi01/workspace/rknnquantizationLearning
conda run -n yolo26 python inlierq_rknn/quantize/check_onnx_qdq.py \
  --onnx outputs/onnx/spacer_640_inlierq_qdq.onnx \
  --summary outputs/onnx/spacer_640_inlierq_qdq_summary.json \
  --check-model
```

必须看到：
- `quantize_linear_count > 0`
- `dequantize_linear_count > 0`

如果这两个值为0，先不要转RKNN，需要回到fake-quant导出方式调整。

## 4. 转换为RKNN
Q/DQ ONNX通常不需要RKNN再次量化，因此默认 `do_quantization=False`：

```bash
cd /home/paipaiqi01/workspace/rknnquantizationLearning
conda run -n rknn python inlierq_rknn/rknn/convert_qdq_rknn.py \
  --onnx outputs/onnx/spacer_640_inlierq_qdq.onnx \
  --output outputs/rknn/spacer_640_inlierq.rknn \
  --target rk3588 \
  --optimization-level 3 \
  --verbose
```

如果RKNN明确报Q/DQ模型不接受 `do_quantization=False`，再做一次调试性尝试：

```bash
conda run -n rknn python inlierq_rknn/rknn/convert_qdq_rknn.py \
  --onnx outputs/onnx/spacer_640_inlierq_qdq.onnx \
  --output outputs/rknn/spacer_640_inlierq_requant_debug.rknn \
  --target rk3588 \
  --optimization-level 3 \
  --do-quantization \
  --dataset outputs/calib/spacer_train64.txt \
  --verbose
```

这一步可能二次量化，仅用于定位问题，不作为首选实验结果。

## 5. 已知问题：ONNX 1.21与RKNN 2.3.2
如果看到：

```text
AttributeError: module 'onnx' has no attribute 'mapping'
```

这是 `rknn` 环境中的 ONNX 版本过新导致的兼容问题，不是Q/DQ模型本身错误。`convert_qdq_rknn.py` 已加入兼容shim，会在导入RKNN前补回 `onnx.mapping.TENSOR_TYPE_TO_NP_TYPE` 和 `onnx.mapping.NP_TYPE_TO_TENSOR_TYPE`。

修复后直接重跑第4步转换命令即可。若仍失败，备选方案是在 `rknn` 环境中降级ONNX：

```bash
conda run -n rknn python -m pip install "onnx==1.14.1"
```

另外，RKNN可能提示QAT模型在 `optimization_level=3` 下部分优化会影响精度。建议先用更保守的level 2跑一个版本：

```bash
conda run -n rknn python inlierq_rknn/rknn/convert_qdq_rknn.py \
  --onnx outputs/onnx/spacer_640_inlierq_qdq.onnx \
  --output outputs/rknn/spacer_640_inlierq_opt2.rknn \
  --target rk3588 \
  --optimization-level 2 \
  --verbose
```

如果level 2能通过，再跑level 3做速度/精度对比。

## 6. 当前转换结果
用户已成功生成：

```text
outputs/rknn/spacer_640_inlierq_opt2.rknn  # optimization-level=2
outputs/rknn/spacer_640_inlierq.rknn       # optimization-level=3
```

Phase 3主链路已完成，后续进入Phase 4评测对比。
