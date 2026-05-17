# RKNN INT8 量化问题排查与解决记录

## 1. 目标

本次目标是比较 `spacer.pt` / FP RKNN / INT8 RKNN 的精度和速度差异，并尝试使用 `rknn-toolkit2` 对 YOLO 检测模型做常规 PTQ INT8 量化。

最终可用结果如下：

```text
INT8 split-output RKNN, imgsz=640

mAP@0.5:      0.9593
mAP@0.5:0.95: 0.5648
Precision:    0.8545
Recall:       0.9789

Pre-process:  3.26 ms
Inference:    21.92 ms
Post-process: 0.50 ms
Total:        25.68 ms
FPS:          38.93
```

之前未量化的 640 RKNN 大约是：

```text
Backend: rknn_lite2
Target: rk3588
Images: 77
Pre-process: 3.34 ms
Inference: 46.33 ms
Post-process: 0.71 ms
Total: 50.39 ms
FPS: 19.85
```

所以最终量化后速度有提升，并且精度恢复到了可用水平。

## 2. 第一版 INT8 量化方法

最开始的量化脚本逻辑是：

```text
ONNX -> rknn.config() -> rknn.build(do_quantization=True, dataset=calib.txt) -> export .rknn
```

校准集来自 `data.yaml` 的 `train` split：

```bash
OMP_NUM_THREADS=1 python convert_to_rknn_int8.py \
  --onnx /root/ultralytics/spacerRKNNmodels/noQunati/spacer_640TO640_rknn_model/spacer_640.onnx \
  --output /root/ultralytics/spacerRKNNmodels/spacer_640to640_rknn/spacer_640_int8_nocustom.rknn \
  --data /root/autodl-tmp/spacer/data.yaml \
  --splits train \
  --calib-count 300 \
  --target rk3588
```

其中：

```text
--splits train
```

表示从训练集抽校准图片。

```text
--calib-count 300
```

表示随机抽 300 张图片生成 RKNN calibration dataset。

## 3. 遇到的第一个问题：custom_string 干扰

原始转换脚本里带了：

```python
custom_string="rknn_custom_op_head"
```

这个参数并不是所有 YOLO 模型都需要。它更像是针对特定自定义 head 或特定 RKNN 导出逻辑的配置。

因此第一步先把 `custom_string` 改成默认不启用：

```python
parser.add_argument("--custom-string", default=None)

config_kwargs = {
    "mean_values": [args.mean],
    "std_values": [args.std],
    "target_platform": args.target,
    "optimization_level": args.optimization_level,
}
if args.custom_string:
    config_kwargs["custom_string"] = args.custom_string
```

这样只有显式传入 `--custom-string` 时才启用它。

不过后续验证发现：去掉 `custom_string` 后，精度仍然是 0。因此它不是根因。

## 4. 核心问题现象：mAP 全 0

板端测试第一版 INT8 模型时，结果是：

```text
mAP@0.5:      0.0000
mAP@0.5:0.95: 0.0000
Precision:    0.0000
Recall:       0.0000
```

同时每张图：

```text
preds=0
```

这说明模型没有输出任何通过置信度阈值的检测框。

当时使用的阈值是：

```text
conf=0.001
iou=0.6
```

这是对齐 Ultralytics `model.val()` 的验证设置。`conf=0.001` 已经非常低，如果这种阈值下还是没有框，说明问题不只是普通精度下降。

## 5. 增加输出调试

为了定位问题，在 `rknn-infer-test.py` 中加入了输出调试：

```bash
--debug-outputs 3
--dump-output-dir /tmp/rknn_debug_outputs
```

调试输出包括：

```text
shape
dtype
min
max
mean
score_col min/max/mean
score_col >= 0.001 的数量
```

第一版 INT8 模型输出如下：

```text
output[0]: shape=(1, 5, 8400), dtype=float32,
           min=0.000000, max=652.650696, mean=153.502655

score_col: min=0.000000, max=0.000000, mean=0.000000, gt_conf=0
```

这条日志非常关键。

## 6. 为什么不是反量化问题

一开始怀疑过：

```text
INT8 输出没有反量化，导致解码失败
```

但实际输出是：

```text
dtype=float32
```

所以 Runtime 已经返回了 float32。问题不是脚本漏做 int8 -> float32 反量化。

真正问题是：

```text
输出中的 score 通道已经全部变成了 0
```

也就是说，分数分支在量化图里已经塌了。

## 7. 根因分析：box 和 score 共用一个输出 tensor

原始 ONNX 输出是：

```text
shape=(1, 5, 8400)
```

其中 5 个通道是：

```text
x, y, w, h, score
```

问题在于：

```text
x/y/w/h 的数值范围大约是 0 ~ 640
score 的数值范围大约是 0 ~ 1
```

如果 RKNN 对整个输出 tensor 做统一量化，那么量化尺度会被 `0~640` 的 box 坐标主导。

这会导致 `0~1` 的 score 通道精度极低，甚至全部被压成 0。

实际日志正好验证了这一点：

```text
score_col max=0.000000
gt_conf=0
```

因此 mAP 全 0 的根因是：

```text
box 坐标和 score 分数混在同一个输出 tensor 中，INT8 量化时 score 被坐标量程压没。
```

## 8. 尝试过但无效的方法：output_optimize

曾尝试在 RKNN config 中关闭输出优化：

```python
output_optimize=False
```

但结果仍然是：

```text
score_col max=0.000000
```

说明问题不是简单的输出优化开关导致，而是合并输出 tensor 的量化尺度问题。

## 9. 有效解决方法：拆分 ONNX 输出

解决思路是：

```text
不要让 box 和 score 共用同一个输出 tensor。
```

因此新增脚本 `split_yolo_output_onnx.py`，把原始输出：

```text
output0: [1, 5, 8400]
```

拆成两个输出：

```text
output0_boxes:  [1, 4, 8400]
output0_scores: [1, 1, 8400]
```

拆分命令：

```bash
python split_yolo_output_onnx.py \
  --input /root/ultralytics/spacerRKNNmodels/noQunati/spacer_640TO640_rknn_model/spacer_640.onnx \
  --output /root/ultralytics/spacerRKNNmodels/noQunati/spacer_640TO640_rknn_model/spacer_640_split.onnx
```

然后对拆分后的 ONNX 做 INT8 量化：

```bash
OMP_NUM_THREADS=1 python convert_to_rknn_int8.py \
  --onnx /root/ultralytics/spacerRKNNmodels/noQunati/spacer_640TO640_rknn_model/spacer_640_split.onnx \
  --output /root/ultralytics/spacerRKNNmodels/spacer_640to640_rknn/spacer_640_int8_split.rknn \
  --data /root/autodl-tmp/spacer/data.yaml \
  --splits train \
  --calib-count 300 \
  --target rk3588
```

## 10. 推理脚本适配双输出

因为 RKNN 模型现在有两个输出：

```text
boxes:  [1, 4, 8400]
scores: [1, 1, 8400]
```

所以 `rknn-infer-test.py` 中增加了双输出兼容逻辑：

```python
if len(outputs) == 2:
    first = np.asarray(outputs[0])
    second = np.asarray(outputs[1])
    if first.shape[1] == 4 and second.shape[1] >= 1:
        pred = np.concatenate([first, second], axis=1)
```

这样可以在后处理阶段重新拼成：

```text
[1, 5, 8400]
```

然后继续使用原来的解码、NMS、mAP 计算逻辑。

## 11. 修复后的验证结果

拆分输出后，板端调试日志变成：

```text
output[0]: shape=(1, 4, 8400), dtype=float32,
           min=4.079067, max=652.650696, mean=191.878311

output[1]: shape=(1, 1, 8400), dtype=float32,
           min=0.000000, max=0.860846, mean=0.001068

score_tensor: min=0.000000, max=0.860846, mean=0.001068, gt_conf=14
```

关键变化是：

```text
score max 从 0.000000 恢复到 0.860846
gt_conf 从 0 恢复到 14
preds 从 0 恢复到 1~2
```

最终精度：

```text
mAP@0.5:      0.9593
mAP@0.5:0.95: 0.5648
Precision:    0.8545
Recall:       0.9789
```

最终速度：

```text
Pre-process:  3.26 ms
Inference:    21.92 ms
Post-process: 0.50 ms
Total:        25.68 ms
FPS:          38.93
```

## 12. 本次量化方法总结

最终有效方法不是简单的：

```text
ONNX -> INT8 RKNN
```

而是：

```text
原始 ONNX
  -> 拆分输出 box/scores
  -> 使用 train split 抽样生成 calibration dataset
  -> rknn.build(do_quantization=True)
  -> RK3588 板端验证 mAP/FPS
```

核心点是：

```text
检测模型的最终输出如果把大范围 box 坐标和小范围 score 分数合在同一个 tensor 中，
直接 INT8 量化可能会导致 score 被压成 0。
```

解决原则是：

```text
让不同数值范围的输出分开量化，尤其是 box 和 score 不要共用一个输出 tensor。
```

## 13. 后续可继续优化的方向

可以继续试：

```text
calib-count 300 -> 500 / 800
splits train -> train val
```

例如：

```bash
OMP_NUM_THREADS=1 python convert_to_rknn_int8.py \
  --onnx /root/ultralytics/spacerRKNNmodels/noQunati/spacer_640TO640_rknn_model/spacer_640_split.onnx \
  --output /root/ultralytics/spacerRKNNmodels/spacer_640to640_rknn/spacer_640_int8_split_calib500.rknn \
  --data /root/autodl-tmp/spacer/data.yaml \
  --splits train val \
  --calib-count 500 \
  --target rk3588
```

目标是观察：

```text
mAP@0.5:0.95 是否恢复更多
Precision 是否提升
Inference 是否保持稳定
```

如果后续仍希望进一步压榨速度，可以继续尝试：

```text
更小输入尺寸
模型结构简化
混合量化
检测 head 保持 FP16
RGA/NPU/CPU 流水线
```

但本次最关键的问题已经解决：

```text
INT8 量化后 mAP=0 的原因是 score 输出被合并 tensor 的量化尺度压没。
拆分 ONNX 输出后，score 恢复，mAP 恢复，速度提升。
```
