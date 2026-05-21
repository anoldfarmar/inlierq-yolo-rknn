# Findings: InlierQ YOLO26 RKNN量化研究

## 1. InlierQ核心算法总结

### 1.1 问题定义
目标检测模型中，由于背景杂乱和传感器噪声，激活值中存在大量**任务无关的"Anomalies"（异常值）**。这些异常值：
- 显著扩大激活值动态范围
- 扭曲激活值分布，偏向任务无关响应
- 在低比特量化（如4-bit）下，宝贵的量化层级被浪费在异常值上

### 1.2 核心思想
将Anomalies与Inliers显式分离，量化误差优化**完全聚焦在Inliers上**。

### 1.3 三大核心组件

#### 组件1: 梯度感知体积显著性得分 (GVSS)
- 利用检测头输出的预测热力图，通过Top-K高置信度区域定义辅助损失
- 反向传播计算每个空间体积（Volume）的显著性得分
- 得分反映"任务相关性"，而非单纯激活值大小
- **关键公式（来自论文）**：
  设检测头输出热力图 H，取 Top-K 高置信度区域：
  $$L_{aux} = \sum_{k \in \text{Top-K}} \|H_k - \hat{H}_k\|^2$$
  对中间层激活值 A^l 的反向梯度：
  $$S^l_i = \left\|\frac{\partial L_{aux}}{\partial A^l_i}\right\|_2$$
  其中 S^l_i 是第 l 层第 i 个空间体积的显著性得分。

#### 组件2: EM + 概率混合模型分类
- 将显著性得分 S^l 视为两成分概率混合模型（Gaussian Mixture Model, GMM）
- 两个成分：Inlier分布（低显著性但任务关键）与Anomaly分布（高显著性但任务无关的杂波，或极低显著性的背景）
- 通过EM算法无监督拟合，输出每个体积属于Inlier的后验概率 p(I|S_i)
- **关键公式**：
  混合模型：$$p(S) = \pi_{in}\mathcal{N}(S|\mu_{in},\sigma^2_{in}) + \pi_{an}\mathcal{N}(S|\mu_{an},\sigma^2_{an})$$
  E步——计算责任度（responsibility）：
  $$\gamma_{i,in} = \frac{\pi_{in}\mathcal{N}(S_i|\mu_{in},\sigma^2_{in})}{p(S_i)}$$
  M步——更新参数：$$\mu_k = \frac{\sum_i \gamma_{i,k}S_i}{\sum_i \gamma_{i,k}}, \quad \sigma^2_k = \frac{\sum_i \gamma_{i,k}(S_i-\mu_k)^2}{\sum_i \gamma_{i,k}}, \quad \pi_k = \frac{\sum_i \gamma_{i,k}}{N}$$
  收敛后，根据后验概率阈值确定Inlier集合 I^l。

#### 组件3: 以留存物为中心的量化优化
- 量化参数（scale S^l, zero-point z^l）的优化**仅使用Inlier集合**
- 使用Hessian矩阵引导的量化误差最小化，丢弃异常值的贡献
- 初始量化参数由MinMax在Inlier范围内确定，再通过梯度下降微调
- **关键公式**：
  量化误差目标函数：
  $$
  $$L_{quant} = \sum_{i \in I^l} \|Q(A_i) - A_i\|^2_{H} = \sum_{i \in I^l} (Q(A_i) - A_i)^T H_i (Q(A_i) - A_i)$$
  $$
  其中 H_i 为Hessian矩阵（通常用输出重建误差的二阶导数近似），Q(·)为量化函数。
  用梯度下降迭代更新 scale S^l：
  $$
  S^l \leftarrow S^l - \eta \cdot \nabla_{S^l} L_{quant}
  $$
  

---

## 2. 用户原有量化方法（Baseline）完整分析

### 2.1 管线流程
```
spacer_640.pt (FP32)
  → ultralytics YOLO.export() → spacer_640.onnx [1,5,8400]
  → split_yolo_output_onnx.py → spacer_640_split.onnx [1,4,8400] + [1,1,8400]
  → convert_to_rknn_int8.py:
      rknn.config(mean_values, std_values, target="rk3588")
      rknn.load_onnx()
      rknn.build(do_quantization=True, dataset=calib.txt)  ← 300张校准图
      rknn.export_rknn()
  → spacer_640_int8.rknn
```

### 2.2 量化方式
- **完全依赖 rknn-toolkit2 内置校准**（MinMax 或 KL-divergence）
- 校准集: 300张从train split随机抽取的图片
- 不做任何per-layer优化或Anomaly过滤

### 2.3 关键bug fix: 输出拆分
- 原因: box坐标(0~640)和score(0~1)混在同一tensor [1,5,8400] 中，INT8量化时score被压为0
- 解决: 拆分为 boxes[1,4,8400] + scores[1,1,8400]，各自独立量化

### 2.4 基线性能（RK3588, imgsz=640）
```
mAP@0.5:      0.9593
mAP@0.5:0.95: 0.5648
Precision:    0.8545
Recall:       0.9789
Inference:    21.92 ms  (FPS: 38.93)
```

---

## 3. RKNN技术栈关键发现

### 3.1 核心发现: ONNX Q/DQ算子支持 ✅
**rknn-toolkit2 2.3.x 支持 `QuantizeLinear` 和 `DequantizeLinear` 算子**
- 这两个算子是ONNX Q/DQ模式的基础
- 意味着: PyTorch FakeQuantize → ONNX Q/DQ → rknn-toolkit2 这条路**理论可行**
- 不支持: `QLinearConv`, `DynamicQuantizeLinear` 等整数算子
- 自v2.0.0起官方声明 "Improved support for QAT models of pytorch and onnx"

### 3.2 混合量化API（备选方案）
rknn-toolkit2提供 `hybrid_quantization_step1/step2`:
- Step1生成 `.quantization.cfg` (YAML)可逐层指定量化类型
- 支持的量化类型: `dynamic_fixed_point-i8`, `dynamic_fixed_point-i16`, `float16`, `float32`
- **限制**: 只能控制量化类型，不能直接指定自定义 scale/zp 值

### 3.3 确定的技术路径: PyTorch InlierQ → ONNX Q/DQ → RKNN
```
[PC端] InlierQ校准(PyTorch, 64张spacer图)
  → 计算per-layer scale/zp
  → torch.quantization.FakeQuantize 插入
  → torch.onnx.export(opset=13+)
  → ONNX with Q/DQ nodes
        ↓
[PC端] rknn-toolkit2:
  → rknn.load_onnx(Q/DQ ONNX)
  → rknn.build(do_quantization=False)  # 模型已量化
  → rknn.export_rknn()
        ↓
[RK3588] rknn-lite2 推理评测
```

---

## 4. Spacer数据集信息

### 4.1 数据结构
```
D:/Study/rh/2026_01/RS/导线间隔器/导线间隔器/spacer-merged_processed/
├── images/
│   ├── train/  (569张)
│   ├── val/    (80张)
│   └── test/   (77张)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### 4.2 场景特点
- 单类别检测（spacer/间隔棒）
- 背景：输电线路、天空、杆塔、导线
- Anomaly来源：复杂电网背景（导线纹理、绝缘子、杆塔结构）
- 与COCO差异：场景专注、单类、背景结构化

### 4.3 对InlierQ算法的影响分析
- **单类别GVSS**: 论文依赖多类别热力图梯度。单类spacer场景，GVSS信号来源于spacer vs 背景的置信度差异 → 理论上仍有效，但区分度可能降低
- **结构化背景**: 电网场景的Anomaly呈规律性分布（如导线沿线）→ EM-GMM的Gaussian假设可能需要验证
- **小目标风险**: 如果spacer在640×640图中像素占比小 → Inlier比例低，Anomaly比例高

---

## 5. InlierQ适配YOLO26的关键设计

### 5.1 GVSS计算适配
YOLO26导出ONNX时输出可拆为 `[1, 4, 8400]` boxes 与 `[1, 1, 8400]` scores；PyTorch forward在当前Ultralytics版本中返回 postprocessed `[1, 300, 6]` 以及 raw dict。

GVSS实现方案:
1. 前向传播 → 获取 raw `one2many["scores"]` logits，shape `[1, 1, 8400]`，作为热力图 H。
2. Top-K选择: 对 `scores.sigmoid()` 取 K 个最高 confidence 的grid cell。
3. L_aux = BCEWithLogits(scores[Top-K], 1)  (自监督: 用模型自身的高置信度预测做伪标签)。
4. backward → 各层 hook 捕获 ∂L_aux/∂A^l。
5. 逐空间位置计算 L2 norm → S^l。

### 5.2 YOLO26层结构 (已实际加载模型确认)
模型高层结构为:
`Conv, Conv, C3k2, Conv, C3k2, Conv, C3k2, Conv, C3k2, SPPF, C2PSA, Upsample, Concat, C3k2, Upsample, Concat, C3k2, Conv, Concat, C3k2, Conv, Concat, C3k2, Detect`。

推荐Hook范围见 `outputs/model_analysis/spacer_640_hook_plan.md`:
- GVSS显著性计算优先使用 block-level hooks: `model.2/4/6/8/9/10/13/16/19/22` 与 Detect分支 `model.23.cv2/cv3/one2one_cv2/one2one_cv3` 各尺度模块。
- per-conv量化参数统计可使用模型报告中的242个候选Conv/Detect相关层。
- GVSS smoke验证中，score-only辅助损失会让Detect score分支`cv3.*`产生梯度，而box分支`cv2.*`无梯度；backbone/neck仍可获得有效显著性。若后续希望box分支也参与Inlier划分，需要额外加入box相关辅助损失。
- 64张校准图GVSS聚合后，13个有效层产生显著性分布。常规层的GMM高显著性inlier比例大约在20%-44%；Detect score分支较稀疏，全层inlier比例约0.3%-0.7%。
- 对非零显著性比例低于10%的稀疏层，EM-GMM只在非零S^l上拟合，零梯度位置直接视为non-inlier，避免大量零值让两个高斯成分退化。
- Inlier activation MinMax初始化已完成，13个有效Hook层均输出量化参数初值。常规层activation范围通常包含SiLU负值（约-0.278）和正激活峰值；Detect score分支的Inlier activation是负logits，u8 asymmetric zero-point会饱和到255，提示后续score分支更适合使用symmetric int8或在输出拆分/sigmoid后单独处理。
- 基于Inlier activation样本的symmetric int8 scale grid-search微调已完成。常规层相对MinMax scale通常更小，Inlier重建MSE下降明显；这说明只围绕Inlier范围优化量化分辨率是有效的。该实现目前是重建误差近似，不是论文完整的Hessian/Fisher优化。
- diagonal-Fisher近似scale微调已完成。实现中使用GVSS saliency平方作为每个空间位置的Fisher proxy，仅对EM-GMM判定的Inlier激活加权，并对log(scale)使用Adam迭代优化weighted reconstruction MSE。该方法不是完整Hessian矩阵，但已经补齐Phase 2中“对角Fisher近似 + 梯度下降更新scale”的可运行版本。
- 已生成Phase 3可消费的统一量化配置 `outputs/quant/inlierq_quant_config.json`。该配置优先采用Fisher-refined symmetric int8 activation scale，保留grid-search scale、u8 asymmetric参数和per-channel symmetric参考参数；Detect score logits层继续选择symmetric int8，避免u8 zero-point饱和到255。
- Phase 3的ONNX导出和RKNN build属于当前管线中最吃CPU/内存的步骤。已按用户要求只实现脚本和轻量dry-run，不在Codex中执行重导出/转换。运行指令集中记录在 `plan/phase3_runbook.md`。
- `fake_quantize.py`的轻量dry-run已验证：13个Phase 2配置层全部找到并包装，126个Conv2d权重完成per-channel symmetric fake quant，一次dummy forward正常。需要注意：真正导出后必须用 `check_onnx_qdq.py` 确认ONNX里存在QuantizeLinear/DequantizeLinear节点；若节点数量为0，不能进入RKNN转换。
- 当前 `yolo26` 环境缺少ONNX检查相关依赖（`onnx/onnxruntime/tqdm`），而 `rknn` 环境已有 `rknn/onnx/tqdm`。因此建议在运行Phase 3导出前先给 `yolo26` 安装 `onnx onnxruntime tqdm`。
- 用户实际导出的 `outputs/onnx/spacer_640_inlierq_qdq.onnx` 已通过ONNX checker，包含10个QuantizeLinear和10个DequantizeLinear节点。当前Q/DQ节点覆盖backbone/neck的10个block-level层；Detect score层 `model.23.cv3.*` 没有出现在Ultralytics导出的Q/DQ节点中，这说明Ultralytics export路径可能没有保留这些wrapper，后续若RKNN精度不理想，需要考虑torch export后端或对ONNX图做显式Q/DQ插入。
- RKNN转换失败并非Q/DQ节点错误，而是工具链兼容问题：rknn-toolkit2 2.3.2内部依赖旧版 `onnx.mapping`，但当前 `rknn` 环境ONNX 1.21只保留 `onnx._mapping`。已在 `convert_qdq_rknn.py` 中添加脚本级compat shim，补回 `TENSOR_TYPE_TO_NP_TYPE` 与 `NP_TYPE_TO_TENSOR_TYPE`；若后续仍失败，建议将 `rknn` 环境ONNX降级到1.14.1。
- RKNN在加载QAT/QDQ模型时提示 `optimization_level=3` 的部分优化可能影响精度。为避免优化pass破坏Q/DQ精度，建议先用 `optimization_level=2` 生成保守版本，再额外跑level 3做性能/精度对比。
- Phase 3已实际转换成功，得到两个InlierQ RKNN模型：`spacer_640_inlierq_opt2.rknn` 和 `spacer_640_inlierq.rknn`。opt2文件更大，预期更保守；opt3文件更小，预期速度/图优化更强但精度需实测确认。
- 用户提供的 `eval/rknn-infer-test.py` 可以作为Phase 4统一评测基础，但原版本会误解InlierQ导出的postprocessed `[1,300,6]` 输出。已补充分支：postprocessed输出按 `xyxy, score, class` 解码，raw/split输出继续按YOLO raw head解码。Phase 4评测时应对InlierQ模型使用 `--output-format postprocessed` 或默认 `auto`，对baseline split模型使用 `--output-format raw` 或默认 `auto`。
- 四模型RK3588实测已完成，结果见 `eval/ModelComparisonResults.xlsx`。InlierQ取得最高精度：`mAP@0.5=0.9937`、`mAP@0.5:0.95=0.6596`，但FPS仅`17.04`。相比未量化baseline的`mAP@0.5:0.95=0.6483`，相对提升约`1.74%`；相比`int8_split_baseline`的`FPS=36.15`，速度下降约`52.86%`。当前部署收益不划算。
- 对当前单类别spacer检测任务，InlierQ优势没有充分体现。可能原因包括：任务为单类别且模型本身精度已高、异常值/背景对量化误差的破坏不够强、当前Q/DQ导出只保留了backbone/neck 10个block-level Q/DQ而没有覆盖Detect分支、InlierQ模型走postprocessed输出路径而baseline走split raw输出路径。若继续研究，建议优先做Detect分支显式ONNX Q/DQ插入、raw split输出一致化、多类别/更复杂背景任务验证。

### 5.3 输出拆分
InlierQ在PyTorch侧处理最后输出层时，可能遇到与baseline相同的问题（box/score范围差异）。
**方案**: 继续沿用 split-output 策略，或在PyTorch侧对输出层做per-channel量化（而非per-tensor）。

---

## 6. 实现架构

```
D:/Study/rh/2026_05/paper/inlierq_rknn/
├── calibrate/
│   ├── gvss.py              # GVSS显著性得分计算（Hook + 梯度）
│   ├── em_gmm.py            # EM-GMM聚类模块
│   └── inlier_optimizer.py  # Inlier-aware Hessian引导量化优化
├── quantize/
│   ├── export_quant_config.py # 汇总Phase 2量化参数，生成Phase 3统一配置
│   ├── fake_quantize.py     # PyTorch FakeQuantize插入 + 导出
│   └── check_onnx_qdq.py    # ONNX Q/DQ节点检查脚本
├── rknn/
│   ├── convert_qdq_rknn.py  # Q/DQ ONNX → RKNN转换
│   └── eval_rk3588.py       # RK3588评测脚本
├── data/
│   └── prepare_calib.py     # 校准集准备（从spacer数据集抽取64张）
└── compare/
    └── benchmark.py         # Basline vs InlierQ对比脚本
```
