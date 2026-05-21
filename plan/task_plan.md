# Task Plan: 基于InlierQ的YOLO26 RKNN量化部署 — Spacer检测

## 目标
将InlierQ算法应用于已训练的YOLO26 spacer检测模型，实现 PyTorch(.pt) → InlierQ校准 → ONNX(Q/DQ) → RKNN(.rknn) → RK3588部署，并与原有rknn-toolkit2默认量化方法进行mAP和推理速度对比评测。

## 当前阶段
Completed / 实验结论归档

## 下一步
当前InlierQ适配实验已完成。结论：在单类别spacer检测场景中，本项目实现的InlierQ Q/DQ RKNN仅带来有限精度收益，但吞吐明显下降；推荐部署仍优先使用 `int8_split_baseline`。后续若继续研究，应重点尝试Detect分支显式Q/DQ、raw split输出Q/DQ，以及多类别/更复杂背景场景。

## 环境信息（已确认）
| 项目 | 值 |
|------|-----|
| 模型 | `spacer_640.pt` (YOLO26, FP32, imgsz=640) |
| 数据集 | spacer-merged_processed: train=569 / val=80 / test=77 |
| rknn-toolkit2 | 2.3.2 |
| 目标平台 | RK3588 (SSH连接) |
| 目标精度 | W8A8 (INT8权重 + INT8激活) |
| 原有量化方法 | rknn-toolkit2内置校准 (MinMax/KL, 300张校准图, output split) |
| 基线性能 | mAP@0.5=0.9593, mAP@0.5:0.95=0.5648, FPS=38.93 |

## 最终评测结果（RK3588, test split, imgsz=640）
结果表来源: `eval/ModelComparisonResults.xlsx`

| 模型 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference(ms) | Total(ms) | FPS |
|------|---------|---------------|-----------|--------|---------------|-----------|-----|
| baseline（pt直接转RKNN，未量化） | 0.9443 | 0.6483 | 0.9715 | 0.9583 | 48.53 | 53.22 | 18.79 |
| int8_split_baseline（RKNN内置INT8 + output split） | 0.9593 | 0.5648 | 0.8545 | 0.9789 | 23.21 | 27.66 | 36.15 |
| inlierq_opt2（Q/DQ, optimization-level=2） | 0.9936 | 0.6556 | 0.9860 | 0.9774 | 63.93 | 68.02 | 14.70 |
| inlierq（Q/DQ, optimization-level=3） | 0.9937 | 0.6596 | 1.0000 | 0.9685 | 54.66 | 58.70 | 17.04 |

### 结论
- InlierQ在精度上最好，`mAP@0.5:0.95=0.6596`，相比未量化baseline的`0.6483`提升约`+1.74%`相对提升。
- 但InlierQ FPS仅`17.04`，相比`int8_split_baseline`的`36.15`下降约`52.86%`。
- 对当前单目标spacer检测部署，InlierQ的精度收益不足以抵消速度损失；推荐实际部署使用`int8_split_baseline`。

## 阶段规划

### Phase 1: 环境搭建与模型分析
- [x] 确认PC端PyTorch环境（torch版本、ultralytics版本、CUDA可用性）
  - Windows: `D:\Anaconda\envs\yolov8\python.exe` 可启动，且安装了 torch/ultralytics
  - 阻塞: Windows 导入 torch 时 `_ctypes` DLL load failed: 拒绝访问
  - 结论: 模型结构分析建议切到用户已配置好的 WSL 环境执行
- [x] 加载spacer_640.pt，分析YOLO26完整层结构（层名、类型、参数形状）
  - 已新增脚本: `inlierq_rknn/analyze_model.py`
  - 已生成结构报告: `outputs/model_analysis/spacer_640_model_analysis.md`
- [x] 确定需要Hook的层范围（Conv2d、C3k2/C2PSA/SPPF输出、Detect各分支）
  - 已生成Hook计划: `outputs/model_analysis/spacer_640_hook_plan.md`
- [x] 确认检测头输出格式和GVSS热力图计算的具体方式
  - raw输出包含`one2many["scores"]`和`one2one["scores"]`: `[1, 1, 8400]`
  - GVSS优先使用`one2many["scores"]` logits，Top-K前先做sigmoid或直接用BCEWithLogitsLoss
- [x] 准备校准集：从spacer train split随机抽取64张（对齐论文配置）
  - 已生成: `outputs/calib/spacer_train64.txt`
  - 抽样脚本: `inlierq_rknn/data/prepare_calib.py`
- **状态:** completed

### Phase 2: InlierQ核心算法PyTorch实现
- [x] 实现GVSS模块
  - [x] Hook注册到block-level目标层，捕获FP32激活值
  - [x] 检测头raw `one2many["scores"]`作为热力图 H
  - [x] Top-K高置信度区域选择（K值可配置，默认50）
  - [x] L_aux定义：BCEWithLogits(scores_topk, 1)
  - [x] backward → 逐空间位置梯度L2范数 → S^l（单张smoke已通过）
  - [x] 遍历64张校准图并聚合/保存每层S^l分布
- [x] 实现EM-GMM聚类模块
  - [x] 两成分Gaussian Mixture Model
  - [x] EM迭代（E步责任度 + M步参数更新）
  - [x] 收敛判定：参数变化 < 1e-4 或 max_iter=100
  - [x] 输出Inlier集合 I^l（后验概率 > 0.5）
  - [x] 对非零显著性低于10%的稀疏层，仅在非零S^l上拟合GMM，零梯度位置视为non-inlier
- [x] 实现Inlier-aware量化参数优化
  - [x] MinMax初始化 scale S^l（仅在Inlier范围）
  - [x] Hessian近似计算（对角Fisher信息矩阵）
    - 已实现: `inlierq_rknn/calibrate/fisher_scale_refine.py`
    - 近似方式: 使用 GVSS saliency^2 作为空间位置 diagonal-Fisher proxy
  - [x] 重建误差近似下的scale微调（grid search，非完整Hessian）
  - [x] 梯度下降迭代优化 S^l
    - 已实现: 对 log(scale) 使用 Adam 迭代优化 weighted reconstruction MSE
  - [x] Per-channel量化参数输出（activation per-channel symmetric初值）
  - [x] 统一量化配置导出
    - 脚本: `inlierq_rknn/quantize/export_quant_config.py`
    - 输出: `outputs/quant/inlierq_quant_config.json`
    - 策略: 优先使用 diagonal-Fisher refined scale；保留grid-search/u8/per-channel参考参数
- **状态:** completed

### Phase 3: ONNX Q/DQ导出与RKNN转换
- [x] 实现 fake_quantize.py
  - 脚本: `inlierq_rknn/quantize/fake_quantize.py`
  - 使用固定scale/zp fake quant wrapper插入InlierQ优化后的activation scale
  - 权重量化: Conv2d per-channel symmetric fake quant
  - 激活量化: per-tensor symmetric INT8（来自Phase 2统一配置）
  - 输出层特殊处理（box/score可能需per-channel或split）
  - 已完成轻量dry-run：13/13 activation层映射成功，126个Conv2d权重fake quant，dummy forward通过
- [x] 导出ONNX Q/DQ模型
  - opset ≥ 13（支持QuantizeLinear/DequantizeLinear）
  - 验证ONNX图中包含 Q/DQ 节点
  - 已生成: `outputs/onnx/spacer_640_inlierq_qdq.onnx`
  - 检查结果: QuantizeLinear=10, DequantizeLinear=10
  - 如有需要，复用split-output策略
- [x] 实现ONNX Q/DQ检查脚本
  - 脚本: `inlierq_rknn/quantize/check_onnx_qdq.py`
  - 输出节点计数、Q/DQ节点数量和summary JSON
- [x] 实现RKNN转换脚本
  - 脚本: `inlierq_rknn/rknn/convert_qdq_rknn.py`
  - 默认 `do_quantization=False`，避免对Q/DQ ONNX二次量化
- [x] 转换为RKNN（重计算，由用户运行）
  - rknn.load_onnx(Q/DQ ONNX)
  - rknn.build(do_quantization=False) ← 关键：模型已量化
  - 已生成: `outputs/rknn/spacer_640_inlierq_opt2.rknn` (optimization-level=2)
  - 已生成: `outputs/rknn/spacer_640_inlierq.rknn` (optimization-level=3)
- **状态:** completed

### Phase 4: Baseline复现与对比管线
- [x] 使用原有脚本重新生成baseline .rknn
  - PT → ONNX → split_output → rknn.build(do_quantization=True, calib=300)
  - 确保使用相同的spacer_640.pt作为起点
- [x] 建立统一评测脚本 eval_rk3588.py
  - 当前脚本: `eval/rknn-infer-test.py`
  - 已修改支持两类输出:
    - baseline raw/split输出: `[1,5,8400]` 或 `[1,4,8400] + [1,1,8400]`
    - InlierQ postprocessed输出: `[1,300,6]` (`xyxy, score, class`)
  - 默认 `imgsz=640`，与当前 `spacer_640.pt` 对齐
  - 加载.rknn → 推理77张test set → 计算mAP@0.5和mAP@0.5:0.95
  - 记录推理耗时（pre-process / inference / post-process）
  - 确保两种方法使用相同的测试集和后处理参数
- [x] 完成四模型对比评测
  - 结果文件: `eval/ModelComparisonResults.xlsx`
  - 对比模型: baseline / int8_split_baseline / inlierq_opt2 / inlierq
- **状态:** completed

### Phase 5: RK3588实测与对比分析
- [x] 在RK3588上运行baseline模型评测
- [x] 在RK3588上运行InlierQ模型评测
- [x] 对比指标: mAP@0.5, mAP@0.5:0.95, Precision, Recall, FPS
- [x] 逐层/分支问题分析
  - ONNX Q/DQ仅覆盖10个block-level层，Detect score层未被Ultralytics导出路径保留
  - Q/DQ模型最终输出为postprocessed `[1,300,6]`，与baseline split raw输出路径不同
- [x] 输出实验对比报告
  - `eval/ModelComparisonResults.xlsx`
- **状态:** completed

## 关键问题
1. ~~rknn-toolkit2是否支持ONNX Q/DQ？~~ → ✅ 确认支持 (QuantizeLinear/DequantizeLinear)
2. YOLO26 anchor-free检测头的conf通道是否足以提供有区分度的GVSS信号？（单类别 vs 论文的多类别）
3. ~~`rknn.build(do_quantization=False)` 是否能正确保留ONNX Q/DQ中的量化参数？~~ → ✅ 已完成Q/DQ ONNX到RKNN转换并实测。
4. InlierQ的64张校准图 vs baseline的300张 → 少样本优势能否体现？
5. YOLO26的SiLU/C2f/SPPF结构在ONNX Q/DQ导出时是否会引入不兼容算子？

## 决策记录
| 决策 | 原因 |
|------|------|
| 技术路径: PyTorch FakeQuantize → ONNX Q/DQ → RKNN | rknn-toolkit2 2.3.2已支持Q/DQ算子，且v2.0.0起官方改善QAT模型导入 |
| 优先实现W8A8 | 与用户现有baseline对齐（同为INT8），且RK3588 NPU对INT8支持最成熟 |
| 校准集使用spacer train split 64张 | 对齐论文设置（64样本），与baseline的300张形成对比 |
| 保留output split策略 | box坐标范围(0~640)与score(0~1)差异大，per-channel Q/DQ也可能需要；split是已验证的可靠方案 |
| 使用PyTorch原生量化API | torch.quantization 与 ONNX Q/DQ 导出生态最兼容 |
| Phase 3重计算由用户运行 | ONNX导出与RKNN build最吃CPU/内存；脚本已提供阶段日志和进度输出，运行指令见`plan/phase3_runbook.md` |
| 当前部署推荐 | InlierQ精度小幅领先但FPS下降明显；当前spacer单类别检测部署优先选择`int8_split_baseline` |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| Windows PowerShell profile 调用 `D:\Anaconda\Scripts\conda.exe` 报 Access is denied | 多次 | 使用 `login=false` 绕开 profile；后续建议在 WSL 执行量化相关命令 |
| `D:\Anaconda\python.exe` 直接调用无输出/失败 | 2 | 改用 `D:\Anaconda\envs\yolov8\python.exe` |
| `yolov8` 环境导入 torch 失败: `_ctypes` DLL load failed: 拒绝访问 | 3 | 当前 Windows 无法完成模型加载；切换到已配置 rknn-toolkit2 的 WSL 环境运行分析脚本 |
| `data.yaml`中`train: train/images`与实际目录`images/train`不一致 | 1 | `prepare_calib.py`已增加fallback，路径不存在时自动使用`images/<split>` |
| GVSS smoke首次反传失败：scores无grad_fn | 1 | 输入tensor设置`requires_grad_(True)`，模型参数仍冻结；重跑成功 |
| EM-GMM在Detect稀疏score层上受大量零梯度影响，初版inlier比例异常 | 1 | 增加稀疏层规则：非零比例<10%时只在非零S^l上拟合GMM，零梯度位置不作为inlier |
| Detect score层Inlier activation全为负logits，u8 asymmetric zero-point饱和到255 | 1 | 已记录；后续Q/DQ导出时优先考虑score logits使用symmetric int8或在sigmoid/输出拆分后单独处理 |
| 完整Hessian矩阵优化未实现（当前为diagonal-Fisher proxy） | 1 | 已实现GVSS saliency^2对角Fisher近似 + Adam更新scale；完整二阶Hessian矩阵仍作为论文增强复现项 |
| `yolo26`缺少ONNX导出检查依赖 | 1 | 当前检测到`onnx=False, onnxruntime=False, tqdm=False`；运行Phase 3重导出前建议先安装`onnx onnxruntime tqdm` |
| RKNN load_onnx/build失败: `onnx.mapping`旧接口缺失 | 2 | `rknn`环境ONNX 1.21与rknn-toolkit2 2.3.2兼容问题；已在`convert_qdq_rknn.py`中补齐`TENSOR_TYPE_TO_NP_TYPE`和`NP_TYPE_TO_TENSOR_TYPE` shim，备选方案为降级`onnx==1.14.1` |

## 备注
- Phase 1先加载模型确认层结构，再细化Phase 2的具体Hook策略
- 需要在PC端先搭建可用的rknn-toolkit2 2.3.2环境
- RK3588通过SSH评测，需要确认板端rknn-lite2 runtime版本与toolkit版本匹配
