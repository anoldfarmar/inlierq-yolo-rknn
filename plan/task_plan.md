# Task Plan: 基于InlierQ的YOLO26 RKNN量化部署 — Spacer检测

## 目标
将InlierQ算法应用于已训练的YOLO26 spacer检测模型，实现 PyTorch(.pt) → InlierQ校准 → ONNX(Q/DQ) → RKNN(.rknn) → RK3588部署，并与原有rknn-toolkit2默认量化方法进行mAP和推理速度对比评测。

## 当前阶段
Phase 2

## 下一步
继续 Phase 2 收尾：将 `outputs/quant/inlier_activation_qparams.json` 与 `outputs/quant/inlier_activation_scale_refined.json` 整合为后续 fake quant/Q-DQ 可消费的统一量化配置；完整 Hessian/Fisher 版本作为增强项保留。

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
- [ ] 实现Inlier-aware量化参数优化
  - [x] MinMax初始化 scale S^l（仅在Inlier范围）
  - [ ] Hessian近似计算（对角Fisher信息矩阵）
  - [x] 重建误差近似下的scale微调（grid search，非完整Hessian）
  - [ ] 梯度下降迭代优化 S^l（增强项）
  - [x] Per-channel量化参数输出（activation per-channel symmetric初值）
- **状态:** in_progress

### Phase 3: ONNX Q/DQ导出与RKNN转换
- [ ] 实现 fake_quantize.py
  - 使用 torch.quantization.FakeQuantize 插入InlierQ优化后的scale/zp
  - 权重量化: per-channel symmetric INT8
  - 激活量化: per-tensor asymmetric INT8（或per-channel视情况）
  - 输出层特殊处理（box/score可能需per-channel或split）
- [ ] 导出ONNX Q/DQ模型
  - opset ≥ 13（支持QuantizeLinear/DequantizeLinear）
  - 验证ONNX图中包含 Q/DQ 节点
  - 如有需要，复用split-output策略
- [ ] 转换为RKNN
  - rknn.load_onnx(Q/DQ ONNX)
  - rknn.build(do_quantization=False) ← 关键：模型已量化
  - 若do_quantization=False不可用，测试do_quantization=True是否会二次量化
  - rknn.export_rknn() → spacer_640_inlierq.rknn
- **状态:** pending

### Phase 4: Baseline复现与对比管线
- [ ] 使用原有脚本重新生成baseline .rknn
  - PT → ONNX → split_output → rknn.build(do_quantization=True, calib=300)
  - 确保使用相同的spacer_640.pt作为起点
- [ ] 建立统一评测脚本 eval_rk3588.py
  - 加载.rknn → 推理77张test set → 计算mAP@0.5和mAP@0.5:0.95
  - 记录推理耗时（pre-process / inference / post-process）
  - 确保两种方法使用相同的测试集和后处理参数
- **状态:** pending

### Phase 5: RK3588实测与对比分析
- [ ] 在RK3588上运行baseline模型评测
- [ ] 在RK3588上运行InlierQ模型评测
- [ ] 对比指标: mAP@0.5, mAP@0.5:0.95, Precision, Recall, FPS
- [ ] 逐层量化误差对比分析（如果工具支持）
- [ ] 输出实验对比报告
- **状态:** pending

## 关键问题
1. ~~rknn-toolkit2是否支持ONNX Q/DQ？~~ → ✅ 确认支持 (QuantizeLinear/DequantizeLinear)
2. YOLO26 anchor-free检测头的conf通道是否足以提供有区分度的GVSS信号？（单类别 vs 论文的多类别）
3. `rknn.build(do_quantization=False)` 是否能正确保留ONNX Q/DQ中的量化参数？需要实测验证。
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
| 完整Hessian/Fisher优化尚未实现 | 1 | 已先实现Inlier重建误差grid-search微调，作为可验证近似；完整Fisher/梯度下降保留为增强项 |

## 备注
- Phase 1先加载模型确认层结构，再细化Phase 2的具体Hook策略
- 需要在PC端先搭建可用的rknn-toolkit2 2.3.2环境
- RK3588通过SSH评测，需要确认板端rknn-lite2 runtime版本与toolkit版本匹配
