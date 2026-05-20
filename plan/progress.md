# Progress: InlierQ YOLO26 RKNN量化实现

## 会话 2 — 2026-05-20

### 本次执行更新
- 已阅读 `plan/task_plan.md`、`plan/progress.md`、`plan/findings.md` 和 `paper/note_26_05.md`
- 已检查工程文件，当前仓库包含 baseline 脚本:
  - `org-rknn-quantization/convert_to_rknn_int8.py`
  - `org-rknn-quantization/split_yolo_output_onnx.py`
  - `org-rknn-quantization/pt2rknn.py`
- 已新增 Phase 1 工具:
  - `inlierq_rknn/analyze_model.py`: 加载 `spacer_640.pt`，导出层结构、候选 Hook 层和 forward 输出摘要
  - `inlierq_rknn/data/prepare_calib.py`: 从 train split 抽取 64 张校准图
- 已生成 64 张校准图列表:
  - `outputs/calib/spacer_train64.txt`
- Windows 环境检查结果:
  - `D:\Anaconda\envs\yolov8\python.exe` 可启动
  - `yolov8` 环境安装了 torch/ultralytics
  - 但导入 torch 失败: `_ctypes` DLL load failed: 拒绝访问
  - 当前 Codex 会话中 `wsl.exe` 未发现可用发行版，因此模型加载分析需要用户切换到 WSL 执行
- 用户重新安装`yolo26`环境后已可用：Python 3.12.13, torch 2.12.0+cu130, CUDA可用, ultralytics 8.4.51
- 已运行`inlierq_rknn/analyze_model.py`，生成：
  - `outputs/model_analysis/spacer_640_model_analysis.json`
  - `outputs/model_analysis/spacer_640_model_analysis.md`
- 已整理Phase 2 Hook计划：
  - `outputs/model_analysis/spacer_640_hook_plan.md`
- 已根据新数据路径重新生成64张校准图列表：
  - `outputs/calib/spacer_train64.txt`
- 已更新`inlierq_rknn/data/prepare_calib.py`，支持`--data`/`--split`读取`data.yaml`，并兼容当前`data.yaml`中`train/images`与实际`images/train`不一致的问题
- 已新增并验证Phase 2 GVSS smoke脚本：
  - `inlierq_rknn/calibrate/gvss.py`
  - 输出: `outputs/gvss/gvss_smoke.json`
  - 1张校准图验证通过，raw scores shape为`[1, 1, 8400]`，Top-K=50，13个Hook层产生显著性统计
  - Detect score分支`model.23.cv3.*`有梯度，box分支`model.23.cv2.*`无梯度，符合当前score-only辅助损失预期
- 已完成64张校准图GVSS聚合：
  - `outputs/gvss/gvss_train64.json`
  - `outputs/gvss/gvss_train64_summary.json`
  - `outputs/gvss/gvss_train64_saliency.npz`
- 已新增并运行EM-GMM聚类：
  - 脚本: `inlierq_rknn/calibrate/em_gmm.py`
  - summary: `outputs/gvss/em_gmm_train64_summary.json`
  - masks: `outputs/gvss/em_gmm_train64_masks.npz`
  - posteriors: `outputs/gvss/em_gmm_train64_posteriors.npz`
  - 13个有效层完成Inlier/Anomaly划分；稀疏层使用“只拟合非零S^l”的规则避免零梯度主导聚类
- 已新增并运行Inlier-aware MinMax量化参数初始化：
  - 脚本: `inlierq_rknn/calibrate/inlier_optimizer.py`
  - 输出: `outputs/quant/inlier_activation_qparams.json`
  - 覆盖13个有效Hook层，输出activation u8 asymmetric、i8 symmetric、i8 per-channel symmetric初值
  - Detect score层Inlier activation为负logits，u8 asymmetric zero-point饱和到255，后续需要在Q/DQ策略中单独处理
- 已新增并运行Inlier scale重建误差微调：
  - 脚本: `inlierq_rknn/calibrate/scale_refine.py`
  - 输出: `outputs/quant/inlier_activation_scale_refined.json`
  - 目标: uniform_inlier_reconstruction_mse_symmetric_i8
  - 常规层MSE改善约20%-73%，Detect score层因Inlier样本稀疏仅小幅改善约0.6%-3.5%

### 本次新增信息
- 用户原有量化方法确认: rknn-toolkit2内置校准（MinMax/KL），300张校准图
- 关键bug: 输出split（box/score分开量化）是baseline能工作的必要条件
- 数据集: spacer-merged_processed, train/val/test = 569/80/77
- rknn-toolkit2: 2.3.2, RK3588: SSH连接
- 模型: spacer_640.pt (imgsz=640), 目标精度: W8A8
- **关键发现**: rknn-toolkit2 2.3.x **支持 QuantizeLinear/DequantizeLinear** → ONNX Q/DQ路径可行

### 会话1 — 2026-05-20

### 初始状态
- 已阅读并理解 InlierQ 论文
- 已阅读并分析 note_26_05.md 中的论文笔记
- 已有训练好的 YOLO26 spacer 检测 .pt 模型

### 总体进度
- [x] 创建 task_plan.md、findings.md、progress.md 规划文件
- [x] 梳理InlierQ算法三大核心组件（GVSS/EM-GMM/Inlier-aware优化）
- [x] 记录关键公式与伪代码
- [x] 分析用户原有baseline量化方法（convert_to_rknn_int8.py + split_yolo_output_onnx.py）
- [x] 调研rknn-toolkit2对ONNX Q/DQ的支持情况 → ✅ 支持
- [x] 确定技术路径: PyTorch InlierQ → ONNX Q/DQ → RKNN
- [x] 准备64张InlierQ校准图列表
- [x] 在可用PyTorch环境中生成YOLO26层结构报告
- [x] 完成Phase 1: 环境搭建与模型分析
- [x] 开始Phase 2: InlierQ核心算法PyTorch实现
- [x] GVSS遍历64张校准图并导出聚合显著性分布
- [x] EM-GMM聚类生成Inlier masks
- [ ] Inlier-aware量化参数初始化与优化
  - [x] Inlier MinMax scale/zero-point初始化
  - [x] 重建误差近似下的scale微调
  - [ ] 统一量化配置导出
