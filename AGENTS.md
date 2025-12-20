# Repository Guidelines
面向对象：AI Agent（编程+实验+写报告+写PPT）与“纯小白”用户（需要边做边教会我）
## 我是谁 & 我需要什么
- 我是研究生，方向：MEMS 传感器件设计及工艺开发
- 我是编程/深度学习新手：你必须“教我”，不仅要给代码，也要解释在做什么、为什么这么做、怎么验证对不对
- 目标：在24号前完成《模式识别》课程实验，可复现、有对比、有消融、有图、有结论

## 硬件与系统约束
- OS: Windows 11 (24H2)
- CPU: Intel Core Ultra 9 285H
- RAM: 32GB
- GPU: RTX 4070 SUPER 12GB
要求：
- 训练脚本必须支持 AMP（混合精度）与断点续训
- 不要把数据集提交到 git；只提交脚本与说明

## 实验题目
晶圆工艺场景下的混合缺陷晶圆图谱多任务识别与可解释诊断：自监督表征学习 + 扩散式长尾增强 + 弱监督成分分离

## 数据集与任务定义
### 1 主数据：MixedWM38（用于训练/测试）
任务：
- T1 分类（两条线都要支持，至少先跑通 38 类）：
  - 38 类单标签分类（1 normal + 8 single + 29 mixed）
  - 8 类基础缺陷多标签（如果原仓库支持/可构造）
- T2 缺陷定位分割：
  - 输出二值 mask（缺陷/非缺陷）
- T3 成分分离：
  - 对“混合缺陷”输出 8 通道基础缺陷热力图（弱监督，无像素级真值）

### 2 预训练数据：WM-811K
- 目的：用大量无标签 wafer maps 训练 encoder 表征（SimCLR 或 MoCo 风格）
- 注意：增强必须“wafer 友好”：旋转/翻转/轻微形态学噪声；避免大裁剪破坏中心/边缘语义

## 设计的实验组
- E0 基线：仅多任务（分类+分割），不使用 SSL/扩散/分离
- E1：E0 + SSL 预训练初始化 encoder（WM-811K）
- E2：E0 + 尾部类别加入 DDPM 生成样本（用于提升 macro-F1）
- E3：E1 + 8 通道弱监督成分分离头（要求不显著降低识别性能）

交付要求：
- 每个实验组都要：配置文件、训练命令、评估命令、结果汇总 CSV、关键可视化（混淆矩阵、mask overlay、分离热力图示例）

## 指标与可视化
分类/多标签：
- Macro-F1（主指标）
- mAP（如果做 8 类多标签）
- confusion matrix
分割/分离：
- Dice、IoU
可解释性可视化：
- 分割：mask overlay
- 分离：8 通道热力图 + 与原图对照

## 写作交付
### 1 实验报告
结构必须包含：
- 摘要（我做了什么，结果如何）
- 背景与动机（工艺良率/混合缺陷挑战）
- 数据集与任务定义
- 方法（E0/E1/E2/E3）
- 实验设置（硬件、超参、训练细节）
- 结果与分析（表格 + 图）
- 消融实验（E0→E3）
- 失败与取舍（哪些没做/为何，如何替代）
- 结论与展望

你写报告时必须：
- 每一节都“先解释概念，再说明我代码怎么实现，再说明怎么验证”
- 给出“我该怎么复现实验”的命令清单

### 2 PPT
- 目标：10–12 页
- 必须包含：问题定义、数据集、方法框图、E0/E1 对比表、关键可视化（混淆矩阵/overlay/热力图）、结论
- 若你能生成 pptx：请用 python-pptx 写 scripts/build_pptx.py，并在 docs/BUILD_PPT.md 写清楚怎么生成

### 3 教学要求
你需要额外生成一份：
- docs/LEARNER_GUIDE.md
内容包括：
- 我每一步要运行什么命令
- 常见报错（CUDA/依赖/路径/显存）怎么排查
- 关键概念小抄：macro-F1、Dice、对比学习、长尾、弱监督
- 如何读训练日志/判断是否过拟合

## 项目结构与任务范围
- 任务：T1 38类分类（可映射为8类多标签）、T2 二值分割、T3 8通道弱监督分离。
- 数据：MixedWM38（主训练/评估）、WM-811K（SSL预训练）。
- 代码：`src/`（data/models/training/evaluation/visualization）、`configs/`（实验YAML）、`scripts/`（数据/报告/验证工具）、`tests/`、`docs/`。
- 产出：`results/`、`weights/`、`inference/`、`logs/`；数据仅放 `data/`（遵循 `.gitignore`）。

## 统一环境与命令（必须使用 conda run -n wafer-seg-class）
- 安装与检查：`conda run -n wafer-seg-class pip install -r requirements.txt`；`conda run -n wafer-seg-class python check_env.py`
- 数据准备：`conda run -n wafer-seg-class python scripts/prepare_mixedwm38.py --input data/raw/Wafer_Map_Datasets.npz --output data/processed`
- 训练入口：`conda run -n wafer-seg-class python train.py --config configs/e0.yaml [--debug|--resume ...]`
- SSL预训练：`conda run -n wafer-seg-class python train_ssl.py --config configs/ssl.yaml`
- 评估入口：`conda run -n wafer-seg-class python eval.py --config configs/e0.yaml --ckpt results/e0/checkpoints/best.pt`

## 课程实验全自动流程（可复现/对比/消融/有图/有结论/有教学）
- 实验组：E0 基线（分类+分割），E1（E0+SSL），E2（长尾增强），E3（E1+分离/原型降级）。
- 顺序执行 `configs/e0.yaml → e1.yaml → e2.yaml → e3.yaml` 的训练与评估，即可全自动完成《模式识别》课程实验。
- 对比与结论生成：
  - `conda run -n wafer-seg-class python scripts/generate_comparison.py --results_root results --out results/comparison.csv`
  - `conda run -n wafer-seg-class python scripts/generate_report.py --results_root results --out report/REPORT.md`
  - `conda run -n wafer-seg-class python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md`
  - `conda run -n wafer-seg-class python scripts/build_pptx.py --slides_md slides/SLIDES.md --results_root results --out slides/final.pptx`

## 产物规范（报告与教学依赖）
- `results/<exp>/` 必含：`metrics.csv`、`confusion_matrix.png`、`seg_overlays/`、`curves/`、`config_snapshot.yaml`、`meta.json`；E3 额外 `separation_maps/`。
- 教学与复现：`docs/LEARNER_GUIDE.md`、`docs/SETUP_WINDOWS.md`；报告与PPT：`report/REPORT.md`、`slides/SLIDES.md`。

## 编码与测试
- Python 4空格、PEP 8；命名 `snake_case`/`PascalCase`/`UPPER_CASE`；注释语言与文件一致。
- pytest：`conda run -n wafer-seg-class pytest -m "not slow"`；测试文件 `tests/test_*.py`。

## 提交与PR
- 提交信息沿用 `feat:`/简短说明风格；PR需说明配置变更、指标对比与关键图。
- 严禁提交原始数据、权重和大型结果文件。
