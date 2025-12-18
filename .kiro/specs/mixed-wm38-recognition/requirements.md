# Requirements Document

## Introduction

本项目旨在构建一个面向晶圆工艺场景的混合缺陷晶圆图谱（Wafer Map）多任务识别与可解释诊断系统。系统需要支持三个核心任务：38类缺陷分类（T1）、缺陷定位分割（T2）、以及弱监督成分分离（T3）。项目采用渐进式实验设计（E0→E3），逐步引入自监督预训练、长尾增强和弱监督分离技术，最终交付可复现的实验代码、结果报告和教学文档。

## Glossary

- **Wafer_Map_System**: 晶圆图谱多任务识别系统，负责分类、分割和成分分离
- **MixedWM38**: 混合缺陷晶圆图谱数据集，包含38类（1 normal + 8 single + 29 mixed）
- **WM-811K**: 大规模无标签晶圆图谱数据集，用于自监督预训练
- **SSL**: Self-Supervised Learning，自监督学习
- **DDPM**: Denoising Diffusion Probabilistic Model，去噪扩散概率模型
- **Macro-F1**: 宏平均F1分数，各类别F1的算术平均
- **Dice**: Dice系数，分割任务的重叠度指标
- **IoU**: Intersection over Union，交并比
- **AMP**: Automatic Mixed Precision，自动混合精度训练
- **Checkpoint**: 模型训练检查点，用于断点续训和最佳模型保存
- **YAML**: 配置文件格式，用于实验参数管理

## Requirements

### Requirement 1: 环境搭建与项目初始化

**User Story:** As a 深度学习新手, I want 清晰的环境搭建指南和项目结构, so that 我能在Windows系统上快速配置并理解项目组织方式。

#### Acceptance Criteria

1. WHEN a user clones the repository THEN the Wafer_Map_System SHALL provide a docs/SETUP_WINDOWS.md file containing conda/venv environment setup instructions for Windows 11 with mirror source configuration and troubleshooting guide
2. WHEN a user follows the setup guide THEN the Wafer_Map_System SHALL enable successful installation of all dependencies with clear error resolution steps for common issues
3. WHEN the environment is configured THEN the Wafer_Map_System SHALL support CUDA-enabled PyTorch with RTX 4070 SUPER GPU
4. WHEN a user inspects the project THEN the Wafer_Map_System SHALL present a standardized directory structure with configs/, src/, data/, results/, docs/, report/, slides/ folders

### Requirement 2: 数据准备与加载

**User Story:** As a 研究者, I want 标准化的数据准备流程, so that 我能正确加载MixedWM38和WM-811K数据集进行训练和评估。

#### Acceptance Criteria

1. WHEN a user runs the data preparation script THEN the Wafer_Map_System SHALL generate a standardized data directory structure without including raw data files in git
2. WHEN loading MixedWM38 dataset THEN the Wafer_Map_System SHALL support both 38-class single-label and 8-class multi-label classification formats
3. WHEN 8-class multi-label annotations are unavailable THEN the Wafer_Map_System SHALL construct multi-label targets from 38-class labels using predefined mapping rules and log the construction statistics
4. WHEN loading data for segmentation THEN the Wafer_Map_System SHALL provide binary defect masks paired with wafer map images
5. WHEN segmentation masks are unavailable THEN the Wafer_Map_System SHALL construct pseudo-masks using thresholding or morphological operations, record generation parameters in config_snapshot.yaml, and export sample overlay images to results/<exp_name>/pseudo_mask_samples/ for verification
6. WHEN applying data augmentation THEN the Wafer_Map_System SHALL use wafer-friendly transforms including rotation, flip, and mild morphological noise while avoiding large crops that destroy center/edge semantics
7. WHEN storing dataset configuration THEN the Wafer_Map_System SHALL use configs/*.yaml format for all experiment parameters

### Requirement 3: 基线模型训练（E0）

**User Story:** As a 实验者, I want 一个多任务基线模型, so that 我能建立分类和分割任务的性能基准。

#### Acceptance Criteria

1. WHEN a user runs train.py with E0 config THEN the Wafer_Map_System SHALL train a multi-task model supporting both classification and segmentation heads
2. WHEN training completes THEN the Wafer_Map_System SHALL save the best checkpoint based on a configurable validation metric (default: Macro-F1) specified in the config file
3. WHEN training with AMP enabled THEN the Wafer_Map_System SHALL utilize mixed precision to reduce GPU memory usage
4. WHEN training is interrupted THEN the Wafer_Map_System SHALL support resuming from the latest checkpoint
5. WHEN a user runs eval.py THEN the Wafer_Map_System SHALL output metrics.csv containing Macro-F1, Dice, and IoU scores with per-class breakdown; IF multi-label classification is enabled THEN the Wafer_Map_System SHALL additionally include mAP in metrics.csv
6. WHEN evaluation completes THEN the Wafer_Map_System SHALL generate confusion_matrix.png and seg_overlays/ visualization files

### Requirement 4: 自监督预训练（E1）

**User Story:** As a 研究者, I want 使用WM-811K进行自监督预训练, so that 我能获得更好的特征表示来提升下游任务性能。

#### Acceptance Criteria

1. WHEN a user runs SSL pretraining THEN the Wafer_Map_System SHALL train an encoder using contrastive learning (SimCLR or MoCo style) on WM-811K dataset
2. WHEN loading pretrained weights THEN the Wafer_Map_System SHALL initialize the E1 model encoder from SSL checkpoint with key mapping support
3. WHEN loading pretrained weights THEN the Wafer_Map_System SHALL log matched keys count, missing keys count, and unexpected keys count for weight loading diagnostics
4. WHEN comparing E1 with E0 THEN the Wafer_Map_System SHALL output comparable metrics and delta table showing performance differences
5. WHEN SSL pretraining applies augmentation THEN the Wafer_Map_System SHALL use wafer-friendly transforms preserving center/edge semantics

### Requirement 5: 长尾增强（E2）

**User Story:** As a 研究者, I want 处理类别不平衡问题, so that 我能提升尾部类别的识别性能。

#### Acceptance Criteria

1. WHEN training E2 model THEN the Wafer_Map_System SHALL apply tail-class augmentation strategies to improve minority class performance
2. WHEN DDPM generation is too complex THEN the Wafer_Map_System SHALL fallback to class-balanced sampling with strong augmentation and focal/class-balanced loss
3. WHEN comparing E2 with E0 THEN the Wafer_Map_System SHALL output comparable metrics and delta table showing performance differences especially on tail classes
4. WHEN using fallback strategy THEN the Wafer_Map_System SHALL document the trade-off decision in the experiment report

### Requirement 6: 弱监督成分分离（E3）

**User Story:** As a 研究者, I want 对混合缺陷进行成分分离, so that 我能理解混合缺陷由哪些基础缺陷组成。

#### Acceptance Criteria

1. WHEN training E3 model THEN the Wafer_Map_System SHALL add an 8-channel separation head for basic defect component heatmaps
2. WHEN separation head is added THEN the Wafer_Map_System SHALL output delta table comparing with E1 baseline and document any performance trade-offs in the experiment report without enforcing hard thresholds
3. WHEN evaluating E3 THEN the Wafer_Map_System SHALL generate separation_maps/ containing 8-channel heatmap visualizations
4. WHEN full separation is too complex THEN the Wafer_Map_System SHALL fallback to prototype similarity heatmaps as a degraded weak-supervised solution

### Requirement 7: 实验管理与可复现性

**User Story:** As a 研究者, I want 统一的实验管理接口, so that 我能方便地运行、比较和复现所有实验。

#### Acceptance Criteria

1. WHEN running any experiment THEN the Wafer_Map_System SHALL use unified entry points: train.py --config configs/*.yaml and eval.py --config configs/*.yaml --ckpt path/to/best.pt
2. WHEN an experiment completes THEN the Wafer_Map_System SHALL output results to results/<exp_name>/ with the following structure:
   - metrics.csv (all metrics with per-class breakdown)
   - confusion_matrix.png
   - seg_overlays/ (segmentation visualization images)
   - separation_maps/ (E3 only, 8-channel heatmaps)
   - curves/ (loss_curve.png, metric_curve.png)
   - config_snapshot.yaml (copy of config used)
3. WHEN training starts THEN the Wafer_Map_System SHALL fix random seed and record git commit hash to results/<exp_name>/meta.json for reproducibility (write "unknown" if git unavailable)
4. WHEN saving results THEN the Wafer_Map_System SHALL include a copy of the config file used for that experiment as config_snapshot.yaml
5. WHEN running debug mode THEN the Wafer_Map_System SHALL complete a full train-eval cycle within 5 minutes using minimal samples per class

### Requirement 8: 可视化与报告

**User Story:** As a 报告撰写者, I want 自动生成的可视化结果, so that 我能直接用于实验报告和PPT。

#### Acceptance Criteria

1. WHEN evaluation completes THEN the Wafer_Map_System SHALL generate confusion_matrix.png in results/<exp_name>/ for classification results
2. WHEN segmentation evaluation completes THEN the Wafer_Map_System SHALL generate mask overlay images in results/<exp_name>/seg_overlays/ showing predicted vs ground truth
3. WHEN E3 evaluation completes THEN the Wafer_Map_System SHALL generate 8-channel separation heatmaps in results/<exp_name>/separation_maps/ with original image comparison
4. WHEN training completes THEN the Wafer_Map_System SHALL generate loss_curve.png and metric_curve.png in results/<exp_name>/curves/ directory
5. WHEN all experiments complete THEN the Wafer_Map_System SHALL generate results/comparison.csv with delta values between E0/E1/E2/E3

### Requirement 9: 文档与教学

**User Story:** As a 深度学习新手, I want 详细的教学文档, so that 我能理解每个步骤的原理并独立排查问题。

#### Acceptance Criteria

1. WHEN documentation is complete THEN the Wafer_Map_System SHALL provide docs/LEARNER_GUIDE.md with step-by-step commands and expected outputs
2. WHEN documentation is complete THEN the Wafer_Map_System SHALL include common error troubleshooting for CUDA, dependencies, paths, and GPU memory issues
3. WHEN documentation is complete THEN the Wafer_Map_System SHALL provide concept explanations for Macro-F1, Dice, contrastive learning, long-tail, and weak supervision
4. WHEN documentation is complete THEN the Wafer_Map_System SHALL explain how to read training logs and detect overfitting

### Requirement 10: 报告与PPT生成

**User Story:** As a 学生, I want 自动生成的实验报告和PPT大纲, so that 我能快速完成课程作业提交。

#### Acceptance Criteria

1. WHEN all experiments complete THEN the Wafer_Map_System SHALL generate report/REPORT.md containing abstract, background, methods, results, ablation, and conclusions
2. WHEN report is generated THEN the Wafer_Map_System SHALL include reproducible command lists for each experiment
3. WHEN PPT outline is needed THEN the Wafer_Map_System SHALL generate slides/SLIDES.md with 10-12 page structure
4. WHEN PPT generation is requested THEN the Wafer_Map_System SHALL provide scripts/build_pptx.py using python-pptx library
