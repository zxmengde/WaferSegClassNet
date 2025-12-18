# Implementation Plan

## Phase 1: 环境与项目骨架

- [x] 1. 项目结构初始化与环境配置






  - [x] 1.1 检查项目环境是否配置良好，是否可运行

    - .gitignore等项目开发相关文件的编辑等
    - _Requirements: 1.1_

## Phase 2: 配置系统与数据模块

- [ ] 2. 配置系统实现
  - [ ] 2.1 创建 src/config_schema.py 定义配置数据类
    - ExperimentConfig, DataConfig, ModelConfig, TrainingConfig, LossConfig
    - YAML加载与验证函数
    - _Requirements: 2.7_
  - [ ] 2.2 编写属性测试：配置YAML格式有效性
    - **Property 9: 配置文件YAML格式有效性**
    - **Validates: Requirements 2.7**
  - [ ] 2.3 创建 configs/e0_template.yaml 基线配置模板
    - 包含所有必需字段和默认值
    - 作为其他配置的基础模板
    - _Requirements: 7.1_

- [ ] 3. 数据模块实现
  - [ ] 3.1 创建 src/data/mappings.py 标签映射
    - LABEL_38_TO_8 字典（权威来源）
    - map_38_to_8() 和 validate_mapping() 函数
    - 打印覆盖率统计
    - _Requirements: 2.2, 2.3_
  - [ ] 3.2 编写属性测试：38→8标签映射正确性
    - **Property 1: 38类到8类标签映射正确性**
    - **Validates: Requirements 2.2, 2.3**
  - [ ] 3.3 创建 src/data/dataset.py MixedWM38Dataset类
    - 支持single_label和multi_label模式
    - 支持伪mask生成（触发时记录参数到config_snapshot）
    - 注：伪mask overlay导出由visualization/overlays.py统一负责
    - _Requirements: 2.2, 2.4, 2.5_
  - [ ] 3.4 编写属性测试：分割mask二值性
    - **Property 2: 分割mask二值性**
    - **Validates: Requirements 2.4**
  - [ ] 3.5 创建 src/data/augmentation.py 数据增强
    - WaferFriendlyAugmentation类
    - 白名单/黑名单验证
    - _Requirements: 2.6_
  - [ ] 3.6 编写轻量pytest：数据增强白名单检查
    - 验证增强配置不包含黑名单操作
    - _Requirements: 2.6_
  - [ ] 3.7 创建 src/data/dataloader.py 数据加载器
    - get_dataloaders() 函数
    - 支持debug模式（每类最多5样本）
    - _Requirements: 7.5_

- [ ] 4. Checkpoint - 数据模块验证
  - 运行数据准备和加载测试
  - 若遇问题：记录假设 + 给出备选方案 + 采用保守实现

## Phase 3: 模型模块

- [ ] 5. 模型架构实现
  - [ ] 5.1 创建 src/models/encoder.py 编码器
    - WaferEncoder类（基于现有convBlock改造）
    - load_pretrained() 方法（支持key mapping）
    - 输出weight_loading.json统计
    - _Requirements: 4.2, 4.3_
  - [ ] 5.2 编写轻量pytest：权重加载统计字段完整性
    - 验证输出包含matched/missing/unexpected字段
    - _Requirements: 4.2, 4.3_
  - [ ] 5.3 创建 src/models/decoder.py 分割解码器
    - 基于现有transposeConvBlock改造
    - _Requirements: 3.1_
  - [ ] 5.4 创建 src/models/heads.py 任务头
    - ClassificationHead（38类/8类）
    - SeparationHead（8通道，E3用）
    - _Requirements: 6.1_
  - [ ] 5.5 编写轻量pytest：分离头输出形状
    - 验证输出为8通道
    - _Requirements: 6.1_
  - [ ] 5.6 创建 src/models/multitask.py 多任务模型
    - WaferMultiTaskModel类
    - forward() 返回 {cls_logits, seg_mask, sep_heatmaps}
    - _Requirements: 3.1_

- [ ] 6. 损失函数实现
  - [ ] 6.1 创建 src/models/losses.py
    - DiceLoss, BCEDiceLoss（基于现有loss.py改造）
    - FocalLoss, ClassBalancedLoss
    - MultiTaskLoss（组合损失）
    - _Requirements: 3.1, 5.1, 5.2_

- [ ] 7. Checkpoint - 模型模块验证
  - 运行模型前向传播测试
  - 若遇问题：记录假设 + 给出备选方案 + 采用保守实现

## Phase 4: 训练与评估模块

- [ ] 8. 训练模块实现
  - [ ] 8.1 创建 src/training/trainer.py Trainer类
    - AMP混合精度支持
    - 断点续训支持
    - 可配置的best checkpoint指标
    - 保存meta.json（git_commit, seed）
    - _Requirements: 3.2, 3.3, 3.4, 7.3_
  - [ ] 8.2 编写轻量pytest：检查点保存/加载
    - 验证模型权重和优化器状态恢复
    - _Requirements: 3.4_
  - [ ] 8.3 创建 src/training/utils.py 训练工具
    - set_seed(), get_git_commit()
    - save_config_snapshot(), save_meta_json()
    - _Requirements: 7.3, 7.4_

- [ ] 9. 评估模块实现
  - [ ] 9.1 创建 src/evaluation/metrics.py 指标计算
    - compute_macro_f1(), compute_map()
    - compute_dice(), compute_iou()
    - _Requirements: 3.5_
  - [ ] 9.2 编写轻量pytest：指标计算正确性
    - 使用已知输入验证输出
    - _Requirements: 3.5_
  - [ ] 9.3 创建 src/evaluation/evaluator.py Evaluator类
    - evaluate() 方法
    - 输出metrics.csv
    - 伪mask使用时调用overlays.py导出pseudo_mask_samples/（至少10张）
    - _Requirements: 3.5, 3.6, 2.5_

- [ ] 10. 可视化模块实现
  - [ ] 10.1 创建 src/visualization/plots.py
    - plot_confusion_matrix()
    - plot_loss_curves(), plot_metric_curves()
    - _Requirements: 8.1, 8.4_
  - [ ] 10.2 创建 src/visualization/overlays.py（伪mask overlay唯一责任模块）
    - generate_seg_overlays()
    - generate_separation_heatmaps()
    - generate_pseudo_mask_overlays()（伪mask样例导出，至少10张）
    - 统一负责所有overlay导出逻辑
    - _Requirements: 8.2, 8.3, 2.5_

- [ ] 11. Checkpoint - 训练评估模块验证
  - 运行训练和评估测试
  - 若遇问题：记录假设 + 给出备选方案 + 采用保守实现

## Phase 5: 统一入口与Debug模式

- [ ] 12. 统一入口实现
  - [ ] 12.1 创建 train.py 训练入口
    - 支持 --config, --debug, --resume 参数
    - 输出到 results/<exp_name>/
    - debug模式下训练结束自动调用评估，生成完整产物
    - _Requirements: 7.1, 7.2_
  - [ ] 12.2 编写属性测试：实验输出结构完整性
    - **Property 8: 实验输出结构完整性**
    - **Validates: Requirements 7.2, 7.3, 7.4**
  - [ ] 12.3 创建 eval.py 评估入口
    - 支持 --config, --ckpt 参数
    - _Requirements: 7.1_
  - [ ] 12.4 实现debug模式（一条命令训练+评估闭环）
    - `python train.py --config configs/e0.yaml --debug`
    - 每类最多5样本，epochs=2
    - 训练结束自动评估
    - 5分钟内完成，产出 results/e0_debug/ 下：
      - metrics.csv
      - confusion_matrix.png
      - curves/loss_curve.png, metric_curve.png
      - config_snapshot.yaml
      - meta.json
    - _Requirements: 7.5_

- [ ] 13. Checkpoint - Debug模式验证
  - 运行 `python train.py --config configs/e0.yaml --debug` 验证5分钟内完成
  - 验证所有产物生成
  - 若遇问题：记录假设 + 给出备选方案 + 采用保守实现

## Phase 6: E0基线实验

- [ ] 14. E0基线训练与评估
  - [ ] 14.1 创建 configs/e0.yaml 完整配置
    - 基于e0_template.yaml
    - 38类分类 + 分割
    - 默认超参数
    - _Requirements: 3.1_
  - [ ] 14.2 运行E0完整训练
    - `python train.py --config configs/e0.yaml`
    - 验证输出结构完整
    - _Requirements: 3.1, 3.2, 3.5, 3.6_
  - [ ] 14.3 运行E0评估并生成可视化
    - `python eval.py --config configs/e0.yaml --ckpt results/e0/checkpoints/best.pt`
    - 验证metrics.csv, confusion_matrix.png, seg_overlays/
    - _Requirements: 8.1, 8.2_

## Phase 7: SSL预训练模块（必做，小规模可跑通）

- [ ] 15. SSL预训练实现
  - [ ] 15.1 创建 src/training/ssl_trainer.py SimCLR训练器
    - 对比学习损失（InfoNCE）
    - 支持MixedWM38 train图像作为无标签数据源（保守fallback）
    - _Requirements: 4.1_
  - [ ] 15.2 创建 train_ssl.py SSL预训练入口
    - 支持 --config configs/ssl.yaml
    - _Requirements: 4.1_
  - [ ] 15.3 创建 configs/ssl.yaml 和 configs/ssl_debug.yaml
    - ssl.yaml: 完整SSL预训练配置，支持启动、保存checkpoint、断点续训（不强制跑满epoch）
    - ssl_debug.yaml: 小规模快速验证配置
    - 若WM-811K不可用，自动使用MixedWM38 train图像
    - _Requirements: 4.1_
  - [ ] 15.4 运行SSL debug验证
    - `python train_ssl.py --config configs/ssl_debug.yaml`
    - 短时间内完成，产出：
      - results/ssl_debug/checkpoints/last.pt
      - results/ssl_debug/config_snapshot.yaml
      - results/ssl_debug/curves/loss_curve.png
    - 验收标准：ssl_debug.yaml必须在Final Checkpoint里跑通并产出指定文件
    - _Requirements: 4.1_

## Phase 8: E1 SSL权重加载实验

- [ ] 16. E1实验
  - [ ] 16.1 创建 configs/e1.yaml 配置
    - pretrained_weights默认使用Phase 15产出的ssl checkpoint
    - 提供可切换到公开checkpoint的选项（在报告里体现对比）
    - 配置key_mapping规则
    - _Requirements: 4.2_
  - [ ] 16.2 运行E1训练
    - 验证权重加载日志（matched/missing/unexpected）
    - 验证weight_loading.json生成
    - _Requirements: 4.2, 4.3, 4.4_
  - [ ] 16.3 运行E1评估
    - 对比E0结果
    - _Requirements: 4.4_

## Phase 9: E2长尾增强实验

- [ ] 17. E2长尾模块
  - [ ] 17.1 创建 src/data/sampler.py 加权采样器
    - WeightedClassSampler类
    - _Requirements: 5.1_
  - [ ] 17.2 更新损失函数支持Focal/ClassBalanced
    - _Requirements: 5.2_
  - [ ] 17.3 创建 configs/e2.yaml 配置
    - sampler: weighted
    - loss.classification: focal
    - _Requirements: 5.1, 5.2_
  - [ ] 17.4 运行E2训练与评估
    - 输出tail_class_analysis.csv
    - _Requirements: 5.3_

## Phase 10: E3成分分离实验（评估阶段生成）

- [ ] 18. E3分离模块
  - [ ] 18.1 创建 src/models/separation.py Prototype分离
    - PrototypeSeparator类（E3-Fallback）
    - 构建prototype，计算cosine similarity
    - 在eval阶段生成separation_maps（不需重新训练）
    - _Requirements: 6.1, 6.4_
  - [ ] 18.2 创建 configs/e3.yaml 配置
    - separation_enabled: true
    - separation_mode: prototype
    - 基于E1的checkpoint
    - _Requirements: 6.1_
  - [ ] 18.3 运行E3评估生成separation_maps
    - `python eval.py --config configs/e3.yaml --ckpt results/e1/checkpoints/best.pt`
    - 生成separation_maps/（图片+tensor）
    - _Requirements: 6.2, 6.3_

## Phase 11: 对比表与报告生成

- [ ] 19. 对比表生成
  - [ ] 19.1 创建 scripts/generate_comparison.py
    - 汇总E0/E1/E2/E3的metrics.csv
    - 计算delta
    - 输出results/comparison.csv
    - _Requirements: 8.5_

- [ ] 20. 报告与PPT生成
  - [ ] 20.1 创建 scripts/generate_report.py
    - 从results汇总生成report/REPORT.md
    - 包含表格、图片引用、命令清单
    - 若SSL使用MixedWM38作为数据源，自动写明该取舍
    - _Requirements: 10.1, 10.2_
  - [ ] 20.2 创建 scripts/generate_slides_md.py
    - 生成slides/SLIDES.md（10-12页结构）
    - _Requirements: 10.3_
  - [ ] 20.3 创建 scripts/build_pptx.py
    - 使用python-pptx生成slides/final.pptx
    - _Requirements: 10.4_

## Phase 12: 教学文档

- [ ] 21. 教学文档编写
  - [ ] 21.1 创建 docs/LEARNER_GUIDE.md
    - 命令清单、排错指南、概念小抄
    - 问题处理原则：记录假设 + 给出备选 + 默认保守实现
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  - [ ] 21.2 创建 docs/BUILD_PPT.md
    - PPT生成说明
    - _Requirements: 10.4_

## Phase 13: 最终验证

- [ ] 22. Final Checkpoint - 全流程验证
  - 验证完整流程并执行以下命令清单：
    ```bash
    # 1. Debug验证
    python train.py --config configs/e0.yaml --debug
    
    # 2. E0基线
    python train.py --config configs/e0.yaml
    python eval.py --config configs/e0.yaml --ckpt results/e0/checkpoints/best.pt
    
    # 3. SSL预训练（ssl_debug.yaml必须跑通并产出指定文件）
    python train_ssl.py --config configs/ssl_debug.yaml
    python train_ssl.py --config configs/ssl.yaml  # 支持启动、保存checkpoint、断点续训
    
    # 4. E1 SSL权重加载
    python train.py --config configs/e1.yaml
    python eval.py --config configs/e1.yaml --ckpt results/e1/checkpoints/best.pt
    
    # 5. E2长尾增强
    python train.py --config configs/e2.yaml
    python eval.py --config configs/e2.yaml --ckpt results/e2/checkpoints/best.pt
    
    # 6. E3成分分离
    python eval.py --config configs/e3.yaml --ckpt results/e1/checkpoints/best.pt
    
    # 7. 生成对比表
    python scripts/generate_comparison.py --results_root results --out results/comparison.csv
    
    # 8. 生成报告
    python scripts/generate_report.py --results_root results --out report/REPORT.md
    
    # 9. 生成PPT大纲
    python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md
    
    # 10. 生成PPT文件
    python scripts/build_pptx.py --slides_md slides/SLIDES.md --results_root results --out slides/final.pptx
    ```
  - 验收标准：
    - ssl_debug.yaml必须跑通并产出：results/ssl_debug/checkpoints/last.pt, config_snapshot.yaml, curves/loss_curve.png
    - ssl.yaml必须支持启动、保存checkpoint、断点续训（不强制跑满epoch）
  - 若遇问题：记录假设 + 给出备选方案 + 采用保守实现
