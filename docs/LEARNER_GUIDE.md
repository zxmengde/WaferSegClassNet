# 新手学习指南

> 面向深度学习新手的完整教程，手把手教你完成晶圆缺陷识别实验
> 
> **问题处理原则**：记录假设 + 给出备选方案 + 默认采用保守实现

---

## 📚 目录

1. [快速开始命令清单](#1-快速开始命令清单)
2. [环境配置](#2-环境配置)
3. [数据准备](#3-数据准备)
4. [实验执行（E0-E3）](#4-实验执行e0-e3)
5. [常见报错排查](#5-常见报错排查)
6. [关键概念小抄](#6-关键概念小抄)
7. [如何读懂训练日志](#7-如何读懂训练日志)
8. [问题处理原则](#8-问题处理原则)

---

## 1. 快速开始命令清单

### 1.1 环境安装

```bash
# 方法1：使用 environment.yml（推荐）
conda env create -f environment.yml

# 方法2：手动安装
conda create -n wafer-seg-class python=3.10 -y
conda run -n wafer-seg-class pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
conda run -n wafer-seg-class pip install -r requirements.txt
```

### 1.2 数据准备

```bash
# 完整数据
conda run -n wafer-seg-class python scripts/prepare_mixedwm38.py --input data/raw/MixedWM38.npz --output data/processed

# Debug模式（每类最多5样本，快速验证）
conda run -n wafer-seg-class python scripts/prepare_mixedwm38.py --input data/raw/MixedWM38.npz --output data/processed --debug --max-per-class 5

# 验证数据完整性
conda run -n wafer-seg-class python scripts/sanity_check_data.py --data_root data/processed
```

### 1.3 Debug训练（5分钟内完成）

```bash
conda run -n wafer-seg-class python train.py --config configs/e0.yaml --debug
```

### 1.4 完整实验流程

```bash
# ========== E0 基线实验 ==========
conda run -n wafer-seg-class python train.py --config configs/e0.yaml
conda run -n wafer-seg-class python eval.py --config configs/e0.yaml --ckpt results/e0/checkpoints/best.pt

# ========== SSL 预训练 ==========
# Debug验证（快速）
conda run -n wafer-seg-class python train_ssl.py --config configs/ssl_debug.yaml

# 完整SSL预训练（可选，支持断点续训）
conda run -n wafer-seg-class python train_ssl.py --config configs/ssl.yaml

# ========== E1 SSL权重加载实验 ==========
conda run -n wafer-seg-class python train.py --config configs/e1.yaml
conda run -n wafer-seg-class python eval.py --config configs/e1.yaml --ckpt results/e1/checkpoints/best.pt

# ========== DDPM 生成式尾部增强 ==========
conda run -n wafer-seg-class python scripts/train_ddpm.py --config configs/ddpm.yaml
conda run -n wafer-seg-class python scripts/sample_ddpm.py --config configs/ddpm.yaml --ckpt results/ddpm_tail/checkpoints/best.pt

# ========== E2 长尾增强实验 ==========
conda run -n wafer-seg-class python train.py --config configs/e2.yaml
conda run -n wafer-seg-class python eval.py --config configs/e2.yaml --ckpt results/e2/checkpoints/best.pt

# ========== E3 成分分离实验 ==========
# 基于E1的checkpoint生成分离热力图
conda run -n wafer-seg-class python eval.py --config configs/e3.yaml --ckpt results/e1/checkpoints/best.pt

# ========== 生成报告和PPT ==========
# 生成对比表
conda run -n wafer-seg-class python scripts/generate_comparison.py --results_root results --out results/comparison.csv

# 生成实验报告
conda run -n wafer-seg-class python scripts/generate_report.py --results_root results --out report/REPORT.md

# 生成PPT大纲
conda run -n wafer-seg-class python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md

# 生成PPT文件
conda run -n wafer-seg-class python scripts/build_pptx.py --slides_md slides/SLIDES.md --results_root results --out slides/final.pptx
```

### 1.5 断点续训

```bash
# 从最后的checkpoint恢复训练
conda run -n wafer-seg-class python train.py --config configs/e0.yaml --resume results/e0/checkpoints/last.pt
```

---

## 2. 环境配置

### 2.1 激活conda环境

**命令：**
```bash
# 无需手动激活，后续命令统一使用 conda run -n wafer-seg-class
```

**预期输出：**
- 无需激活，命令行前缀保持不变

**验证环境：**
```bash
conda run -n wafer-seg-class python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**预期输出：**
```
PyTorch: 2.5.1+cu121
CUDA: True
```

### 2.2 验证完整环境

```bash
conda run -n wafer-seg-class python scripts/verify_setup.py
```

**预期输出：**
```
✓ PyTorch: 2.x.x
✓ CUDA available: True
✓ GPU: NVIDIA GeForce RTX 4070 SUPER
✓ All dependencies installed
```

### 2.3 常见环境问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 找不到conda命令 | conda未添加到PATH | 使用Anaconda Prompt或重新安装 |
| CUDA不可用 | PyTorch版本与CUDA不匹配 | 重装PyTorch（见SETUP_WINDOWS.md） |
| ModuleNotFoundError | 依赖未安装 | `conda run -n wafer-seg-class pip install -r requirements.txt` |

---

## 3. 数据准备

### 3.1 数据集放置

```
data/
├── raw/
│   ├── Wafer_Map_Datasets.npz      ← MixedWM38 原始数据
│   └── MIR-WM811K/
│       └── Python/WM811K.pkl       ← WM-811K 原始数据（SSL用）
└── processed/           ← 运行脚本后自动生成
```

### 3.2 运行数据准备

```bash
# Debug模式（推荐先用这个验证流程）
conda run -n wafer-seg-class python scripts/prepare_mixedwm38.py --input data/raw/Wafer_Map_Datasets.npz --output data/processed --debug --max-per-class 5

# 完整数据
conda run -n wafer-seg-class python scripts/prepare_mixedwm38.py --input data/raw/Wafer_Map_Datasets.npz --output data/processed
```

**可选：WM-811K（用于SSL预训练）**
```bash
conda run -n wafer-seg-class python scripts/prepare_wm811k.py --input data/raw/MIR-WM811K/Python/WM811K.pkl --output data/wm811k
```

**预期输出：**
```
[Info] Loading data from data/raw/Wafer_Map_Datasets.npz
[Info] Total samples: 38015
[Info] Processing images...
[Info] Saved to data/processed/
✓ Data preparation completed!
```

### 3.3 数据格式说明

**38类标签映射：**
- 类0：Normal（正常）
- 类1-8：8种单一缺陷（Center, Donut, EL, ER, LOC, NF, S, Random）
- 类9-37：29种混合缺陷

**8类多标签格式：**
```
[Center, Donut, Edge-Loc, Edge-Ring, Local, Near-full, Scratch, Random]
例如：[1, 0, 1, 0, 0, 0, 0, 0] 表示 Center + Edge-Loc 混合缺陷
```

---

## 4. 实验执行（E0-E3）

### 4.1 E0 基线实验

**目的：** 建立多任务学习基线（分类+分割）

**训练：**
```bash
conda run -n wafer-seg-class python train.py --config configs/e0.yaml
```

**评估：**
```bash
conda run -n wafer-seg-class python eval.py --config configs/e0.yaml --ckpt results/e0/checkpoints/best.pt
```

**输出目录：**
```
results/e0/
├── checkpoints/
│   ├── best.pt          # 最佳模型（按Macro-F1）
│   └── last.pt          # 最后epoch模型
├── metrics.csv          # 指标汇总
├── confusion_matrix.png # 混淆矩阵
├── seg_overlays/        # 分割可视化
├── curves/              # 训练曲线
├── config_snapshot.yaml # 配置快照
└── meta.json            # 元信息（git commit, seed）
```

### 4.2 E1 SSL预训练实验

**目的：** 使用自监督预训练提升特征表示

**步骤1：SSL预训练（可选）**
```bash
# Debug验证
conda run -n wafer-seg-class python train_ssl.py --config configs/ssl_debug.yaml

# 完整预训练
conda run -n wafer-seg-class python train_ssl.py --config configs/ssl.yaml
```

**步骤2：E1训练**
```bash
conda run -n wafer-seg-class python train.py --config configs/e1.yaml
```

**步骤3：评估**
```bash
conda run -n wafer-seg-class python eval.py --config configs/e1.yaml --ckpt results/e1/checkpoints/best.pt
```

**验证权重加载：**
- 查看 `results/e1/weight_loading.json`
- 应包含 `matched`, `missing`, `unexpected` 字段

### 4.3 E2 长尾增强实验

**目的：** 处理类别不平衡问题

**步骤1：DDPM生成尾部样本**
```bash
conda run -n wafer-seg-class python scripts/train_ddpm.py --config configs/ddpm.yaml
conda run -n wafer-seg-class python scripts/sample_ddpm.py --config configs/ddpm.yaml --ckpt results/ddpm_tail/checkpoints/best.pt
```

**训练：**
```bash
conda run -n wafer-seg-class python train.py --config configs/e2.yaml
```

**评估：**
```bash
conda run -n wafer-seg-class python eval.py --config configs/e2.yaml --ckpt results/e2/checkpoints/best.pt
```

**特殊输出：**
- `results/e2/tail_class_analysis.csv` - 尾部类别分析
- `data/synthetic/ddpm/synthetic_stats.json` - 合成样本统计

### 4.4 E3 成分分离实验

**目的：** 对混合缺陷进行成分分离

**评估（基于E1模型）：**
```bash
conda run -n wafer-seg-class python eval.py --config configs/e3.yaml --ckpt results/e1/checkpoints/best.pt
```

**特殊输出：**
```
results/e3/
├── separation_maps/     # 8通道分离热力图
│   ├── sample_xxx.png   # 可视化图片
│   └── sample_xxx.pt    # 原始tensor
└── prototypes.pt        # 原型向量
```

---

## 5. 常见报错排查

### 5.1 CUDA相关

#### 问题：CUDA out of memory

**症状：**
```
RuntimeError: CUDA out of memory. Tried to allocate xxx MiB
```

**排查步骤：**
1. 检查当前batch_size（建议从16开始）
2. 启用AMP混合精度：`training.amp_enabled: true`
3. 使用梯度累积：`training.grad_accum_steps: 2`
4. 降低image_size：`data.image_size: [128, 128]`

**修改配置示例：**
```yaml
data:
  batch_size: 8          # 从32降到8

training:
  amp_enabled: true      # 启用混合精度
  grad_accum_steps: 2    # 梯度累积
```

#### 问题：CUDA not available

**症状：**
```python
>>> torch.cuda.is_available()
False
```

**排查步骤：**
1. 检查NVIDIA驱动：`nvidia-smi`
2. 检查PyTorch CUDA版本：`conda run -n wafer-seg-class python -c "import torch; print(torch.version.cuda)"`
3. 重新安装PyTorch：
   ```bash
   conda run -n wafer-seg-class pip uninstall torch torchvision -y
   conda run -n wafer-seg-class pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### 5.2 依赖相关

#### 问题：ModuleNotFoundError

**症状：**
```
ModuleNotFoundError: No module named 'xxx'
```

**解决：**
```bash
conda run -n wafer-seg-class pip install xxx
```

**常见缺失包：**
```bash
conda run -n wafer-seg-class pip install opencv-python pyyaml tqdm hypothesis python-pptx
```

#### 问题：版本冲突

**解决：**
```bash
conda run -n wafer-seg-class pip install -r requirements.txt --force-reinstall
```

### 5.3 路径相关

#### 问题：FileNotFoundError

**症状：**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/processed/Images/...'
```

**排查步骤：**
1. 检查数据是否准备完成：
   ```bash
   dir data\processed\Images
   ```
2. 重新运行数据准备脚本
3. 检查配置文件中的 `data_root` 路径

### 5.4 显存不足排查步骤

**按优先级尝试：**

| 步骤 | 操作 | 配置修改 |
|------|------|----------|
| 1 | 降低batch_size | `data.batch_size: 8` |
| 2 | 启用AMP | `training.amp_enabled: true` |
| 3 | 使用梯度累积 | `training.grad_accum_steps: 2` |
| 4 | 降低图像尺寸 | `data.image_size: [128, 128]` |
| 5 | 减少num_workers | `data.num_workers: 0` |

### 5.5 训练不收敛

**现象：** Loss不下降，Acc一直是0

**可能原因及解决：**
1. **学习率过大**：改为 `learning_rate: 0.0001`
2. **数据问题**：运行 `conda run -n wafer-seg-class python scripts/sanity_check_data.py`
3. **模型问题**：先用debug模式验证

---

## 6. 关键概念小抄

### 6.1 评估指标

#### Macro-F1（主指标）

**定义：** 各类别F1分数的算术平均

**公式：**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Macro-F1 = (1/N) × Σ F1_i
```

**特点：**
- 对类别不平衡敏感
- 每个类别权重相同
- 范围：0-1（越高越好）

#### Dice系数（分割指标）

**定义：** 预测mask与真实mask的重叠度

**公式：**
```
Dice = 2 × |A ∩ B| / (|A| + |B|)
```

**特点：**
- 范围：0-1（越高越好）
- 对小目标敏感

#### IoU（交并比）

**定义：** 预测mask与真实mask的交集/并集

**公式：**
```
IoU = |A ∩ B| / |A ∪ B|
```

**与Dice的关系：**
```
Dice = 2 × IoU / (1 + IoU)
```

#### mAP（多标签指标）

**定义：** 各类别Average Precision的平均值

**使用场景：** 8类多标签分类

### 6.2 深度学习术语

| 术语 | 解释 |
|------|------|
| Epoch | 模型看完整个训练集一遍 |
| Batch | 一次前向传播处理的样本数 |
| Learning Rate | 参数更新的步长 |
| Loss | 模型预测与真实值的差距 |
| Overfitting | 训练集好、验证集差 |
| Underfitting | 训练集和验证集都差 |

### 6.3 本实验特有概念

#### 对比学习（Contrastive Learning）

**原理：** 通过拉近相似样本、推远不相似样本来学习表征

**本项目应用：** SimCLR风格的自监督预训练（E1）

**数据增强要求：** 晶圆友好（旋转、翻转），避免大裁剪

#### 长尾分布（Long-Tail）

**定义：** 少数类别样本数远少于多数类别

**本项目情况：** 某些混合缺陷类只有几十个样本

**解决方案（E2）：**
- DDPM 生成式尾部增强（合成样本加入训练集）
- 类均衡采样（WeightedRandomSampler）
- Focal Loss / Class-Balanced Loss

#### 弱监督（Weak Supervision）

**定义：** 使用不完整或噪声标签进行训练

**本项目应用（E3）：**
- 只有图像级标签（"有Center缺陷"）
- 没有像素级标签
- 使用原型相似度生成分离热力图

#### 多任务学习（Multi-Task Learning）

**原理：** 同时训练多个相关任务，共享特征表示

**本项目任务：**
- T1：38类分类
- T2：二值分割
- T3：8通道成分分离

---

## 7. 如何读懂训练日志

### 7.1 训练日志示例

```
Epoch 1/100
Training: 100%|██████████| 152/152 [00:45<00:00, 3.37it/s, loss=4.15, acc=0.046, dice=0.629]
Validating: 100%|██████████| 19/19 [00:02<00:00, 9.12it/s]
Train - Loss: 4.1537, Acc: 0.0461, Dice: 0.6291
Val - Loss: 4.2863, Acc: 0.0000, Dice: 0.6902, Macro-F1: 0.0000
Saved best model (macro_f1: 0.0000)
```

### 7.2 关键信息解读

| 信息 | 含义 |
|------|------|
| `152/152` | 处理了152个batch |
| `3.37it/s` | 每秒处理3.37个batch |
| `00:45` | 本epoch用时45秒 |
| `loss=4.15` | 当前batch的损失 |
| `Macro-F1: 0.0000` | 验证集宏平均F1（主指标） |

### 7.3 判断训练是否正常

#### ✅ 正常训练特征

```
Epoch 1:  Loss=4.15, Macro-F1=0.05
Epoch 10: Loss=2.50, Macro-F1=0.35
Epoch 50: Loss=1.20, Macro-F1=0.65
Epoch 100: Loss=0.80, Macro-F1=0.75
```

- Loss逐渐下降
- Macro-F1逐渐上升
- Train和Val指标差距不大（<10%）

#### ❌ 过拟合

```
Epoch 50: Train Loss=0.5, Val Loss=3.2
Epoch 51: Train Loss=0.4, Val Loss=3.5
```

**特征：** Train Loss很低，Val Loss很高或上升

**解决：**
- Early Stopping
- 数据增强
- Dropout

#### ❌ 欠拟合

```
Epoch 100: Train Loss=3.8, Val Loss=3.9
```

**特征：** Train Loss和Val Loss都很高

**解决：**
- 增加模型容量
- 训练更多epoch
- 调整学习率

#### ❌ Loss不下降

```
Epoch 1: Loss=4.15
Epoch 2: Loss=4.14
Epoch 3: Loss=4.16
```

**可能原因：**
- 学习率过大或过小
- 数据问题
- 模型问题

**解决：** 先用debug模式验证

---

## 8. 问题处理原则

### 8.1 核心原则

当遇到问题时，遵循以下原则：

1. **记录假设**：明确说明你认为问题的原因
2. **给出备选方案**：提供至少2-3个可能的解决方案
3. **默认保守实现**：选择最稳定、最简单的方案

### 8.2 示例：显存不足

**假设：** batch_size=32对于12GB显存可能过大

**备选方案：**
1. 降低batch_size到16或8
2. 启用AMP混合精度
3. 使用梯度累积
4. 降低图像尺寸

**保守实现：** 先降低batch_size到8，这是最简单且最可靠的方案

### 8.3 示例：SSL预训练数据不可用

**假设：** WM-811K数据集不可用或处理失败

**备选方案：**
1. 使用MixedWM38训练集作为SSL数据源
2. 先排查 `data/raw/MIR-WM811K/Python/WM811K.pkl` 是否存在
3. 使用公开的预训练权重

**保守实现：** 使用MixedWM38训练集，这样不需要额外数据

### 8.4 示例：分离头实现复杂

**假设：** 完整的弱监督分离训练可能过于复杂

**备选方案：**
1. 实现完整的弱监督训练
2. 使用原型相似度方法（不需要额外训练）
3. 使用CAM方法

**保守实现：** 使用原型相似度方法，在eval阶段生成分离热力图

---

## 9. 实验结果验证清单

### 9.1 E0 基线验证

- [ ] `results/e0/metrics.csv` 存在且包含 Macro-F1, Dice, IoU
- [ ] `results/e0/confusion_matrix.png` 存在
- [ ] `results/e0/seg_overlays/` 包含至少10张图片
- [ ] `results/e0/config_snapshot.yaml` 存在
- [ ] `results/e0/meta.json` 包含 git_commit 和 seed

### 9.2 E1 SSL验证

- [ ] `results/e1/weight_loading.json` 存在
- [ ] weight_loading.json 包含 matched, missing, unexpected 字段
- [ ] E1的Macro-F1应该 >= E0（SSL应该有帮助）

### 9.3 E2 长尾验证

- [ ] `results/e2/tail_class_analysis.csv` 存在
- [ ] `data/synthetic/ddpm/synthetic_stats.json` 记录合成样本数
- [ ] 检查尾部类别F1变化（若下降需记录失败原因）

### 9.4 E3 分离验证

- [ ] `results/e3/separation_maps/` 存在
- [ ] 包含8通道热力图可视化
- [ ] `results/e3/prototypes.pt` 存在

### 9.5 报告验证

- [ ] `results/comparison.csv` 包含E0/E1/E2/E3对比
- [ ] `report/REPORT.md` 包含完整实验报告
- [ ] `slides/SLIDES.md` 包含10-12页PPT大纲
- [ ] `slides/final.pptx` 存在（可选）

---

## 10. 获取帮助

### 10.1 遇到问题时

1. **查看日志**：`results/<exp_name>/train.log`
2. **检查配置**：`configs/<exp_name>.yaml`
3. **运行debug**：快速定位问题
4. **查看本指南**：常见问题章节

### 10.2 调试技巧

```bash
# 快速验证流程
conda run -n wafer-seg-class python train.py --config configs/e0.yaml --debug

# 检查数据
conda run -n wafer-seg-class python scripts/sanity_check_data.py --data_root data/processed

# 验证环境
conda run -n wafer-seg-class python scripts/verify_setup.py
```

### 10.3 记住

- 深度学习是实验科学，多试多调
- 每次只改一个参数，观察效果
- 保存好的checkpoint，避免重复训练
- 遇到问题先用debug模式验证

---

**祝你实验顺利！🎉**
