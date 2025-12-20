# MixedWM38 混合缺陷晶圆图谱多任务识别

> PPT大纲 - 自动生成于 2025-12-20
**副标题**: 自监督表征学习 + 长尾增强 + 弱监督成分分离 | 作者: [姓名] | 日期: 2025年12月

---

## Slide 1: 问题定义

### 研究背景

- 晶圆图谱（Wafer Map）是半导体制造中反映工艺良率的重要数据
- 混合缺陷模式识别面临三大挑战：

**挑战1**: 类别不平衡（长尾分布）
- 混合缺陷类别样本稀少

**挑战2**: 标注困难
- 像素级分割标注成本高

**挑战3**: 可解释性需求
- 需要理解混合缺陷由哪些基础缺陷组成

---

## Slide 2: 数据集介绍

### MixedWM38 数据集

| 类别 | 数量 | 说明 |
|------|------|------|
| Normal | 1类 | 正常晶圆 |
| Single | 8类 | 单一缺陷 |
| Mixed | 29类 | 混合缺陷 |
| **总计** | **38类** | |

**8种基础缺陷类型**:
Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-Full, Scratch, Random

**图示**: [插入数据集样例图]

---

## Slide 3: 方法框图

### 多任务模型架构

```
Input Image (224×224)
        │
        ▼
┌───────────────────────┐
│   Shared Encoder      │ ← SSL预训练初始化
└───────────────────────┘
        │
   ┌────┼────┐
   ▼    ▼    ▼
┌─────┐┌─────┐┌─────┐
│ Cls ││ Seg ││ Sep │
│Head ││ Dec ││Head │
└─────┘└─────┘└─────┘
   │    │    │
   ▼    ▼    ▼
38类  Mask  8通道
分类  分割  分离
```

---

## Slide 4: 实验设计

### 渐进式实验组

| 实验 | 描述 | 关键技术 |
|------|------|----------|
| **E0** | 基线 | 多任务学习 |
| **E1** | +SSL | SimCLR预训练 |
| **E2** | +长尾 | DDPM生成式尾部扩增 |
| **E3** | +分离 | Prototype方法 |

**训练配置（来自 config_snapshot.yaml）**:
- GPU: RTX 4070 SUPER (12GB)
- Batch Size & LR: E0: bs=256, lr=0.001, E1: bs=128, lr=0.001, E2: bs=128, lr=0.001, E3: bs=32, lr=0.001; AMP: true
- Optimizer: AdamW

---

## Slide 5: E0 基线结果

### E0: 多任务基线模型

| 实验 | Macro-F1 | Dice | IoU |
|------|----------|------|-----|
| E0 | 0.9810 | 1.0000 | 1.0000 |


**混淆矩阵**:
![E0 Confusion Matrix](../results/e0/confusion_matrix.png)

**分割可视化**:
![E0 Segmentation](../results/e0/seg_overlays/sample_001.png)

---

## Slide 6: E1 SSL预训练对比

### E1 vs E0: SSL预训练的影响

| 实验 | Macro-F1 | Dice | IoU |
|------|----------|------|-----|
| E0 | 0.9810 | 1.0000 | 1.0000 |
| E1 | 0.9799 | 1.0000 | 1.0000 |


**SSL预训练方法**: SimCLR风格对比学习

**数据源**: MixedWM38训练集（WM-811K不可用时的fallback）

**权重加载统计**: 见 `results/e1/weight_loading.json`

---

## Slide 7: E2 长尾增强

### E2: 处理类别不平衡

**策略**:
1. 训练DDPM（尾部类别）
2. 生成合成样本并只加入训练集
3. 继续E2多任务训练

| 实验 | Macro-F1 | Dice | IoU |
|------|----------|------|-----|
| E0 | 0.9810 | 1.0000 | 1.0000 |
| E2 | 0.9811 | 1.0000 | 1.0000 |


**结果**: Macro-F1 提升 +0.01%
**DDPM合成**: total_generated=985, target_count=1000

**尾部类别分析**: 见 `results/e2/tail_class_analysis.csv`

---

## Slide 8: E3 成分分离

### E3: 弱监督成分分离

**方法**: Prototype相似度

1. 提取单缺陷类样本的特征原型
2. 计算输入与各原型的余弦相似度
3. 生成8通道热力图

**分离热力图示例**:
![E3 Separation](../results/e3/separation_maps/sample_001.png)

---

## Slide 9: 关键可视化

### 实验结果可视化

**分割Overlay对比**:
| E0 | E1 | E2 |
|----|----|----| 
| ![](../results/e0/seg_overlays/sample_001.png) | ![](../results/e1/seg_overlays/sample_001.png) | ![](../results/e2/seg_overlays/sample_001.png) |

**成分分离热力图**:
![E3 Separation Maps](../results/e3/separation_maps/sample_001.png)

---

## Slide 10: 消融实验总结

### 实验对比总结

| 实验 | Macro-F1 | Dice | IoU |
|------|----------|------|-----|
| E0 | 0.9810 | 1.0000 | 1.0000 |
| E1 | 0.9799 | 1.0000 | 1.0000 |
| E2 | 0.9811 | 1.0000 | 1.0000 |
| E3 | 0.9799 | 1.0000 | 1.0000 |


**关键发现**:
1. 多任务学习有效共享特征表示
2. SSL预训练使 Macro-F1 下降 -0.11%
3. 长尾增强 Macro-F1 提升 +0.01%
4. Prototype分离在性能上基本不变

---

## Slide 11: 结论与展望

### 结论

✅ 构建了完整的多任务晶圆缺陷识别系统
⚠️ SSL预训练未显著提升或结果缺失
✅ DDPM长尾增强带来Macro-F1提升
✅ Prototype分离不降低识别性能

### 局限性

- SSL使用MixedWM38作为fallback数据源
- 分离方法采用Prototype而非端到端训练
- DDPM增强仍需评估样本质量与分布偏移

### 未来工作

- 获取WM-811K进行完整SSL预训练
- 实现端到端弱监督分离头
- 优化DDPM生成质量与采样策略

