5# 代码迁移决策文档

## 现有代码分析

### 框架
- **当前框架**: TensorFlow 2.x / Keras
- **目标框架**: PyTorch 2.x

### 迁移原因
1. PyTorch 对 AMP 混合精度训练支持更好
2. PyTorch 的断点续训实现更直观
3. PyTorch 社区资源更丰富，便于后续扩展
4. 用户硬件（RTX 4070 SUPER）对 PyTorch 支持良好

## 可复用组件

### 1. 配置与映射（直接复用）
- `CLASS_MAPPING`: 38类标签到ID的映射
- `CLASS_NAME_MAPPING`: ID到类名的映射
- `IMAGE_SIZE`: (224, 224)
- 其他超参数配置

### 2. 网络架构设计（需重写，保留设计思路）

| TensorFlow 组件 | PyTorch 对应 | 说明 |
|----------------|-------------|------|
| `convBlock` | `ConvBlock` | Conv2d + BN + Dropout + SepConv + BN + MaxPool |
| `gapConvBlock` | `GapConvBlock` | 同上，但用 AvgPool |
| `terminalConvBlock` | `TerminalConvBlock` | 无池化的卷积块 |
| `transposeConvBlock` | `TransposeConvBlock` | 转置卷积 + skip connection |
| `getEncoder` | `WaferEncoder` | 编码器网络 |
| `getDecoder` | `WaferDecoder` | U-Net 风格解码器 |
| `addProjectionHead` | `ProjectionHead` | 对比学习投影头 |
| `getModel` | `WaferMultiTaskModel` | 多任务模型 |

### 3. 损失函数（需重写，保留公式）

| TensorFlow 损失 | PyTorch 对应 | 说明 |
|----------------|-------------|------|
| `diceCoef` | `dice_coef` | Dice 系数计算 |
| `diceCoefLoss` | `DiceLoss` | 1 - Dice |
| `bceDiceLoss` | `BCEDiceLoss` | BCE + Dice |
| `SupervisedContrastiveLoss` | `InfoNCELoss` | 对比学习损失 |

### 4. 数据加载（需重写，保留逻辑）

| TensorFlow 组件 | PyTorch 对应 | 说明 |
|----------------|-------------|------|
| `waferSegDataLoaderContrastive` | `MixedWM38Dataset` (SSL mode) | SSL 数据加载 |
| `waferSegClassDataLoader` | `MixedWM38Dataset` | 多任务数据加载 |
| `getDataLoader` | `get_dataloaders` | 数据加载器工厂 |

## 需要新增的功能

### 1. 训练模块
- AMP 混合精度训练
- 断点续训（保存/加载完整训练状态）
- 可配置的 best checkpoint 指标
- meta.json 记录（git commit, seed）

### 2. 评估模块
- Macro-F1, mAP, Dice, IoU 计算
- 混淆矩阵生成
- 分割 overlay 可视化
- 分离热力图可视化

### 3. 配置系统
- YAML 配置文件
- 配置数据类验证
- config_snapshot 保存

### 4. 长尾处理（E2）
- WeightedRandomSampler
- FocalLoss
- ClassBalancedLoss

### 5. 成分分离（E3）
- PrototypeSeparator
- 8通道热力图生成

## 迁移策略

### Phase 1: 基础设施
1. 创建 PyTorch 项目结构
2. 实现配置系统
3. 实现数据加载模块

### Phase 2: 模型
1. 实现编码器（基于现有设计）
2. 实现解码器
3. 实现多任务模型
4. 实现损失函数

### Phase 3: 训练
1. 实现 Trainer 类
2. 实现 AMP 和断点续训
3. 实现评估模块

### Phase 4: 扩展
1. 实现 SSL 预训练
2. 实现长尾处理
3. 实现成分分离

## 验证计划

1. **数据加载验证**: 确保 PyTorch DataLoader 输出与 TF 一致
2. **模型前向验证**: 确保输出形状正确
3. **损失计算验证**: 使用相同输入验证损失值
4. **训练验证**: Debug 模式下完成一次完整训练循环

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 模型性能差异 | 使用相同超参数，对比验证 |
| 训练不稳定 | 使用 AMP，监控梯度 |
| 显存不足 | 使用 grad_accum_steps |
| 依赖冲突 | 使用 conda 环境隔离 |
