# 数据目录规范

## 目录结构

```
data/
├── raw/                          # 原始数据（不提交到git）
│   └── Wafer_Map_Datasets.npz    # MixedWM38 原始数据文件
│
├── processed/                    # 处理后的数据（不提交到git）
│   ├── Images/                   # RGB 晶圆图像 (.npy)
│   │   ├── Image_0.npy
│   │   ├── Image_1.npy
│   │   └── ...
│   ├── Labels/                   # 标签文件 (.npy)
│   │   ├── Image_0.npy
│   │   ├── Image_1.npy
│   │   └── ...
│   ├── Masks/                    # 分割 mask (.npy)
│   │   ├── Image_0.npy
│   │   ├── Image_1.npy
│   │   └── ...
│   └── data_stats.txt            # 数据统计信息
│
├── synthetic/                    # DDPM 合成数据（不提交到git）
│   └── ddpm/
│       ├── Images/               # 合成RGB晶圆图像 (.npy)
│       ├── Labels/               # 合成标签 (.npy, 8维)
│       ├── Masks/                # 合成mask (.npy)
│       └── synthetic_stats.json  # 合成统计信息
│
└── README.md                     # 数据说明
```

## MixedWM38 数据集

### 数据集简介

MixedWM38 是一个混合缺陷晶圆图谱数据集，包含：
- **38 类**：1 个正常类 + 8 个单缺陷类 + 29 个混合缺陷类
- **约 38,015 个样本**
- **图像尺寸**：处理后为 224×224×3

### 8 类基础缺陷

| ID | 名称 | 英文 | 描述 |
|----|------|------|------|
| 1 | 中心 | Center | 缺陷集中在晶圆中心 |
| 2 | 甜甜圈 | Donut | 环形缺陷 |
| 3 | 边缘-左 | Edge-Loc (EL) | 边缘局部缺陷 |
| 4 | 边缘-右 | Edge-Ring (ER) | 边缘环形缺陷 |
| 5 | 局部 | Loc | 局部缺陷 |
| 6 | 近边 | Near-Full (NF) | 近全片缺陷 |
| 7 | 划痕 | Scratch (S) | 划痕缺陷 |
| 8 | 随机 | Random | 随机分布缺陷 |

### 数据获取

1. **下载数据集**：
   - 从原始论文或公开仓库获取 `Wafer_Map_Datasets.npz`
   - 放置到 `data/raw/` 目录

2. **运行数据准备脚本**：
   ```bash
   # 完整数据
   conda run -n wafer-seg-class python scripts/prepare_mixedwm38.py --input data/raw/Wafer_Map_Datasets.npz --output data/processed
   
   # Debug 模式（每类最多5样本）
   conda run -n wafer-seg-class python scripts/prepare_mixedwm38.py --input data/raw/Wafer_Map_Datasets.npz --output data/processed --debug
   ```

3. **验证数据完整性**：
   ```bash
   conda run -n wafer-seg-class python scripts/sanity_check_data.py --data_root data/processed
   ```

## 数据格式说明

### Images (RGB 图像)

- 格式：NumPy 数组 (.npy)
- 形状：(224, 224, 3)
- 数据类型：uint8
- 值范围：0-255
- 颜色编码：
  - 背景：(255, 0, 255) 紫色
  - 正常区域：(0, 255, 255) 青色
  - 缺陷区域：(255, 255, 0) 黄色

### Labels (标签)

- 格式：NumPy 数组 (.npy)
- 形状：(8,) - 8 位二进制向量
- 数据类型：int
- 编码：每位表示一种基础缺陷是否存在
  - 例如：[1, 0, 1, 0, 0, 0, 0, 0] 表示 Center + EL 混合缺陷

### Masks (分割掩码)

- 格式：NumPy 数组 (.npy)
- 形状：(224, 224, 1)
- 数据类型：uint8
- 值：0（非缺陷）或 255（缺陷）

### Synthetic (DDPM 合成数据)

- 目录结构与 `processed/` 一致
- `synthetic_stats.json` 记录合成数量、目标阈值与生成配置
- 合成样本仅用于训练集（E2）

## WM-811K 数据集（可选，用于 SSL 预训练）

如果需要使用 WM-811K 进行自监督预训练：

1. 下载 WM-811K 数据集
2. 放置到 `data/raw/MIR-WM811K/` 目录（确保包含 `Python/WM811K.pkl`）
3. 运行预处理脚本：

```bash
conda run -n wafer-seg-class python scripts/prepare_wm811k.py --input data/raw/MIR-WM811K/Python/WM811K.pkl --output data/wm811k
```

输出目录：
- `data/wm811k/Images/`（RGB .npy）
- `data/wm811k/metadata.csv`

**注意**：如果 WM-811K 不可用，系统会自动使用 MixedWM38 的训练图像作为 SSL 数据源。

## .gitignore 配置

确保以下内容在 `.gitignore` 中：

```
# 数据文件
data/raw/
data/processed/
data/wm811k/
data/synthetic/
*.npz
*.npy
```

## 常见问题

### Q: 数据文件太大无法下载？

A: 可以使用 debug 模式先跑通流程：
```bash
conda run -n wafer-seg-class python scripts/prepare_mixedwm38.py --debug --max-per-class 5
```

### Q: 标签分布不均衡怎么办？

A: 这是正常的长尾分布。当前 E2 使用 DDPM 生成式尾部增强，必要时可叠加加权采样或 Focal Loss。

### Q: 没有真实的分割 mask 怎么办？

A: 系统会自动生成伪 mask（基于阈值分割），并记录生成参数到 config_snapshot.yaml。
