# PPT生成说明

本文档说明如何使用项目提供的脚本自动生成实验报告PPT。

---

## 📋 目录

1. [前置条件](#1-前置条件)
2. [快速开始](#2-快速开始)
3. [详细步骤](#3-详细步骤)
4. [脚本说明](#4-脚本说明)
5. [自定义修改](#5-自定义修改)
6. [常见问题](#6-常见问题)

---

## 1. 前置条件

### 1.1 依赖安装

PPT生成需要 `python-pptx` 库：

```bash
conda run -n wafer-seg-class pip install python-pptx
```

### 1.2 实验结果

确保已完成实验并生成结果文件：

```
results/
├── e0/
│   ├── metrics.csv
│   ├── confusion_matrix.png
│   └── seg_overlays/
├── e1/
│   ├── metrics.csv
│   └── weight_loading.json
├── e2/
│   ├── metrics.csv
│   └── tail_class_analysis.csv
└── ddpm_tail/
    ├── config_snapshot.yaml
    └── history.json
└── e3/
    ├── metrics.csv
    └── separation_maps/
```

---

## 2. 快速开始

### 一键生成PPT

```bash
# 步骤1：生成PPT大纲（Markdown格式）
conda run -n wafer-seg-class python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md

# 步骤2：生成PPT文件
conda run -n wafer-seg-class python scripts/build_pptx.py --slides_md slides/SLIDES.md --results_root results --out slides/final.pptx
```

### 预期输出

```
[INFO] 解析到 12 页幻灯片
[INFO] 处理第 1 页: 封面
[INFO] 处理第 2 页: 问题定义
...
[SUCCESS] PPTX已生成: slides/final.pptx
```

---

## 3. 详细步骤

### 3.1 生成PPT大纲

```bash
conda run -n wafer-seg-class python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--results_root` | `results` | 实验结果根目录 |
| `--out` | `slides/SLIDES.md` | 输出Markdown文件路径 |

**输出文件：** `slides/SLIDES.md`

**内容结构（10-12页）：**
1. 封面
2. 问题定义
3. 数据集介绍
4. 方法框图
5. 实验设计
6. E0基线结果
7. E1 SSL预训练对比
8. E2长尾增强
9. E3成分分离
10. 关键可视化
11. 消融实验总结
12. 结论与展望

### 3.2 生成PPT文件

```bash
conda run -n wafer-seg-class python scripts/build_pptx.py --slides_md slides/SLIDES.md --results_root results --out slides/final.pptx
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--slides_md` | `slides/SLIDES.md` | PPT大纲Markdown文件 |
| `--results_root` | `results` | 实验结果根目录（用于查找图片） |
| `--out` | `slides/final.pptx` | 输出PPTX文件路径 |

**输出文件：** `slides/final.pptx`

---

## 4. 脚本说明

### 4.1 generate_slides_md.py

**功能：** 从实验结果自动生成PPT大纲

**工作流程：**
1. 扫描 `results/` 目录，查找可用实验（e0, e1, e2, e3）
2. 读取各实验的 `metrics.csv` 提取指标
3. 生成包含表格、图片引用的Markdown文件

**自动提取的内容：**
- 各实验的 Macro-F1, Dice, IoU 指标
- 混淆矩阵图片路径
- 分割可视化图片路径
- 分离热力图路径

### 4.2 build_pptx.py

**功能：** 从Markdown大纲生成PPTX文件

**工作流程：**
1. 解析 `SLIDES.md` 文件，按 `---` 分割幻灯片
2. 提取每页的标题、内容、图片引用
3. 使用 `python-pptx` 创建演示文稿
4. 自动添加图片（如果存在）

**幻灯片布局：**
- 第1页：标题布局（封面）
- 其他页：标题+内容布局

**图片处理：**
- 自动解析Markdown图片语法 `![alt](path)`
- 支持相对路径（如 `../results/e0/confusion_matrix.png`）
- 每页最多添加2张图片

---

## 5. 自定义修改

### 5.1 修改PPT大纲

生成 `SLIDES.md` 后，可以手动编辑：

```bash
# 生成初始大纲
conda run -n wafer-seg-class python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md

# 手动编辑
notepad slides/SLIDES.md

# 重新生成PPT
conda run -n wafer-seg-class python scripts/build_pptx.py --slides_md slides/SLIDES.md --results_root results --out slides/final.pptx
```

### 5.2 SLIDES.md 格式说明

```markdown
## Slide N: 标题

### 副标题（可选）

内容文本...

- 列表项1
- 列表项2

![图片描述](../results/e0/confusion_matrix.png)

---

## Slide N+1: 下一页标题
...
```

**格式规则：**
- 使用 `---` 分隔幻灯片
- 使用 `## Slide N: 标题` 定义页面标题
- 使用 `![](path)` 引用图片
- 支持Markdown列表和表格

### 5.3 添加自定义页面

在 `SLIDES.md` 中添加新页面：

```markdown
---

## Slide 14: 附录

### 额外实验结果

- 实验细节1
- 实验细节2

![附加图片](../results/extra/figure.png)
```

### 5.4 修改作者信息

编辑 `SLIDES.md` 中的封面页：

```markdown
## Slide 1: 封面

### 晶圆工艺场景下的混合缺陷晶圆图谱多任务识别与可解释诊断

**副标题**: 自监督表征学习 + 长尾增强 + 弱监督成分分离

**作者**: 张三

**日期**: 2025年12月
```

---

## 6. 常见问题

### Q1: python-pptx未安装

**错误信息：**
```
[ERROR] python-pptx未安装，无法生成PPTX
```

**解决方案：**
```bash
conda run -n wafer-seg-class pip install python-pptx
```

### Q2: SLIDES.md文件不存在

**错误信息：**
```
[ERROR] SLIDES.md文件不存在: slides/SLIDES.md
```

**解决方案：**
```bash
# 先生成大纲
conda run -n wafer-seg-class python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md
```

### Q3: 图片无法添加到PPT

**警告信息：**
```
[WARN] 无法添加图片 ../results/e0/confusion_matrix.png: ...
```

**可能原因：**
1. 图片文件不存在
2. 路径错误

**解决方案：**
1. 确认实验已完成并生成图片
2. 检查 `SLIDES.md` 中的图片路径是否正确

### Q4: 结果目录不存在

**错误信息：**
```
[ERROR] 结果目录不存在: results
```

**解决方案：**
1. 确认已运行实验
2. 检查 `--results_root` 参数是否正确

### Q5: 如何只生成Markdown不生成PPTX

如果只需要Markdown大纲（用于其他PPT工具）：

```bash
# 只运行第一步
conda run -n wafer-seg-class python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md

# 然后手动使用其他工具（如Marp、Slidev）转换
```

### Q6: PPT样式不满意

**解决方案：**

1. **方法1**：手动编辑生成的PPTX文件
   - 用PowerPoint打开 `slides/final.pptx`
   - 修改样式、布局、字体等

2. **方法2**：修改脚本模板
   - 编辑 `scripts/build_pptx.py` 中的布局设置
   - 调整字体大小、图片位置等

3. **方法3**：使用其他工具
   - 将 `SLIDES.md` 导入到 Marp、Slidev 等工具
   - 使用自定义主题

---

## 7. 输出文件清单

成功执行后，应生成以下文件：

```
slides/
├── SLIDES.md      # PPT大纲（Markdown格式）
└── final.pptx     # PPT文件
```

---

## 8. 完整命令清单

```bash
# 1. 安装依赖
conda run -n wafer-seg-class pip install python-pptx

# 2. 生成PPT大纲
conda run -n wafer-seg-class python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md

# 3. 生成PPT文件
conda run -n wafer-seg-class python scripts/build_pptx.py --slides_md slides/SLIDES.md --results_root results --out slides/final.pptx

# 4. 查看生成的文件
dir slides
```

---

## 9. 降级方案

如果 `python-pptx` 安装失败或PPT生成有问题：

1. **使用Markdown大纲**
   - `slides/SLIDES.md` 包含完整的PPT结构
   - 可以手动复制内容到PowerPoint

2. **使用在线工具**
   - 将 `SLIDES.md` 上传到 [Marp](https://marp.app/)
   - 或使用 [Slidev](https://sli.dev/)

3. **手动创建PPT**
   - 参考 `SLIDES.md` 的结构
   - 从 `results/` 目录复制图片

---

**提示：** 建议先生成Markdown大纲，检查内容无误后再生成PPTX文件。
