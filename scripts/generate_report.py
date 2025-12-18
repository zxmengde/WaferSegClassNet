#!/usr/bin/env python
"""
实验报告生成脚本

从results目录汇总生成report/REPORT.md
包含表格、图片引用、命令清单

用法:
    python scripts/generate_report.py --results_root results --out report/REPORT.md
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_json(filepath: Path) -> Optional[Dict]:
    """加载JSON文件"""
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def load_yaml(filepath: Path) -> Optional[Dict]:
    """加载YAML文件"""
    if not filepath.exists():
        return None
    try:
        import yaml
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def load_csv_as_dict(filepath: Path) -> Dict[str, Any]:
    """加载CSV文件为字典"""
    result = {}
    if not filepath.exists():
        return result
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        result[row[0]] = float(row[1])
                    except ValueError:
                        result[row[0]] = row[1]
    except Exception:
        pass
    return result


def check_ssl_data_source(results_root: Path) -> str:
    """
    检查SSL预训练使用的数据源
    
    Returns:
        数据源描述字符串
    """
    # 尝试多个可能的SSL配置路径
    ssl_paths = [
        results_root / "ssl_debug" / "config_snapshot.yaml",
        results_root / "ssl" / "config_snapshot.yaml",
    ]
    
    ssl_config = None
    for path in ssl_paths:
        ssl_config = load_yaml(path)
        if ssl_config is not None:
            break
    
    if ssl_config is None:
        # 检查是否有任何SSL相关的结果
        ssl_debug_dir = results_root / "ssl_debug"
        if ssl_debug_dir.exists():
            return "MixedWM38训练集（WM-811K不可用，使用保守fallback方案）"
        return "未进行SSL预训练"
    
    wm811k_path = ssl_config.get('data', {}).get('wm811k_path')
    dataset = ssl_config.get('data', {}).get('dataset', 'MixedWM38')
    
    if wm811k_path and Path(wm811k_path).exists():
        return "WM-811K数据集"
    else:
        return f"MixedWM38训练集（WM-811K不可用，使用保守fallback方案）"


def find_available_experiments(results_root: Path) -> Dict[str, Path]:
    """查找可用的实验目录"""
    experiments = {}
    exp_names = ["e0", "e1", "e2", "e3"]
    
    for exp in exp_names:
        # 优先完整实验，其次debug版本
        full_dir = results_root / exp
        debug_dir = results_root / f"{exp}_debug"
        
        if full_dir.exists() and (full_dir / "metrics.csv").exists():
            experiments[exp] = full_dir
        elif debug_dir.exists() and (debug_dir / "metrics.csv").exists():
            experiments[exp] = debug_dir
    
    return experiments


def generate_metrics_table(experiments: Dict[str, Path]) -> str:
    """生成指标对比表格"""
    if not experiments:
        return "暂无实验结果\n"
    
    # 读取所有实验的指标
    all_metrics = {}
    for exp, path in experiments.items():
        all_metrics[exp] = load_csv_as_dict(path / "metrics.csv")
    
    # 获取基线
    baseline = all_metrics.get("e0", {})
    
    # 生成Markdown表格
    lines = []
    lines.append("| 实验 | 来源 | Accuracy | Macro-F1 | Dice | IoU | vs E0 |")
    lines.append("|------|------|----------|----------|------|-----|-------|")
    
    for exp in ["e0", "e1", "e2", "e3"]:
        if exp not in experiments:
            continue
        
        metrics = all_metrics[exp]
        source = experiments[exp].name
        
        acc = metrics.get("Accuracy", 0)
        f1 = metrics.get("Macro-F1", 0)
        dice = metrics.get("Dice", 0)
        iou = metrics.get("IoU", 0)
        
        # 计算delta
        if exp == "e0":
            delta = "-"
        else:
            baseline_f1 = baseline.get("Macro-F1", 0)
            if baseline_f1 > 0:
                delta_pct = ((f1 - baseline_f1) / baseline_f1) * 100
                delta = f"{delta_pct:+.1f}%"
            else:
                delta = "N/A"
        
        lines.append(f"| {exp.upper()} | {source} | {acc:.4f} | {f1:.4f} | {dice:.4f} | {iou:.4f} | {delta} |")
    
    return "\n".join(lines) + "\n"



def generate_report_content(results_root: Path) -> str:
    """生成完整的报告内容"""
    experiments = find_available_experiments(results_root)
    ssl_source = check_ssl_data_source(results_root)
    
    # 获取元数据
    meta = load_json(results_root / "e0" / "meta.json")
    if meta is None:
        meta = load_json(results_root / "e0_debug" / "meta.json")
    
    git_commit = meta.get("git_commit", "unknown") if meta else "unknown"
    seed = meta.get("seed", 42) if meta else 42
    
    report = f"""# 实验报告：MixedWM38混合缺陷晶圆图谱多任务识别

> 自动生成于 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> Git Commit: {git_commit}
> Random Seed: {seed}

## 摘要

本实验针对晶圆工艺场景下的混合缺陷晶圆图谱（Wafer Map）进行多任务识别研究。系统支持三个核心任务：
- **T1 分类**：38类单标签分类（1 normal + 8 single + 29 mixed）
- **T2 分割**：二值缺陷定位分割
- **T3 分离**：8通道弱监督成分分离（针对混合缺陷）

实验采用渐进式设计（E0→E3），逐步引入自监督预训练、长尾增强和弱监督分离技术。

## 背景与动机

### 问题定义

在半导体制造过程中，晶圆图谱（Wafer Map）是反映工艺良率的重要数据。混合缺陷模式的识别面临以下挑战：
1. **类别不平衡**：混合缺陷类别样本稀少（长尾分布）
2. **标注困难**：像素级分割标注成本高
3. **可解释性需求**：需要理解混合缺陷由哪些基础缺陷组成

### 数据集

- **MixedWM38**：混合缺陷晶圆图谱数据集，包含38类（1 normal + 8 single + 29 mixed）
- **WM-811K**：大规模无标签晶圆图谱数据集，用于自监督预训练

## 方法

### 实验组设计

| 实验 | 描述 | 关键技术 |
|------|------|----------|
| E0 | 基线模型 | 多任务学习（分类+分割） |
| E1 | SSL预训练 | SimCLR风格对比学习初始化encoder |
| E2 | 长尾增强 | 加权采样 + Focal Loss |
| E3 | 成分分离 | 8通道弱监督分离头（Prototype方法） |

### SSL预训练数据源

**本实验SSL预训练使用的数据源**: {ssl_source}

{"**注意**：由于WM-811K数据集不可用，SSL预训练使用MixedWM38训练集作为保守fallback方案。这可能导致预训练效果不如使用完整WM-811K数据集。" if "MixedWM38" in ssl_source else ""}

### 模型架构

```
┌─────────────────────────────────────────────────────────┐
│                    Input Image (224×224)                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Shared Encoder (Custom/ResNet)             │
│                  (可选SSL预训练初始化)                    │
└─────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │Classification │ │  Segmentation │ │  Separation   │
    │  Head (38类)  │ │    Decoder    │ │  Head (8通道) │
    └───────────────┘ └───────────────┘ └───────────────┘
            │               │               │
            ▼               ▼               ▼
       cls_logits      seg_mask       sep_heatmaps
```

## 实验设置

### 硬件环境

- GPU: NVIDIA RTX 4070 SUPER (12GB)
- CPU: Intel Core Ultra 9 285H
- RAM: 32GB
- OS: Windows 11

### 训练配置

- Batch Size: 32 (debug模式: 8)
- Learning Rate: 0.001
- Optimizer: AdamW
- Scheduler: Cosine Annealing
- AMP: 启用
- Epochs: 100 (debug模式: 2)

## 实验结果

### 主要指标对比

{generate_metrics_table(experiments)}

### 可视化结果

"""
    
    # 添加可视化图片引用
    for exp in ["e0", "e1", "e2", "e3"]:
        if exp in experiments:
            exp_path = experiments[exp]
            report += f"""
#### {exp.upper()} 实验结果

**混淆矩阵**:
![{exp.upper()} Confusion Matrix](../{exp_path}/confusion_matrix.png)

**分割可视化**: 见 `{exp_path}/seg_overlays/` 目录
"""
            
            # E3特有的分离热力图
            if exp == "e3":
                sep_maps_dir = exp_path / "separation_maps"
                if sep_maps_dir.exists():
                    report += f"""
**成分分离热力图**: 见 `{exp_path}/separation_maps/` 目录
"""
    
    return report



def generate_ablation_section(experiments: Dict[str, Path]) -> str:
    """生成消融实验部分"""
    section = """
## 消融实验分析

### E0 → E1: SSL预训练的影响

自监督预训练通过对比学习在无标签数据上学习特征表示，理论上可以：
- 提供更好的特征初始化
- 提升模型泛化能力
- 减少对标注数据的依赖

### E0 → E2: 长尾增强的影响

针对类别不平衡问题，E2实验采用：
- **加权采样**：按类别频率倒数加权
- **Focal Loss**：降低易分类样本的损失权重
- **强增强**：对尾部类别应用更强的数据增强

### E1 → E3: 成分分离的影响

E3在E1基础上添加8通道分离头，使用Prototype方法：
1. 从训练集提取单缺陷类样本的特征原型
2. 计算输入图像与各原型的余弦相似度
3. 生成8通道热力图表示各基础缺陷的分布

"""
    return section


def generate_commands_section() -> str:
    """生成命令清单部分"""
    return """
## 复现命令清单

### 1. 环境准备

```bash
# 创建conda环境
conda env create -f environment.yml
conda activate wafer

# 或使用pip
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 准备MixedWM38数据集
python scripts/prepare_mixedwm38.py
```

### 3. Debug模式验证（5分钟内完成）

```bash
python train.py --config configs/e0.yaml --debug
```

### 4. E0基线实验

```bash
# 训练
python train.py --config configs/e0.yaml

# 评估
python eval.py --config configs/e0.yaml --ckpt results/e0/checkpoints/best.pt
```

### 5. SSL预训练

```bash
# Debug验证
python train_ssl.py --config configs/ssl_debug.yaml

# 完整预训练（可选）
python train_ssl.py --config configs/ssl.yaml
```

### 6. E1实验（SSL权重加载）

```bash
# 训练
python train.py --config configs/e1.yaml

# 评估
python eval.py --config configs/e1.yaml --ckpt results/e1/checkpoints/best.pt
```

### 7. E2实验（长尾增强）

```bash
# 训练
python train.py --config configs/e2.yaml

# 评估
python eval.py --config configs/e2.yaml --ckpt results/e2/checkpoints/best.pt
```

### 8. E3实验（成分分离）

```bash
# 评估（基于E1的checkpoint）
python eval.py --config configs/e3.yaml --ckpt results/e1/checkpoints/best.pt
```

### 9. 生成对比表和报告

```bash
# 生成对比表
python scripts/generate_comparison.py --results_root results --out results/comparison.csv

# 生成报告
python scripts/generate_report.py --results_root results --out report/REPORT.md

# 生成PPT大纲
python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md

# 生成PPT文件（可选）
python scripts/build_pptx.py --slides_md slides/SLIDES.md --results_root results --out slides/final.pptx
```

"""


def generate_conclusion_section() -> str:
    """生成结论部分"""
    return """
## 结论与展望

### 主要发现

1. **多任务学习有效**：E0基线验证了分类和分割任务可以共享特征表示
2. **SSL预训练**：E1实验探索了自监督预训练对下游任务的影响
3. **长尾处理**：E2实验通过加权采样和Focal Loss缓解类别不平衡
4. **可解释性**：E3实验通过Prototype方法提供混合缺陷的成分分离可视化

### 局限性与取舍

1. **SSL数据源**：由于WM-811K数据集不可用，SSL预训练使用MixedWM38训练集作为fallback
2. **分离方法**：采用Prototype相似度方法而非端到端训练的分离头，作为弱监督方案的降级实现
3. **长尾增强**：未实现DDPM生成式增强，采用类均衡采样+Focal Loss作为替代方案

### 未来工作

1. 获取WM-811K数据集进行完整SSL预训练
2. 实现端到端的弱监督分离头训练
3. 探索DDPM生成式数据增强
4. 扩展到更多缺陷类型和工艺场景

## 参考文献

1. MixedWM38 Dataset
2. SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
3. Focal Loss for Dense Object Detection
4. U-Net: Convolutional Networks for Biomedical Image Segmentation

---

*本报告由 `scripts/generate_report.py` 自动生成*
"""


def generate_full_report(results_root: Path) -> str:
    """生成完整报告"""
    experiments = find_available_experiments(results_root)
    
    report = generate_report_content(results_root)
    report += generate_ablation_section(experiments)
    report += generate_commands_section()
    report += generate_conclusion_section()
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="生成实验报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/generate_report.py --results_root results --out report/REPORT.md
        """
    )
    
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="实验结果根目录 (默认: results)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="report/REPORT.md",
        help="输出Markdown文件路径 (默认: report/REPORT.md)"
    )
    
    args = parser.parse_args()
    
    results_root = Path(args.results_root)
    output_path = Path(args.out)
    
    if not results_root.exists():
        print(f"[ERROR] 结果目录不存在: {results_root}")
        return 1
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成报告
    report_content = generate_full_report(results_root)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"[SUCCESS] 报告已生成: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
