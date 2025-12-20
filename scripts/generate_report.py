#!/usr/bin/env python
"""
实验报告生成脚本

从results目录汇总生成report/REPORT.md
包含表格、图片引用、命令清单

用法:
    conda run -n wafer-seg-class python scripts/generate_report.py --results_root results --out report/REPORT.md
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


MAIN_EXPERIMENTS = ["e0", "e1", "e2", "e3"]
LONG_TAIL_EXPERIMENTS = ["e2", "e2_focal", "e2_cb", "e2_ddpm_wide"]


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


def load_metrics_by_experiment(experiments: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    """加载所有实验的指标"""
    metrics = {}
    for exp, path in experiments.items():
        metrics[exp] = load_csv_as_dict(path / "metrics.csv")
    return metrics


def get_metric_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    """安全读取指标数值"""
    if not metrics:
        return None
    value = metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_tail_summary(exp_path: Path) -> Dict[str, Any]:
    """加载 tail_class_analysis.csv 的汇总信息"""
    tail_path = exp_path / "tail_class_analysis.csv"
    if not tail_path.exists():
        return {}
    summary = {}
    try:
        with open(tail_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                key = row[0].strip()
                if key in ["Tail Macro-F1", "Baseline Tail Macro-F1", "Tail Delta", "Tail Delta %"]:
                    summary[key] = row[1].strip()
    except Exception:
        return {}
    return summary


def get_tail_metric_value(summary: Dict[str, Any], key: str) -> Optional[float]:
    """从 tail summary 中读取数值"""
    if not summary:
        return None
    value = summary.get(key)
    try:
        return float(str(value).replace("%", ""))
    except (TypeError, ValueError):
        return None


def format_cell(value: Any) -> str:
    """格式化表格单元格"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "N/A"
    return str(value)


def is_long_tail_failed(e2_f1: Optional[float], e0_f1: Optional[float]) -> bool:
    """判断长尾实验是否失败"""
    if e2_f1 is None:
        return False
    if e2_f1 <= 0.05:
        return True
    if e0_f1 is None or e0_f1 <= 0:
        return False
    return e2_f1 < e0_f1 * 0.2


def generate_training_config_table(experiments: Dict[str, Path]) -> str:
    """生成训练配置表格（来自config_snapshot.yaml）"""
    lines = []
    lines.append("| 实验 | Batch Size | Learning Rate | Optimizer | Scheduler | AMP | Epochs |")
    lines.append("|------|------------|---------------|-----------|-----------|-----|--------|")
    
    debug_note = ""
    for exp in MAIN_EXPERIMENTS:
        if exp not in experiments:
            continue
        cfg = load_yaml(experiments[exp] / "config_snapshot.yaml")
        if cfg is None:
            lines.append(f"| {exp.upper()} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        data_cfg = cfg.get("data", {})
        train_cfg = cfg.get("training", {})
        lines.append(
            "| {exp} | {bs} | {lr} | {opt} | {sch} | {amp} | {epochs} |".format(
                exp=exp.upper(),
                bs=format_cell(data_cfg.get("batch_size")),
                lr=format_cell(train_cfg.get("learning_rate")),
                opt=format_cell(train_cfg.get("optimizer")),
                sch=format_cell(train_cfg.get("scheduler")),
                amp=format_cell(train_cfg.get("amp_enabled")),
                epochs=format_cell(train_cfg.get("epochs")),
            )
        )
        if not debug_note:
            debug_epochs = train_cfg.get("debug_epochs")
            debug_batch_size = train_cfg.get("debug_batch_size")
            if debug_epochs is not None or debug_batch_size is not None:
                debug_note = f"\n> Debug模式: epochs={format_cell(debug_epochs)}, batch_size={format_cell(debug_batch_size)}\n"
    
    return "\n".join(lines) + (debug_note if debug_note else "\n")


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
    
    data_cfg = ssl_config.get('data', {})
    wm811k_path = data_cfg.get('wm811k_path')
    max_samples = data_cfg.get('max_samples')
    
    if wm811k_path and Path(wm811k_path).exists():
        if max_samples:
            return f"WM-811K数据集（max_samples={max_samples}）"
        return "WM-811K数据集"
    return "MixedWM38训练集（WM-811K不可用，使用保守fallback方案）"


def load_ddpm_info(results_root: Path, exp_name: str, synthetic_root: str) -> Dict[str, Any]:
    """加载DDPM相关信息（训练配置与合成统计）"""
    ddpm_dir = results_root / exp_name
    synthetic_path = results_root.parent / Path(synthetic_root) / "synthetic_stats.json"
    info: Dict[str, Any] = {
        "config": load_yaml(ddpm_dir / "config_snapshot.yaml"),
        "history": load_json(ddpm_dir / "history.json"),
        "synthetic_stats": load_json(synthetic_path),
    }
    return info


def find_available_experiments(results_root: Path, exp_names: Optional[List[str]] = None) -> Dict[str, Path]:
    """查找可用的实验目录"""
    experiments = {}
    exp_names = exp_names or MAIN_EXPERIMENTS
    
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
    
    for exp in MAIN_EXPERIMENTS:
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


def generate_long_tail_table(experiments: Dict[str, Path]) -> str:
    """生成长尾对比表格（含 Tail Macro-F1）"""
    if not experiments:
        return "暂无长尾对比结果\n"

    lines = []
    lines.append("| 实验 | Loss | tail_threshold | synthetic_root | Macro-F1 | Tail Macro-F1 |")
    lines.append("|------|------|----------------|----------------|----------|---------------|")

    for exp in LONG_TAIL_EXPERIMENTS:
        if exp not in experiments:
            continue
        exp_path = experiments[exp]
        metrics = load_csv_as_dict(exp_path / "metrics.csv")
        cfg = load_yaml(exp_path / "config_snapshot.yaml") or {}
        loss_type = cfg.get("loss", {}).get("classification")
        tail_threshold = cfg.get("data", {}).get("tail_class_threshold")
        synthetic_root = cfg.get("data", {}).get("synthetic_root")
        synthetic_root = synthetic_root if synthetic_root else "-"

        macro_f1 = get_metric_value(metrics, "Macro-F1")
        tail_summary = load_tail_summary(exp_path)
        tail_macro = get_tail_metric_value(tail_summary, "Tail Macro-F1")

        lines.append(
            "| {exp} | {loss} | {thr} | {sr} | {f1} | {tail_f1} |".format(
                exp=exp,
                loss=format_cell(loss_type),
                thr=format_cell(tail_threshold),
                sr=format_cell(synthetic_root),
                f1=format_cell(macro_f1),
                tail_f1=format_cell(tail_macro),
            )
        )

    return "\n".join(lines) + "\n"



def generate_report_content(results_root: Path) -> str:
    """生成完整的报告内容"""
    experiments = find_available_experiments(results_root, MAIN_EXPERIMENTS)
    tail_experiments = find_available_experiments(results_root, LONG_TAIL_EXPERIMENTS)
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
| E2 | 长尾增强（DDPM） | DDPM生成尾部样本 + 合成样本加入训练集 |
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

### 训练配置（来自 config_snapshot.yaml）

{generate_training_config_table(experiments)}

## 实验结果

### 主要指标对比

{generate_metrics_table(experiments)}

### 长尾对比（DDPM + Loss）

{generate_long_tail_table(tail_experiments)}

### 可视化结果

"""
    
    # 添加可视化图片引用
    for exp in ["e0", "e1", "e2", "e3"]:
        if exp in experiments:
            exp_path = experiments[exp]
            exp_path_display = exp_path.as_posix()
            report += f"""
#### {exp.upper()} 实验结果

**混淆矩阵**:
![{exp.upper()} Confusion Matrix](../{exp_path_display}/confusion_matrix.png)

**分割可视化**: 见 `{exp_path_display}/seg_overlays/` 目录
"""
            
            # E3特有的分离热力图
            if exp == "e3":
                sep_maps_dir = exp_path / "separation_maps"
                if sep_maps_dir.exists():
                    report += f"""
**成分分离热力图**: 见 `{exp_path_display}/separation_maps/` 目录
"""
    
    return report



def generate_ablation_section(experiments: Dict[str, Path], results_root: Path) -> str:
    """生成消融实验部分"""
    metrics = load_metrics_by_experiment(experiments)
    tail_experiments = find_available_experiments(results_root, LONG_TAIL_EXPERIMENTS)
    e0_f1 = get_metric_value(metrics.get("e0", {}), "Macro-F1")
    e1_f1 = get_metric_value(metrics.get("e1", {}), "Macro-F1")
    e2_f1 = get_metric_value(metrics.get("e2", {}), "Macro-F1")
    e3_f1 = get_metric_value(metrics.get("e3", {}), "Macro-F1")
    ddpm_info = load_ddpm_info(results_root, "ddpm_tail", "data/synthetic/ddpm")
    ddpm_wide_info = load_ddpm_info(results_root, "ddpm_tail_wide", "data/synthetic/ddpm_wide")
    synthetic_stats = ddpm_info.get("synthetic_stats") if ddpm_info else None
    synthetic_wide_stats = ddpm_wide_info.get("synthetic_stats") if ddpm_wide_info else None
    
    lines = []
    lines.append("## 消融实验分析")
    lines.append("")
    lines.append("### E0 → E1: SSL预训练的影响")
    lines.append("")
    lines.append("自监督预训练通过对比学习在无标签数据上学习特征表示，理论上可以：")
    lines.append("- 提供更好的特征初始化")
    lines.append("- 提升模型泛化能力")
    lines.append("- 减少对标注数据的依赖")
    lines.append("")
    if e0_f1 is not None and e1_f1 is not None and e0_f1 > 0:
        delta_pct = (e1_f1 - e0_f1) / e0_f1 * 100
        trend = "提升" if delta_pct >= 0 else "下降"
        lines.append(
            f"**实验结果：E1 Macro-F1 = {e1_f1:.4f}，比 E0 {trend} {delta_pct:+.2f}%**"
        )
    else:
        lines.append("**实验结果：E1 或 E0 结果缺失，无法判断提升幅度。**")
    
    lines.append("")
    lines.append("### E0 → E2: 长尾增强的影响")
    lines.append("")
    lines.append("E2使用DDPM生成尾部类合成样本，并将合成样本仅加入训练集以缓解长尾问题。")
    e2_cfg = load_yaml(experiments["e2"] / "config_snapshot.yaml") if "e2" in experiments else None
    if e2_cfg:
        data_cfg = e2_cfg.get("data", {})
        train_cfg = e2_cfg.get("training", {})
        loss_cfg = e2_cfg.get("loss", {})
        aug_cfg = data_cfg.get("augmentation", {})
        lines.append(
            "- **关键参数**: lr={lr}, weight_decay={wd}, grad_accum={ga}, "
            "sampler={sampler}, loss={loss}, morph_noise={mn}, synthetic_root={sr}".format(
                lr=format_cell(train_cfg.get("learning_rate")),
                wd=format_cell(train_cfg.get("weight_decay")),
                ga=format_cell(train_cfg.get("grad_accum_steps")),
                sampler=format_cell(data_cfg.get("sampler")),
                loss=format_cell(loss_cfg.get("classification")),
                mn=format_cell(aug_cfg.get("morphological_noise")),
                sr=format_cell(data_cfg.get("synthetic_root")),
            )
        )
        if synthetic_stats:
            total_generated = synthetic_stats.get("total_generated")
            target_count = synthetic_stats.get("target_count")
            lines.append(
                "- **DDPM合成**: total_generated={tg}, target_count={tc}, output_root={oroot}".format(
                    tg=format_cell(total_generated),
                    tc=format_cell(target_count),
                    oroot=format_cell(synthetic_stats.get("output_root")),
                )
            )
        if synthetic_wide_stats:
            total_generated = synthetic_wide_stats.get("total_generated")
            target_classes = synthetic_wide_stats.get("target_classes") or []
            lines.append(
                "- **DDPM扩覆盖合成**: total_generated={tg}, tail_classes={tc}, output_root={oroot}".format(
                    tg=format_cell(total_generated),
                    tc=format_cell(len(target_classes)),
                    oroot=format_cell(synthetic_wide_stats.get("output_root")),
                )
            )
    else:
        lines.append("- **关键参数**: 无法读取 config_snapshot.yaml")
    
    if e2_f1 is None or e0_f1 is None:
        lines.append("")
        lines.append("**实验结果：E2 结果缺失，无法评估长尾策略效果。**")
    elif is_long_tail_failed(e2_f1, e0_f1):
        lines.append("")
        lines.append(
            f"**实验结果：训练失败（Macro-F1 = {e2_f1:.4f}，显著低于 E0 = {e0_f1:.4f}）**"
        )
        lines.append("")
        lines.append("**可能原因：**")
        lines.append("1. DDPM生成样本质量不足或分布偏移")
        lines.append("2. 合成样本数量不足，未显著缓解长尾")
        lines.append("3. 训练策略仍偏保守，优化不足")
        lines.append("")
        lines.append("**替代方案建议：**")
        lines.append("1. 提升DDPM训练轮数或提高采样分辨率")
        lines.append("2. 增加合成样本数量并加入质量过滤")
        lines.append("3. 叠加Focal Loss或Class-Balanced Loss")
    else:
        delta_pct = (e2_f1 - e0_f1) / e0_f1 * 100 if e0_f1 > 0 else 0
        trend = "提升" if delta_pct >= 0 else "下降"
        lines.append("")
        lines.append(
            f"**实验结果：E2 Macro-F1 = {e2_f1:.4f}，比 E0 {trend} {delta_pct:+.2f}%**"
        )
    
    lines.append("")
    lines.append("### E2 长尾策略对比")
    lines.append("")
    if tail_experiments:
        lines.append(generate_long_tail_table(tail_experiments))
    else:
        lines.append("暂无长尾对比结果。")
    lines.append("")
    lines.append("### E1 → E3: 成分分离的影响")
    lines.append("")
    lines.append("E3在E1基础上添加8通道分离头，使用Prototype方法：")
    lines.append("1. 从训练集提取单缺陷类样本的特征原型")
    lines.append("2. 计算输入图像与各原型的余弦相似度")
    lines.append("3. 生成8通道热力图表示各基础缺陷的分布")
    if e1_f1 is not None and e3_f1 is not None:
        delta = e3_f1 - e1_f1
        if abs(delta) <= 0.001:
            effect = "性能基本不变"
        elif delta > 0:
            effect = "性能小幅提升"
        else:
            effect = "性能小幅下降"
        lines.append("")
        lines.append(
            f"**实验结果：E3 Macro-F1 = {e3_f1:.4f}，E1 Macro-F1 = {e1_f1:.4f}，{effect}。**"
        )
    
    lines.append("")
    return "\n".join(lines)


def generate_commands_section() -> str:
    """生成命令清单部分"""
    return """
## 复现命令清单

### 1. 环境准备

```bash
# 创建conda环境（如已存在可跳过）
conda env create -f environment.yml

# 安装依赖与环境检查
conda run -n wafer-seg-class pip install -r requirements.txt
conda run -n wafer-seg-class python check_env.py
```

### 2. 数据准备

```bash
# 准备MixedWM38数据集
conda run -n wafer-seg-class python scripts/prepare_mixedwm38.py

# 准备WM-811K数据集（SSL预训练用）
conda run -n wafer-seg-class python scripts/prepare_wm811k.py --input data/raw/MIR-WM811K/Python/WM811K.pkl --output data/wm811k
```

### 3. Debug模式验证（5分钟内完成）

```bash
conda run -n wafer-seg-class python train.py --config configs/e0.yaml --debug
```

### 4. E0基线实验

```bash
# 训练
conda run -n wafer-seg-class python train.py --config configs/e0.yaml

# 评估
conda run -n wafer-seg-class python eval.py --config configs/e0.yaml --ckpt results/e0/checkpoints/best.pt
```

### 5. SSL预训练

```bash
# Debug验证
conda run -n wafer-seg-class python train_ssl.py --config configs/ssl_debug.yaml

# 完整预训练（默认使用WM-811K，如不可用则fallback）
conda run -n wafer-seg-class python train_ssl.py --config configs/ssl.yaml
```

### 6. E1实验（SSL权重加载）

```bash
# 训练
conda run -n wafer-seg-class python train.py --config configs/e1.yaml

# 评估
conda run -n wafer-seg-class python eval.py --config configs/e1.yaml --ckpt results/e1/checkpoints/best.pt
```

### 7. DDPM生成式尾部增强（E2前置）

```bash
# 训练DDPM
conda run -n wafer-seg-class python scripts/train_ddpm.py --config configs/ddpm.yaml

# 生成合成样本
conda run -n wafer-seg-class python scripts/sample_ddpm.py --config configs/ddpm.yaml --ckpt results/ddpm_tail/checkpoints/best.pt

# 扩覆盖版本（更多尾部类）
conda run -n wafer-seg-class python scripts/train_ddpm.py --config configs/ddpm_wide.yaml
conda run -n wafer-seg-class python scripts/sample_ddpm.py --config configs/ddpm_wide.yaml --ckpt results/ddpm_tail_wide/checkpoints/best.pt
```

### 8. E2实验（长尾增强）

```bash
# 训练
conda run -n wafer-seg-class python train.py --config configs/e2.yaml

# 评估
conda run -n wafer-seg-class python eval.py --config configs/e2.yaml --ckpt results/e2/checkpoints/best.pt

# 长尾对比：Focal / Class-Balanced
conda run -n wafer-seg-class python train.py --config configs/e2_focal.yaml
conda run -n wafer-seg-class python eval.py --config configs/e2_focal.yaml --ckpt results/e2_focal/checkpoints/best.pt

conda run -n wafer-seg-class python train.py --config configs/e2_cb.yaml
conda run -n wafer-seg-class python eval.py --config configs/e2_cb.yaml --ckpt results/e2_cb/checkpoints/best.pt

# DDPM扩覆盖对比
conda run -n wafer-seg-class python train.py --config configs/e2_ddpm_wide.yaml
conda run -n wafer-seg-class python eval.py --config configs/e2_ddpm_wide.yaml --ckpt results/e2_ddpm_wide/checkpoints/best.pt
```

### 9. E3实验（成分分离）

```bash
# 评估（基于E1的checkpoint）
conda run -n wafer-seg-class python eval.py --config configs/e3.yaml --ckpt results/e1/checkpoints/best.pt
```

### 10. 生成对比表和报告

```bash
# 生成对比表
conda run -n wafer-seg-class python scripts/generate_comparison.py --results_root results --out results/comparison.csv

# 扩展对比表（含E2变体）
conda run -n wafer-seg-class python scripts/generate_comparison.py --results_root results --out results/comparison_extended.csv --experiments e0 e1 e2 e2_focal e2_cb e2_ddpm_wide e3 --baseline e0

# 生成报告
conda run -n wafer-seg-class python scripts/generate_report.py --results_root results --out report/REPORT.md

# 生成PPT大纲
conda run -n wafer-seg-class python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md

# 生成PPT文件（可选）
conda run -n wafer-seg-class python scripts/build_pptx.py --slides_md slides/SLIDES.md --results_root results --out slides/final.pptx
```

"""


def generate_conclusion_section(experiments: Dict[str, Path], results_root: Path) -> str:
    """生成结论部分"""
    metrics = load_metrics_by_experiment(experiments)
    ssl_source = check_ssl_data_source(results_root)
    e0_f1 = get_metric_value(metrics.get("e0", {}), "Macro-F1")
    e1_f1 = get_metric_value(metrics.get("e1", {}), "Macro-F1")
    e2_f1 = get_metric_value(metrics.get("e2", {}), "Macro-F1")
    e3_f1 = get_metric_value(metrics.get("e3", {}), "Macro-F1")
    ddpm_info = load_ddpm_info(results_root, "ddpm_tail", "data/synthetic/ddpm")
    ddpm_wide_info = load_ddpm_info(results_root, "ddpm_tail_wide", "data/synthetic/ddpm_wide")
    synthetic_stats = ddpm_info.get("synthetic_stats") if ddpm_info else None
    synthetic_wide_stats = ddpm_wide_info.get("synthetic_stats") if ddpm_wide_info else None
    ddpm_ready = bool(
        (synthetic_stats and synthetic_stats.get("total_generated", 0))
        or (synthetic_wide_stats and synthetic_wide_stats.get("total_generated", 0))
    )
    
    lines = []
    lines.append("## 结论与展望")
    lines.append("")
    lines.append("### 主要发现")
    lines.append("")
    lines.append("1. **多任务学习有效**：E0基线验证了分类和分割任务可以共享特征表示")
    if e0_f1 is not None and e1_f1 is not None and e0_f1 > 0:
        delta_pct = (e1_f1 - e0_f1) / e0_f1 * 100
        trend = "提升" if delta_pct >= 0 else "下降"
        lines.append(
            f"2. **SSL预训练**：E1 Macro-F1 {trend} {delta_pct:+.2f}%（{e1_f1:.4f} vs {e0_f1:.4f}）"
        )
    else:
        lines.append("2. **SSL预训练**：E1结果缺失或无法比较")
    
    if e2_f1 is not None and e0_f1 is not None:
        if is_long_tail_failed(e2_f1, e0_f1):
            lines.append(
                f"3. **长尾处理（DDPM）**：E2 Macro-F1 显著下降（{e2_f1:.4f}），DDPM增强未达预期"
            )
        else:
            delta_pct = (e2_f1 - e0_f1) / e0_f1 * 100 if e0_f1 > 0 else 0
            trend = "提升" if delta_pct >= 0 else "下降"
            lines.append(
                f"3. **长尾处理（DDPM）**：E2 Macro-F1 {trend} {delta_pct:+.2f}%（{e2_f1:.4f}）"
            )
    else:
        if ddpm_ready:
            lines.append("3. **长尾处理（DDPM）**：已完成合成样本生成，但E2结果缺失")
        else:
            lines.append("3. **长尾处理（DDPM）**：DDPM未完成或E2结果缺失")
    
    if e1_f1 is not None and e3_f1 is not None:
        delta = e3_f1 - e1_f1
        if abs(delta) <= 0.001:
            effect = "性能基本不变"
        elif delta > 0:
            effect = "性能小幅提升"
        else:
            effect = "性能小幅下降"
        lines.append(
            f"4. **可解释性**：E3提供成分分离热力图，且{effect}"
        )
    else:
        lines.append("4. **可解释性**：E3结果缺失或未运行")
    
    lines.append("")
    lines.append("### 局限性与取舍")
    lines.append("")
    if "WM-811K" in ssl_source:
        lines.append("1. **SSL数据源**：已使用WM-811K进行预训练（数据规模更充分）")
    elif "MixedWM38" in ssl_source:
        lines.append("1. **SSL数据源**：WM-811K不可用，SSL预训练使用MixedWM38训练集作为fallback")
    else:
        lines.append("1. **SSL数据源**：SSL未运行或数据源不可识别")
    lines.append("2. **分离方法**：采用Prototype相似度方法而非端到端训练的分离头")
    if ddpm_ready:
        if e2_f1 is not None and e0_f1 is not None and is_long_tail_failed(e2_f1, e0_f1):
            lines.append("3. **长尾增强**：DDPM增强未达预期，需要优化合成质量/比例")
        else:
            lines.append("3. **长尾增强**：已引入DDPM合成样本，仍需评估质量与分布偏移")
    else:
        lines.append("3. **长尾增强**：DDPM未完成，仍需补充生成式增强")
    
    lines.append("")
    lines.append("### 未来工作")
    lines.append("")
    if "WM-811K" in ssl_source:
        lines.append("1. 扩展SSL预训练策略（更多数据增强与更长训练）")
        lines.append("2. 实现端到端的弱监督分离头训练")
    else:
        lines.append("1. 获取WM-811K数据集进行完整SSL预训练")
        lines.append("2. 实现端到端的弱监督分离头训练")
    if ddpm_ready:
        lines.append("3. 优化DDPM生成质量与采样策略")
    else:
        lines.append("3. 完成DDPM生成式数据增强")
    lines.append("4. 扩展到更多缺陷类型和工艺场景")
    lines.append("")
    lines.append("## 参考文献")
    lines.append("")
    lines.append("1. MixedWM38 Dataset")
    lines.append("2. SimCLR: A Simple Framework for Contrastive Learning of Visual Representations")
    lines.append("3. Focal Loss for Dense Object Detection")
    lines.append("4. U-Net: Convolutional Networks for Biomedical Image Segmentation")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*本报告由 `scripts/generate_report.py` 自动生成*")
    lines.append("")
    
    return "\n".join(lines)


def generate_full_report(results_root: Path) -> str:
    """生成完整报告"""
    experiments = find_available_experiments(results_root, MAIN_EXPERIMENTS)
    
    report = generate_report_content(results_root)
    report += generate_ablation_section(experiments, results_root)
    report += generate_commands_section()
    report += generate_conclusion_section(experiments, results_root)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="生成实验报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    conda run -n wafer-seg-class python scripts/generate_report.py --results_root results --out report/REPORT.md
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
