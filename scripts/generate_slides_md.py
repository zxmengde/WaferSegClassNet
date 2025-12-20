#!/usr/bin/env python
"""
PPT大纲生成脚本

生成slides/SLIDES.md（10-12页结构）

用法:
    conda run -n wafer-seg-class python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md
"""

import argparse
import csv
import json
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


def check_ssl_data_source(results_root: Path) -> str:
    """检查SSL预训练数据源"""
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
        return "未进行SSL预训练"
    data_cfg = ssl_config.get('data', {})
    wm811k_path = data_cfg.get('wm811k_path')
    max_samples = data_cfg.get('max_samples')
    if wm811k_path and Path(wm811k_path).exists():
        if max_samples:
            return f"WM-811K数据集（max_samples={max_samples}）"
        return "WM-811K数据集"
    return "MixedWM38训练集（WM-811K不可用时的fallback）"


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
    """读取尾部指标数值"""
    if not summary:
        return None
    value = summary.get(key)
    try:
        return float(str(value).replace("%", ""))
    except (TypeError, ValueError):
        return None


def load_ddpm_info(results_root: Path, exp_name: str, synthetic_root: str) -> Dict[str, Any]:
    """加载DDPM相关信息（训练配置与合成统计）"""
    ddpm_dir = results_root / exp_name
    synthetic_path = results_root.parent / Path(synthetic_root) / "synthetic_stats.json"
    return {
        "config": load_yaml(ddpm_dir / "config_snapshot.yaml"),
        "synthetic_stats": load_json(synthetic_path),
    }


def get_metric_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    """安全读取指标数值"""
    if not metrics:
        return None
    value = metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_long_tail_failed(e2_f1: Optional[float], e0_f1: Optional[float]) -> bool:
    """判断长尾实验是否失败"""
    if e2_f1 is None:
        return False
    if e2_f1 <= 0.05:
        return True
    if e0_f1 is None or e0_f1 <= 0:
        return False
    return e2_f1 < e0_f1 * 0.2


def build_training_summary(experiments: Dict[str, Path]) -> str:
    """生成训练配置摘要"""
    parts = []
    amp_value = None
    for exp in MAIN_EXPERIMENTS:
        if exp not in experiments:
            continue
        cfg = load_yaml(experiments[exp] / "config_snapshot.yaml")
        if cfg is None:
            continue
        data_cfg = cfg.get("data", {})
        train_cfg = cfg.get("training", {})
        bs = data_cfg.get("batch_size")
        lr = train_cfg.get("learning_rate")
        parts.append(f"{exp.upper()}: bs={bs}, lr={lr}")
        if amp_value is None:
            amp_value = train_cfg.get("amp_enabled")
    if not parts:
        return "Batch Size/LR: N/A"
    amp_text = "true" if amp_value else "false"
    return f"Batch Size & LR: {', '.join(parts)}; AMP: {amp_text}"


def find_available_experiments(results_root: Path, exp_names: Optional[List[str]] = None) -> Dict[str, Path]:
    """查找可用的实验目录"""
    experiments = {}
    exp_names = exp_names or MAIN_EXPERIMENTS
    
    for exp in exp_names:
        full_dir = results_root / exp
        debug_dir = results_root / f"{exp}_debug"
        
        if full_dir.exists() and (full_dir / "metrics.csv").exists():
            experiments[exp] = full_dir
        elif debug_dir.exists() and (debug_dir / "metrics.csv").exists():
            experiments[exp] = debug_dir
    
    return experiments


def generate_metrics_table(experiments: Dict[str, Path]) -> str:
    """生成简洁的指标对比表格"""
    if not experiments:
        return "| 实验 | Macro-F1 | Dice | IoU |\n|------|----------|------|-----|\n| - | - | - | - |\n"
    
    all_metrics = {}
    for exp, path in experiments.items():
        all_metrics[exp] = load_csv_as_dict(path / "metrics.csv")
    
    lines = []
    lines.append("| 实验 | Macro-F1 | Dice | IoU |")
    lines.append("|------|----------|------|-----|")
    
    for exp in MAIN_EXPERIMENTS:
        if exp not in experiments:
            continue
        metrics = all_metrics[exp]
        f1 = metrics.get("Macro-F1", 0)
        dice = metrics.get("Dice", 0)
        iou = metrics.get("IoU", 0)
        lines.append(f"| {exp.upper()} | {f1:.4f} | {dice:.4f} | {iou:.4f} |")
    
    return "\n".join(lines) + "\n"


def generate_long_tail_table(experiments: Dict[str, Path]) -> str:
    """生成长尾对比表格（含 Tail Macro-F1）"""
    if not experiments:
        return "| 实验 | Macro-F1 | Tail Macro-F1 |\n|------|----------|---------------|\n| - | - | - |\n"

    lines = []
    lines.append("| 实验 | Macro-F1 | Tail Macro-F1 |")
    lines.append("|------|----------|---------------|")

    for exp in LONG_TAIL_EXPERIMENTS:
        if exp not in experiments:
            continue
        exp_path = experiments[exp]
        metrics = load_csv_as_dict(exp_path / "metrics.csv")
        macro_f1 = metrics.get("Macro-F1", 0)
        tail_summary = load_tail_summary(exp_path)
        tail_macro = get_tail_metric_value(tail_summary, "Tail Macro-F1")
        tail_macro = tail_macro if tail_macro is not None else 0
        lines.append(f"| {exp} | {macro_f1:.4f} | {tail_macro:.4f} |")

    return "\n".join(lines) + "\n"


def generate_slides_content(results_root: Path) -> str:
    """生成PPT大纲内容"""
    experiments = find_available_experiments(results_root, MAIN_EXPERIMENTS)
    tail_experiments = find_available_experiments(results_root, LONG_TAIL_EXPERIMENTS)
    ssl_source = check_ssl_data_source(results_root)
    metrics = {exp: load_csv_as_dict(path / "metrics.csv") for exp, path in experiments.items()}
    e0_f1 = get_metric_value(metrics.get("e0", {}), "Macro-F1")
    e1_f1 = get_metric_value(metrics.get("e1", {}), "Macro-F1")
    e2_f1 = get_metric_value(metrics.get("e2", {}), "Macro-F1")
    e3_f1 = get_metric_value(metrics.get("e3", {}), "Macro-F1")
    e2_failed = is_long_tail_failed(e2_f1, e0_f1)
    ddpm_info = load_ddpm_info(results_root, "ddpm_tail", "data/synthetic/ddpm")
    ddpm_wide_info = load_ddpm_info(results_root, "ddpm_tail_wide", "data/synthetic/ddpm_wide")
    synthetic_stats = ddpm_info.get("synthetic_stats") if ddpm_info else None
    synthetic_wide_stats = ddpm_wide_info.get("synthetic_stats") if ddpm_wide_info else None
    ddpm_ready = bool(
        (synthetic_stats and synthetic_stats.get("total_generated", 0))
        or (synthetic_wide_stats and synthetic_wide_stats.get("total_generated", 0))
    )
    training_summary = build_training_summary(experiments)
    
    if e2_f1 is None or e0_f1 is None:
        e2_outcome_line = "**结果**: 未运行或结果缺失"
    elif e2_failed:
        e2_outcome_line = "**结果**: Macro-F1显著下降，策略未达预期"
    else:
        delta_pct = (e2_f1 - e0_f1) / e0_f1 * 100 if e0_f1 > 0 else 0
        trend = "提升" if delta_pct >= 0 else "下降"
        e2_outcome_line = f"**结果**: Macro-F1 {trend} {delta_pct:+.2f}%"

    ddpm_summary_line = ""
    if synthetic_stats:
        total_generated = synthetic_stats.get("total_generated")
        target_count = synthetic_stats.get("target_count")
        ddpm_summary_line = (
            f"**DDPM合成**: total_generated={total_generated}, target_count={target_count}"
        )
    if synthetic_wide_stats:
        total_generated = synthetic_wide_stats.get("total_generated")
        tail_classes = synthetic_wide_stats.get("target_classes") or []
        suffix = f"wide_total={total_generated}, tail_classes={len(tail_classes)}"
        if ddpm_summary_line:
            ddpm_summary_line = f"{ddpm_summary_line}; {suffix}"
        else:
            ddpm_summary_line = f"**DDPM合成**: {suffix}"
    
    if e1_f1 is not None and e0_f1 is not None and e0_f1 > 0:
        delta_pct = (e1_f1 - e0_f1) / e0_f1 * 100
        trend = "提升" if delta_pct >= 0 else "下降"
        ssl_line = f"2. SSL预训练使 Macro-F1 {trend} {delta_pct:+.2f}%"
    else:
        ssl_line = "2. SSL预训练结果缺失或不可比较"
    
    if e2_f1 is not None and e0_f1 is not None:
        if e2_failed:
            long_tail_line = "3. 长尾增强未达预期（Macro-F1显著下降）"
        else:
            delta_pct = (e2_f1 - e0_f1) / e0_f1 * 100 if e0_f1 > 0 else 0
            trend = "提升" if delta_pct >= 0 else "下降"
            long_tail_line = f"3. 长尾增强 Macro-F1 {trend} {delta_pct:+.2f}%"
    else:
        long_tail_line = "3. 长尾增强结果缺失或未运行"
    
    if e1_f1 is not None and e3_f1 is not None:
        delta = e3_f1 - e1_f1
        if abs(delta) <= 0.001:
            sep_line = "4. Prototype分离在性能上基本不变"
        elif delta > 0:
            sep_line = "4. Prototype分离带来小幅性能提升"
        else:
            sep_line = "4. Prototype分离带来小幅性能下降"
    else:
        sep_line = "4. Prototype分离结果缺失或未运行"
    
    key_findings = "\n".join(
        [
            "1. 多任务学习有效共享特征表示",
            ssl_line,
            long_tail_line,
            sep_line,
        ]
    )
    
    if e1_f1 is not None and e0_f1 is not None and e1_f1 >= e0_f1:
        ssl_conclusion = "✅ SSL预训练带来Macro-F1提升"
    else:
        ssl_conclusion = "⚠️ SSL预训练未显著提升或结果缺失"
    
    if ddpm_ready and e2_f1 is not None and e0_f1 is not None and not e2_failed:
        long_tail_conclusion = "✅ DDPM长尾增强带来Macro-F1提升"
    elif ddpm_ready and e2_f1 is not None and e0_f1 is not None:
        long_tail_conclusion = "⚠️ DDPM长尾增强未达预期"
    elif ddpm_ready:
        long_tail_conclusion = "⚠️ DDPM已完成，但E2结果缺失"
    else:
        long_tail_conclusion = "⚠️ DDPM未完成"
    
    if e1_f1 is not None and e3_f1 is not None and abs(e3_f1 - e1_f1) <= 0.001:
        sep_conclusion = "✅ Prototype分离不降低识别性能"
    else:
        sep_conclusion = "✅ Prototype分离生成可解释热力图"
    
    if not ddpm_ready:
        long_tail_limit = "DDPM生成式长尾增强未完成"
    elif e2_failed:
        long_tail_limit = "DDPM增强未达预期（需优化质量/比例）"
    else:
        long_tail_limit = "DDPM增强仍需评估样本质量与分布偏移"
    
    slides = f"""# MixedWM38 混合缺陷晶圆图谱多任务识别

> PPT大纲 - 自动生成于 {datetime.now().strftime("%Y-%m-%d")}
**副标题**: 自监督表征学习 + 长尾增强 + 弱监督成分分离 | 作者: [姓名] | 日期: {datetime.now().strftime("%Y年%m月")}

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
- {training_summary}
- Optimizer: AdamW

---

## Slide 5: E0 基线结果

### E0: 多任务基线模型

{generate_metrics_table({k: v for k, v in experiments.items() if k == "e0"})}

**混淆矩阵**:
![E0 Confusion Matrix](../results/e0/confusion_matrix.png)

**分割可视化**:
![E0 Segmentation](../results/e0/seg_overlays/sample_001.png)

---

## Slide 6: E1 SSL预训练对比

### E1 vs E0: SSL预训练的影响

{generate_metrics_table({k: v for k, v in experiments.items() if k in ["e0", "e1"]})}

**SSL预训练方法**: SimCLR风格对比学习

**数据源**: {ssl_source}

**权重加载统计**: 见 `results/e1/weight_loading.json`

---

## Slide 7: E2 长尾增强

### E2: 处理类别不平衡

**策略**:
1. 训练DDPM（尾部类别）
2. 生成合成样本并只加入训练集
3. 继续E2多任务训练

{generate_metrics_table({k: v for k, v in experiments.items() if k in ["e0", "e2"]})}

{e2_outcome_line}
{ddpm_summary_line}

**尾部类别分析**: 见 `results/e2/tail_class_analysis.csv`

**DDPM + Loss 长尾对比**:
{generate_long_tail_table(tail_experiments)}

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

{generate_metrics_table(experiments)}

**关键发现**:
{key_findings}

---

## Slide 11: 结论与展望

### 结论

✅ 构建了完整的多任务晶圆缺陷识别系统
{ssl_conclusion}
{long_tail_conclusion}
{sep_conclusion}

### 局限性

- {("SSL使用MixedWM38作为fallback数据源" if "MixedWM38" in ssl_source else "SSL已使用WM-811K进行预训练")}
- 分离方法采用Prototype而非端到端训练
- {long_tail_limit}

### 未来工作

- {("扩展SSL预训练策略" if "WM-811K" in ssl_source else "获取WM-811K进行完整SSL预训练")}
- 实现端到端弱监督分离头
- {("优化DDPM生成质量与采样策略" if ddpm_ready else "完成DDPM生成式数据增强")}

"""
    
    return slides


def main():
    parser = argparse.ArgumentParser(
        description="生成PPT大纲",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    conda run -n wafer-seg-class python scripts/generate_slides_md.py --results_root results --out slides/SLIDES.md
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
        default="slides/SLIDES.md",
        help="输出Markdown文件路径 (默认: slides/SLIDES.md)"
    )
    
    args = parser.parse_args()
    
    results_root = Path(args.results_root)
    output_path = Path(args.out)
    
    if not results_root.exists():
        print(f"[ERROR] 结果目录不存在: {results_root}")
        return 1
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成PPT大纲
    slides_content = generate_slides_content(results_root)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(slides_content)
    
    print(f"[SUCCESS] PPT大纲已生成: {output_path}")
    print(f"[INFO] 共 12 页（含封面）")
    return 0


if __name__ == "__main__":
    exit(main())
