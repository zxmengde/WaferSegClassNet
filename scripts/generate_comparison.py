#!/usr/bin/env python
"""
实验对比表生成脚本

汇总E0/E1/E2/E3的metrics.csv，计算delta，输出comparison.csv

用法:
    python scripts/generate_comparison.py --results_root results --out results/comparison.csv
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_metrics_csv(filepath: Path) -> Dict[str, float]:
    """
    解析metrics.csv文件，提取主要指标
    
    Args:
        filepath: metrics.csv文件路径
        
    Returns:
        包含指标名称和值的字典
    """
    metrics = {}
    
    if not filepath.exists():
        return metrics
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        in_metric_section = True
        
        for row in reader:
            if len(row) < 2:
                continue
            
            # 检测是否进入per-class部分
            if row[0] == 'Class':
                in_metric_section = False
                continue
            
            if in_metric_section:
                metric_name = row[0].strip()
                try:
                    value = float(row[1])
                    metrics[metric_name] = value
                except (ValueError, IndexError):
                    continue
    
    return metrics


def calculate_delta(current: float, baseline: float) -> Tuple[float, str]:
    """
    计算相对于基线的delta值
    
    Args:
        current: 当前值
        baseline: 基线值
        
    Returns:
        (delta绝对值, delta百分比字符串)
    """
    delta = current - baseline
    
    if baseline == 0:
        if current == 0:
            pct_str = "+0.0%"
        else:
            pct_str = "+∞%"
    else:
        pct = (delta / baseline) * 100
        sign = "+" if pct >= 0 else ""
        pct_str = f"{sign}{pct:.1f}%"
    
    sign = "+" if delta >= 0 else ""
    return delta, f"{sign}{delta:.4f} ({pct_str})"


def find_experiment_dirs(results_root: Path, experiments: List[str]) -> Dict[str, Optional[Path]]:
    """
    查找实验目录
    
    优先查找完整实验目录（如e0），如果不存在则查找debug版本（如e0_debug）
    
    Args:
        results_root: 结果根目录
        experiments: 实验名称列表
        
    Returns:
        实验名称到目录路径的映射
    """
    exp_dirs = {}
    
    for exp in experiments:
        # 优先查找完整实验
        full_dir = results_root / exp
        debug_dir = results_root / f"{exp}_debug"
        
        if full_dir.exists() and (full_dir / "metrics.csv").exists():
            exp_dirs[exp] = full_dir
        elif debug_dir.exists() and (debug_dir / "metrics.csv").exists():
            exp_dirs[exp] = debug_dir
            print(f"[INFO] 使用debug版本: {debug_dir.name}")
        else:
            exp_dirs[exp] = None
            print(f"[WARN] 未找到实验 {exp} 的metrics.csv")
    
    return exp_dirs


def generate_comparison_table(
    results_root: Path,
    output_path: Path,
    experiments: List[str] = None,
    baseline: str = "e0"
) -> bool:
    """
    生成实验对比表
    
    Args:
        results_root: 结果根目录
        output_path: 输出CSV路径
        experiments: 要对比的实验列表，默认为["e0", "e1", "e2", "e3"]
        baseline: 基线实验名称
        
    Returns:
        是否成功生成
    """
    if experiments is None:
        experiments = ["e0", "e1", "e2", "e3"]
    
    # 查找实验目录
    exp_dirs = find_experiment_dirs(results_root, experiments)
    
    # 检查是否有可用的实验
    available_exps = [exp for exp, path in exp_dirs.items() if path is not None]
    if not available_exps:
        print("[ERROR] 没有找到任何可用的实验结果")
        return False
    
    # 读取所有实验的指标
    all_metrics = {}
    for exp, path in exp_dirs.items():
        if path is not None:
            metrics_file = path / "metrics.csv"
            all_metrics[exp] = parse_metrics_csv(metrics_file)
            # 记录实际使用的目录名
            all_metrics[exp]['_source_dir'] = path.name
        else:
            all_metrics[exp] = {}
    
    # 获取基线指标
    baseline_metrics = all_metrics.get(baseline, {})
    
    # 定义要输出的主要指标
    main_metrics = ["Accuracy", "Macro-F1", "Micro-F1", "Weighted-F1", "Dice", "IoU"]
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成对比表
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入标题行
        header = ["Experiment", "Source"]
        for metric in main_metrics:
            header.append(metric)
            if baseline in available_exps:
                header.append(f"{metric}_Delta")
        writer.writerow(header)
        
        # 写入每个实验的数据
        for exp in experiments:
            metrics = all_metrics.get(exp, {})
            source_dir = metrics.get('_source_dir', 'N/A')
            
            row = [exp, source_dir]
            
            for metric in main_metrics:
                value = metrics.get(metric, None)
                
                if value is not None:
                    row.append(f"{value:.4f}")
                    
                    # 计算delta（如果有基线）
                    if baseline in available_exps:
                        baseline_value = baseline_metrics.get(metric, 0)
                        if exp == baseline:
                            row.append("-")
                        else:
                            _, delta_str = calculate_delta(value, baseline_value)
                            row.append(delta_str)
                else:
                    row.append("N/A")
                    if baseline in available_exps:
                        row.append("N/A")
            
            writer.writerow(row)
        
        # 写入空行和汇总信息
        writer.writerow([])
        writer.writerow(["Summary"])
        writer.writerow(["Total experiments found", len(available_exps)])
        writer.writerow(["Baseline", baseline])
        writer.writerow(["Missing experiments", ", ".join([exp for exp in experiments if exp not in available_exps]) or "None"])
    
    print(f"[SUCCESS] 对比表已生成: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="生成实验对比表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/generate_comparison.py --results_root results --out results/comparison.csv
    python scripts/generate_comparison.py --results_root results --experiments e0 e1 --baseline e0
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
        default="results/comparison.csv",
        help="输出CSV文件路径 (默认: results/comparison.csv)"
    )
    
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["e0", "e1", "e2", "e3"],
        help="要对比的实验列表 (默认: e0 e1 e2 e3)"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        default="e0",
        help="基线实验名称 (默认: e0)"
    )
    
    args = parser.parse_args()
    
    results_root = Path(args.results_root)
    output_path = Path(args.out)
    
    if not results_root.exists():
        print(f"[ERROR] 结果目录不存在: {results_root}")
        return 1
    
    success = generate_comparison_table(
        results_root=results_root,
        output_path=output_path,
        experiments=args.experiments,
        baseline=args.baseline
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
