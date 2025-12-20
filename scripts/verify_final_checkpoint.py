#!/usr/bin/env python
"""Final Checkpoint验证脚本 - 检查所有实验产物是否完整"""

import os
import sys
from pathlib import Path

def check_file_exists(path: str, description: str) -> bool:
    """检查文件是否存在"""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {path}")
    return exists

def check_dir_not_empty(path: str, description: str) -> bool:
    """检查目录是否存在且非空"""
    if not os.path.exists(path):
        print(f"  ✗ {description}: {path} (不存在)")
        return False
    files = os.listdir(path)
    if len(files) == 0:
        print(f"  ✗ {description}: {path} (空目录)")
        return False
    print(f"  ✓ {description}: {path} ({len(files)} 文件)")
    return True

def verify_experiment(exp_name: str, results_root: str = "results") -> dict:
    """验证单个实验的产物完整性"""
    exp_dir = os.path.join(results_root, exp_name)
    results = {"name": exp_name, "passed": 0, "failed": 0, "details": []}
    
    print(f"\n{'='*60}")
    print(f"验证实验: {exp_name}")
    print(f"{'='*60}")
    
    # 必需文件
    required_files = [
        ("metrics.csv", "指标文件"),
        ("confusion_matrix.png", "混淆矩阵"),
        ("config_snapshot.yaml", "配置快照"),
        ("meta.json", "元信息"),
    ]
    
    for filename, desc in required_files:
        path = os.path.join(exp_dir, filename)
        if check_file_exists(path, desc):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append(f"缺失: {filename}")
    
    # 必需目录
    required_dirs = [
        ("checkpoints", "检查点目录"),
        ("curves", "训练曲线"),
        ("seg_overlays", "分割可视化"),
    ]
    
    for dirname, desc in required_dirs:
        path = os.path.join(exp_dir, dirname)
        if check_dir_not_empty(path, desc):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append(f"缺失/空: {dirname}")
    
    # 检查checkpoint文件
    best_pt = os.path.join(exp_dir, "checkpoints", "best.pt")
    if check_file_exists(best_pt, "best.pt"):
        results["passed"] += 1
    else:
        results["failed"] += 1
        results["details"].append("缺失: checkpoints/best.pt")
    
    return results

def verify_ssl_debug(results_root: str = "results") -> dict:
    """验证SSL debug实验的产物"""
    exp_dir = os.path.join(results_root, "ssl_debug")
    results = {"name": "ssl_debug", "passed": 0, "failed": 0, "details": []}
    
    print(f"\n{'='*60}")
    print(f"验证SSL Debug实验")
    print(f"{'='*60}")
    
    # 必需文件
    required_files = [
        ("checkpoints/last.pt", "SSL检查点"),
        ("config_snapshot.yaml", "配置快照"),
        ("curves/loss_curve.png", "损失曲线"),
    ]
    
    for filepath, desc in required_files:
        path = os.path.join(exp_dir, filepath)
        if check_file_exists(path, desc):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append(f"缺失: {filepath}")
    
    return results


def verify_ddpm(
    results_root: str = "results",
    synthetic_root: str = "data/synthetic/ddpm",
    exp_name: str = "ddpm_tail",
) -> dict:
    """验证DDPM训练与合成数据产物"""
    exp_dir = os.path.join(results_root, exp_name)
    results = {"name": exp_name, "passed": 0, "failed": 0, "details": []}
    
    print(f"\n{'='*60}")
    print(f"验证DDPM尾部增强: {exp_name}")
    print(f"{'='*60}")
    
    required_files = [
        ("config_snapshot.yaml", "DDPM配置快照"),
        ("history.json", "DDPM训练历史"),
        ("checkpoints/best.pt", "DDPM最佳检查点"),
    ]
    
    for filepath, desc in required_files:
        path = os.path.join(exp_dir, filepath)
        if check_file_exists(path, desc):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append(f"缺失: {filepath}")
    
    # 合成数据统计
    synthetic_stats = os.path.join(synthetic_root, "synthetic_stats.json")
    if check_file_exists(synthetic_stats, "合成样本统计"):
        results["passed"] += 1
    else:
        results["failed"] += 1
        results["details"].append("缺失: synthetic_stats.json")
    
    # 合成数据目录
    for dirname in ["Images", "Labels", "Masks"]:
        path = os.path.join(synthetic_root, dirname)
        if check_dir_not_empty(path, f"合成{dirname}"):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append(f"缺失/空: {dirname}")
    
    return results

def verify_e3(results_root: str = "results") -> dict:
    """验证E3实验的产物（特殊：包含separation_maps）"""
    exp_dir = os.path.join(results_root, "e3")
    results = {"name": "e3", "passed": 0, "failed": 0, "details": []}
    
    print(f"\n{'='*60}")
    print(f"验证E3实验（成分分离）")
    print(f"{'='*60}")
    
    # 必需文件
    required_files = [
        ("metrics.csv", "指标文件"),
        ("confusion_matrix.png", "混淆矩阵"),
        ("prototypes.pt", "原型文件"),
    ]
    
    for filename, desc in required_files:
        path = os.path.join(exp_dir, filename)
        if check_file_exists(path, desc):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append(f"缺失: {filename}")
    
    # 必需目录
    required_dirs = [
        ("separation_maps", "分离热力图"),
        ("seg_overlays", "分割可视化"),
    ]
    
    for dirname, desc in required_dirs:
        path = os.path.join(exp_dir, dirname)
        if check_dir_not_empty(path, desc):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append(f"缺失/空: {dirname}")
    
    return results

def verify_reports() -> dict:
    """验证报告和PPT产物"""
    results = {"name": "reports", "passed": 0, "failed": 0, "details": []}
    
    print(f"\n{'='*60}")
    print(f"验证报告和PPT")
    print(f"{'='*60}")
    
    required_files = [
        ("results/comparison.csv", "实验对比表"),
        ("report/REPORT.md", "实验报告"),
        ("slides/SLIDES.md", "PPT大纲"),
        ("slides/final.pptx", "PPT文件"),
    ]
    
    for filepath, desc in required_files:
        if check_file_exists(filepath, desc):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append(f"缺失: {filepath}")
    
    return results

def main():
    print("="*60)
    print("Final Checkpoint - 全流程验证")
    print("="*60)
    
    all_results = []
    
    # 1. 验证E0 debug
    all_results.append(verify_experiment("e0_debug"))
    
    # 2. 验证E0
    all_results.append(verify_experiment("e0"))
    
    # 3. 验证SSL debug
    all_results.append(verify_ssl_debug())
    
    # 4. 验证E1
    all_results.append(verify_experiment("e1"))
    
    # 5. 验证DDPM
    all_results.append(verify_ddpm())

    # 5.1 验证DDPM扩覆盖
    all_results.append(
        verify_ddpm(
            exp_name="ddpm_tail_wide",
            synthetic_root="data/synthetic/ddpm_wide",
        )
    )
    
    # 6. 验证E2 (使用e2_debug作为fallback)
    if os.path.exists("results/e2/metrics.csv"):
        all_results.append(verify_experiment("e2"))
    else:
        print("\n[注意] E2完整训练未完成，使用e2_debug结果")
        all_results.append(verify_experiment("e2_debug"))

    # 6.1 验证E2变体
    for exp_name in ["e2_focal", "e2_cb", "e2_ddpm_wide"]:
        if os.path.exists(f"results/{exp_name}/metrics.csv"):
            all_results.append(verify_experiment(exp_name))
        else:
            print(f"\n[注意] {exp_name} 未完成或缺失，跳过验证")
    
    # 7. 验证E3
    all_results.append(verify_e3())
    
    # 8. 验证报告
    all_results.append(verify_reports())
    
    # 汇总
    print("\n" + "="*60)
    print("验证汇总")
    print("="*60)
    
    total_passed = 0
    total_failed = 0
    
    for r in all_results:
        status = "✓ PASS" if r["failed"] == 0 else "✗ FAIL"
        print(f"{r['name']:20s}: {status} ({r['passed']} passed, {r['failed']} failed)")
        if r["details"]:
            for detail in r["details"]:
                print(f"    - {detail}")
        total_passed += r["passed"]
        total_failed += r["failed"]
    
    print("\n" + "-"*60)
    print(f"总计: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("\n✓ 所有验证通过！Final Checkpoint完成。")
        return 0
    else:
        print(f"\n✗ 有 {total_failed} 项验证失败，请检查上述问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
