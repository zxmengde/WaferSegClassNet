#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据完整性检查脚本

检查图像/标签/mask文件数量一致性和标签分布

Usage:
    conda run -n wafer-seg-class python scripts/sanity_check_data.py --data_root data/processed
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# 38类标签映射
CLASS_MAPPING = {
    "[0 0 0 0 0 0 0 0]": 0,
    "[1 0 0 0 0 0 0 0]": 1,
    "[0 1 0 0 0 0 0 0]": 2,
    "[0 0 1 0 0 0 0 0]": 3,
    "[0 0 0 1 0 0 0 0]": 4,
    "[0 0 0 0 1 0 0 0]": 5,
    "[0 0 0 0 0 1 0 0]": 6,
    "[0 0 0 0 0 0 1 0]": 7,
    "[0 0 0 0 0 0 0 1]": 8,
    "[1 0 1 0 0 0 0 0]": 9,
    "[1 0 0 1 0 0 0 0]": 10,
    "[1 0 0 0 1 0 0 0]": 11,
    "[1 0 0 0 0 0 1 0]": 12,
    "[0 1 1 0 0 0 0 0]": 13,
    "[0 1 0 1 0 0 0 0]": 14,
    "[0 1 0 0 1 0 0 0]": 15,
    "[0 1 0 0 0 0 1 0]": 16,
    "[0 0 1 0 1 0 0 0]": 17,
    "[0 0 1 0 0 0 1 0]": 18,
    "[0 0 0 1 1 0 0 0]": 19,
    "[0 0 0 1 0 0 1 0]": 20,
    "[0 0 0 0 1 0 1 0]": 21,
    "[1 0 1 0 1 0 0 0]": 22,
    "[1 0 1 0 0 0 1 0]": 23,
    "[1 0 0 1 1 0 0 0]": 24,
    "[1 0 0 1 0 0 1 0]": 25,
    "[1 0 0 0 1 0 1 0]": 26,
    "[0 1 1 0 1 0 0 0]": 27,
    "[0 1 1 0 0 0 1 0]": 28,
    "[0 1 0 1 1 0 0 0]": 29,
    "[0 1 0 1 0 0 1 0]": 30,
    "[0 1 0 0 1 0 1 0]": 31,
    "[0 0 1 0 1 0 1 0]": 32,
    "[0 0 0 1 1 0 1 0]": 33,
    "[1 0 1 0 1 0 1 0]": 34,
    "[1 0 0 1 1 0 1 0]": 35,
    "[0 1 1 0 1 0 1 0]": 36,
    "[0 1 0 1 1 0 1 0]": 37,
}

CLASS_NAME_MAPPING = {
    0: "Normal", 1: "Center", 2: "Donut", 3: "EL", 4: "ER",
    5: "LOC", 6: "NF", 7: "S", 8: "Random",
    9: "C+EL", 10: "C+ER", 11: "C+L", 12: "C+S",
    13: "D+EL", 14: "D+ER", 15: "D+L", 16: "D+S",
    17: "EL+L", 18: "EL+S", 19: "ER+L", 20: "ER+S", 21: "L+S",
    22: "C+EL+L", 23: "C+EL+S", 24: "C+ER+L", 25: "C+ER+S", 26: "C+L+S",
    27: "D+EL+L", 28: "D+EL+S", 29: "D+ER+L", 30: "D+ER+S", 31: "D+L+S",
    32: "EL+L+S", 33: "ER+L+S",
    34: "C+L+EL+S", 35: "C+L+ER+S", 36: "D+L+EL+S", 37: "D+L+ER+S",
}


def check_data(data_root: str) -> dict:
    """
    检查数据完整性
    
    Returns:
        dict: 检查结果
    """
    results = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    images_dir = os.path.join(data_root, "Images")
    labels_dir = os.path.join(data_root, "Labels")
    masks_dir = os.path.join(data_root, "Masks")
    
    # 检查目录存在
    for dir_path, dir_name in [(images_dir, "Images"), (labels_dir, "Labels"), (masks_dir, "Masks")]:
        if not os.path.exists(dir_path):
            results["errors"].append(f"Directory not found: {dir_path}")
            results["passed"] = False
    
    if not results["passed"]:
        return results
    
    # 获取文件列表
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".npy")])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".npy")])
    
    results["stats"]["num_images"] = len(image_files)
    results["stats"]["num_labels"] = len(label_files)
    results["stats"]["num_masks"] = len(mask_files)
    
    print(f"[Info] Found {len(image_files)} images, {len(label_files)} labels, {len(mask_files)} masks")
    
    # 检查数量一致性
    if len(image_files) != len(label_files):
        results["errors"].append(f"Image count ({len(image_files)}) != Label count ({len(label_files)})")
        results["passed"] = False
    
    if len(image_files) != len(mask_files):
        results["errors"].append(f"Image count ({len(image_files)}) != Mask count ({len(mask_files)})")
        results["passed"] = False
    
    # 检查文件名匹配
    image_names = set(image_files)
    label_names = set(label_files)
    mask_names = set(mask_files)
    
    missing_labels = image_names - label_names
    missing_masks = image_names - mask_names
    
    if missing_labels:
        results["warnings"].append(f"Missing labels for {len(missing_labels)} images")
    if missing_masks:
        results["warnings"].append(f"Missing masks for {len(missing_masks)} images")
    
    # 检查数据格式和标签分布
    label_counts = Counter()
    invalid_labels = []
    
    print("[Info] Checking data format and label distribution...")
    
    for i, (img_file, lbl_file, msk_file) in enumerate(zip(image_files[:min(100, len(image_files))], 
                                                           label_files[:min(100, len(label_files))],
                                                           mask_files[:min(100, len(mask_files))])):
        # 检查图像格式
        img = np.load(os.path.join(images_dir, img_file))
        if img.shape != (224, 224, 3):
            results["warnings"].append(f"Unexpected image shape: {img.shape} for {img_file}")
        
        # 检查标签格式
        lbl = np.load(os.path.join(labels_dir, lbl_file))
        lbl_str = str(lbl)
        if lbl_str in CLASS_MAPPING:
            label_counts[CLASS_MAPPING[lbl_str]] += 1
        else:
            invalid_labels.append(lbl_str)
        
        # 检查 mask 格式
        msk = np.load(os.path.join(masks_dir, msk_file))
        if msk.shape != (224, 224, 1):
            results["warnings"].append(f"Unexpected mask shape: {msk.shape} for {msk_file}")
        
        # 检查 mask 是否为二值
        unique_values = np.unique(msk)
        if not all(v in [0, 255] for v in unique_values):
            results["warnings"].append(f"Non-binary mask values: {unique_values} for {msk_file}")
    
    if invalid_labels:
        results["warnings"].append(f"Found {len(invalid_labels)} invalid labels")
    
    # 统计完整标签分布（如果文件数量合理）
    if len(label_files) <= 50000:
        print("[Info] Computing full label distribution...")
        full_label_counts = Counter()
        for lbl_file in label_files:
            lbl = np.load(os.path.join(labels_dir, lbl_file))
            lbl_str = str(lbl)
            if lbl_str in CLASS_MAPPING:
                full_label_counts[CLASS_MAPPING[lbl_str]] += 1
        
        results["stats"]["label_distribution"] = dict(full_label_counts)
        
        # 检查是否所有类别都有样本
        missing_classes = set(range(38)) - set(full_label_counts.keys())
        if missing_classes:
            results["warnings"].append(f"Missing classes: {sorted(missing_classes)}")
    
    return results


def print_report(results: dict):
    """打印检查报告"""
    print("\n" + "=" * 60)
    print("Data Sanity Check Report")
    print("=" * 60)
    
    # 统计信息
    print("\n[Statistics]")
    for key, value in results["stats"].items():
        if key == "label_distribution":
            print(f"  Label distribution:")
            for cls_id in sorted(value.keys()):
                cls_name = CLASS_NAME_MAPPING.get(cls_id, f"Unknown_{cls_id}")
                print(f"    {cls_id:2d} ({cls_name:12s}): {value[cls_id]:5d}")
        else:
            print(f"  {key}: {value}")
    
    # 错误
    if results["errors"]:
        print("\n[ERRORS]")
        for err in results["errors"]:
            print(f"  ❌ {err}")
    
    # 警告
    if results["warnings"]:
        print("\n[WARNINGS]")
        for warn in results["warnings"]:
            print(f"  ⚠️ {warn}")
    
    # 总结
    print("\n" + "-" * 60)
    if results["passed"]:
        print("✅ Data sanity check PASSED")
    else:
        print("❌ Data sanity check FAILED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Check data integrity")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/processed",
        help="Data root directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output report file (optional)",
    )
    
    args = parser.parse_args()
    
    # 检查数据
    results = check_data(args.data_root)
    
    # 打印报告
    print_report(results)
    
    # 保存报告
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            import json
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[Info] Report saved to: {args.output}")
    
    # 返回状态码
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
