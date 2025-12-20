#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MixedWM38 数据准备脚本

基于现有 make_dataset.py 改造，支持 --debug 模式
输出数据统计到日志

Usage:
    conda run -n wafer-seg-class python scripts/prepare_mixedwm38.py --input data/raw/Wafer_Map_Datasets.npz --output data/processed
    conda run -n wafer-seg-class python scripts/prepare_mixedwm38.py --input data/raw/Wafer_Map_Datasets.npz --output data/processed --debug
"""

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 38类标签映射（从 config.py 复制）
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

IMAGE_SIZE = (224, 224)


def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "prepare_data.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def prepare_data(
    input_file: str,
    output_dir: str,
    debug: bool = False,
    max_per_class: int = 5,
):
    """
    准备 MixedWM38 数据集
    
    Args:
        input_file: 输入 npz 文件路径
        output_dir: 输出目录
        debug: 是否为 debug 模式（每类最多 max_per_class 样本）
        max_per_class: debug 模式下每类最大样本数
    """
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    images_dir = os.path.join(output_dir, "Images")
    labels_dir = os.path.join(output_dir, "Labels")
    masks_dir = os.path.join(output_dir, "Masks")
    
    for d in [images_dir, labels_dir, masks_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 加载数据
    logger.info(f"[Info] Loading data from {input_file}")
    data = np.load(input_file)
    images = data["arr_0"]
    labels = data["arr_1"]
    
    logger.info(f"[Info] Total samples: {len(images)}")
    logger.info(f"[Info] Image shape: {images[0].shape}")
    logger.info(f"[Info] Label shape: {labels[0].shape}")
    
    # 统计标签分布
    label_counts = Counter()
    for lbl in labels:
        label_str = str(lbl)
        if label_str in CLASS_MAPPING:
            label_counts[CLASS_MAPPING[label_str]] += 1
    
    logger.info("[Info] Label distribution:")
    for cls_id in sorted(label_counts.keys()):
        logger.info(f"  Class {cls_id} ({CLASS_NAME_MAPPING[cls_id]}): {label_counts[cls_id]}")
    
    # Debug 模式：每类最多 max_per_class 样本
    if debug:
        logger.info(f"[Info] Debug mode: max {max_per_class} samples per class")
        class_counts = {i: 0 for i in range(38)}
        selected_indices = []
        
        for i, lbl in enumerate(labels):
            label_str = str(lbl)
            if label_str in CLASS_MAPPING:
                cls_id = CLASS_MAPPING[label_str]
                if class_counts[cls_id] < max_per_class:
                    selected_indices.append(i)
                    class_counts[cls_id] += 1
        
        logger.info(f"[Info] Selected {len(selected_indices)} samples for debug")
    else:
        selected_indices = list(range(len(images)))
    
    # 处理图像
    logger.info("[Info] Processing images...")
    image_colors = np.array([[255, 0, 255], [0, 255, 255], [255, 255, 0]])
    mask_colors = np.array([[0], [0], [255]])
    
    processed_count = 0
    for idx in tqdm(selected_indices, desc="Processing"):
        inp_image = images[idx].astype(np.uint8)
        
        # 处理异常像素值：将3映射到2（缺陷）
        inp_image = np.clip(inp_image, 0, 2)
        
        inp_image = cv2.resize(inp_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        
        # 生成 RGB 图像
        rgb_image = image_colors[inp_image]
        
        # 生成 mask
        rgb_mask = mask_colors[inp_image]
        
        # 保存
        np.save(os.path.join(images_dir, f"Image_{processed_count}.npy"), rgb_image)
        np.save(os.path.join(labels_dir, f"Image_{processed_count}.npy"), labels[idx])
        np.save(os.path.join(masks_dir, f"Image_{processed_count}.npy"), rgb_mask)
        
        processed_count += 1
    
    # 输出统计
    logger.info(f"[Info] Processed {processed_count} samples")
    logger.info(f"[Info] Images saved to: {images_dir}")
    logger.info(f"[Info] Labels saved to: {labels_dir}")
    logger.info(f"[Info] Masks saved to: {masks_dir}")
    
    # 保存统计信息
    stats = {
        "total_samples": processed_count,
        "debug_mode": debug,
        "image_size": IMAGE_SIZE,
        "label_distribution": {CLASS_NAME_MAPPING[k]: v for k, v in label_counts.items()},
    }
    
    stats_file = os.path.join(output_dir, "data_stats.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("MixedWM38 Data Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total samples: {processed_count}\n")
        f.write(f"Debug mode: {debug}\n")
        f.write(f"Image size: {IMAGE_SIZE}\n")
        f.write("\nLabel distribution:\n")
        for cls_id in sorted(label_counts.keys()):
            f.write(f"  {CLASS_NAME_MAPPING[cls_id]}: {label_counts[cls_id]}\n")
    
    logger.info(f"[Info] Statistics saved to: {stats_file}")
    logger.info("[Info] Data preparation completed!")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare MixedWM38 dataset")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/raw/Wafer_Map_Datasets.npz",
        help="Input npz file path",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/processed",
        help="Output directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: use only a few samples per class",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=5,
        help="Max samples per class in debug mode",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory",
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_dir)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"[Error] Input file not found: {args.input}")
        logger.info("[Info] Please download MixedWM38 dataset and place it at the specified path")
        sys.exit(1)
    
    # 准备数据
    prepare_data(
        input_file=args.input,
        output_dir=args.output,
        debug=args.debug,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
