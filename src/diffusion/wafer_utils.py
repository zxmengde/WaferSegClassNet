# -*- coding: utf-8 -*-
"""
DDPM 数据辅助工具
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.mappings import CLASS_MAPPING, LABEL_38_TO_8


RGB_PALETTE = np.array(
    [
        [255, 0, 255],   # background
        [0, 255, 255],   # wafer
        [255, 255, 0],   # defect
    ],
    dtype=np.uint8,
)


def rgb_to_index_map(rgb_image: np.ndarray) -> np.ndarray:
    """
    将 RGB 晶圆图转换为 0/1/2 索引图
    """
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    palette = RGB_PALETTE.reshape(1, 1, 3, 3)
    diff = (rgb_image[:, :, None, :] - palette) ** 2
    dist = diff.sum(axis=-1)
    return dist.argmin(axis=2).astype(np.int64)


def index_map_to_rgb(index_map: np.ndarray) -> np.ndarray:
    """
    将 0/1/2 索引图转换为 RGB 晶圆图
    """
    index_map = np.clip(index_map, 0, 2).astype(np.int64)
    return RGB_PALETTE[index_map]


def index_map_to_mask(index_map: np.ndarray) -> np.ndarray:
    """
    将索引图转换为二值 mask（缺陷=255）
    """
    mask = (index_map == 2).astype(np.uint8) * 255
    return mask[:, :, None]


def get_label_vector(label_38: int) -> np.ndarray:
    """
    将 38 类标签转为 8 维多标签向量
    """
    return np.array(LABEL_38_TO_8[label_38], dtype=np.int64)


def load_label_counts(labels_dir: Path) -> Dict[int, int]:
    """
    统计 labels 目录中的 38 类分布
    """
    counts: Dict[int, int] = {}
    label_files = sorted([p for p in labels_dir.iterdir() if p.suffix == ".npy"])
    for lbl_path in label_files:
        label_raw = np.load(lbl_path)
        label_str = str(label_raw)
        if label_str not in CLASS_MAPPING:
            continue
        label_38 = CLASS_MAPPING[label_str]
        counts[label_38] = counts.get(label_38, 0) + 1
    return counts


def get_tail_classes(
    counts: Dict[int, int],
    threshold: int,
) -> List[int]:
    """
    根据阈值选择尾部类别
    """
    return [k for k, v in counts.items() if 0 < v < threshold]
