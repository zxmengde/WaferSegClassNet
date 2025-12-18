# -*- coding: utf-8 -*-
"""
标签映射模块

定义 38 类到 8 类的标签映射（权威来源）
"""

import numpy as np
from typing import Dict, List, Tuple

# 8 类基础缺陷名称
DEFECT_NAMES_8 = [
    "Center",   # 0
    "Donut",    # 1
    "EL",       # 2 (Edge-Loc)
    "ER",       # 3 (Edge-Ring)
    "LOC",      # 4
    "NF",       # 5 (Near-Full)
    "S",        # 6 (Scratch)
    "Random",   # 7
]

# 38 类标签字符串到 ID 的映射
CLASS_MAPPING = {
    "[0 0 0 0 0 0 0 0]": 0,   # Normal
    "[1 0 0 0 0 0 0 0]": 1,   # Center
    "[0 1 0 0 0 0 0 0]": 2,   # Donut
    "[0 0 1 0 0 0 0 0]": 3,   # EL
    "[0 0 0 1 0 0 0 0]": 4,   # ER
    "[0 0 0 0 1 0 0 0]": 5,   # LOC
    "[0 0 0 0 0 1 0 0]": 6,   # NF
    "[0 0 0 0 0 0 1 0]": 7,   # S
    "[0 0 0 0 0 0 0 1]": 8,   # Random
    "[1 0 1 0 0 0 0 0]": 9,   # C+EL
    "[1 0 0 1 0 0 0 0]": 10,  # C+ER
    "[1 0 0 0 1 0 0 0]": 11,  # C+L
    "[1 0 0 0 0 0 1 0]": 12,  # C+S
    "[0 1 1 0 0 0 0 0]": 13,  # D+EL
    "[0 1 0 1 0 0 0 0]": 14,  # D+ER
    "[0 1 0 0 1 0 0 0]": 15,  # D+L
    "[0 1 0 0 0 0 1 0]": 16,  # D+S
    "[0 0 1 0 1 0 0 0]": 17,  # EL+L
    "[0 0 1 0 0 0 1 0]": 18,  # EL+S
    "[0 0 0 1 1 0 0 0]": 19,  # ER+L
    "[0 0 0 1 0 0 1 0]": 20,  # ER+S
    "[0 0 0 0 1 0 1 0]": 21,  # L+S
    "[1 0 1 0 1 0 0 0]": 22,  # C+EL+L
    "[1 0 1 0 0 0 1 0]": 23,  # C+EL+S
    "[1 0 0 1 1 0 0 0]": 24,  # C+ER+L
    "[1 0 0 1 0 0 1 0]": 25,  # C+ER+S
    "[1 0 0 0 1 0 1 0]": 26,  # C+L+S
    "[0 1 1 0 1 0 0 0]": 27,  # D+EL+L
    "[0 1 1 0 0 0 1 0]": 28,  # D+EL+S
    "[0 1 0 1 1 0 0 0]": 29,  # D+ER+L
    "[0 1 0 1 0 0 1 0]": 30,  # D+ER+S
    "[0 1 0 0 1 0 1 0]": 31,  # D+L+S
    "[0 0 1 0 1 0 1 0]": 32,  # EL+L+S
    "[0 0 0 1 1 0 1 0]": 33,  # ER+L+S
    "[1 0 1 0 1 0 1 0]": 34,  # C+L+EL+S
    "[1 0 0 1 1 0 1 0]": 35,  # C+L+ER+S
    "[0 1 1 0 1 0 1 0]": 36,  # D+L+EL+S
    "[0 1 0 1 1 0 1 0]": 37,  # D+L+ER+S
}

# 38 类 ID 到名称的映射
CLASS_NAME_MAPPING = {
    0: "Normal",
    1: "Center", 2: "Donut", 3: "EL", 4: "ER",
    5: "LOC", 6: "NF", 7: "S", 8: "Random",
    9: "C+EL", 10: "C+ER", 11: "C+L", 12: "C+S",
    13: "D+EL", 14: "D+ER", 15: "D+L", 16: "D+S",
    17: "EL+L", 18: "EL+S", 19: "ER+L", 20: "ER+S", 21: "L+S",
    22: "C+EL+L", 23: "C+EL+S", 24: "C+ER+L", 25: "C+ER+S", 26: "C+L+S",
    27: "D+EL+L", 28: "D+EL+S", 29: "D+ER+L", 30: "D+ER+S", 31: "D+L+S",
    32: "EL+L+S", 33: "ER+L+S",
    34: "C+L+EL+S", 35: "C+L+ER+S", 36: "D+L+EL+S", 37: "D+L+ER+S",
}

# 38 类到 8 类多标签的映射
# 每个 38 类标签对应一个 8 维二进制向量
LABEL_38_TO_8: Dict[int, List[int]] = {
    0:  [0, 0, 0, 0, 0, 0, 0, 0],  # Normal
    1:  [1, 0, 0, 0, 0, 0, 0, 0],  # Center
    2:  [0, 1, 0, 0, 0, 0, 0, 0],  # Donut
    3:  [0, 0, 1, 0, 0, 0, 0, 0],  # EL
    4:  [0, 0, 0, 1, 0, 0, 0, 0],  # ER
    5:  [0, 0, 0, 0, 1, 0, 0, 0],  # LOC
    6:  [0, 0, 0, 0, 0, 1, 0, 0],  # NF
    7:  [0, 0, 0, 0, 0, 0, 1, 0],  # S
    8:  [0, 0, 0, 0, 0, 0, 0, 1],  # Random
    9:  [1, 0, 1, 0, 0, 0, 0, 0],  # C+EL
    10: [1, 0, 0, 1, 0, 0, 0, 0],  # C+ER
    11: [1, 0, 0, 0, 1, 0, 0, 0],  # C+L
    12: [1, 0, 0, 0, 0, 0, 1, 0],  # C+S
    13: [0, 1, 1, 0, 0, 0, 0, 0],  # D+EL
    14: [0, 1, 0, 1, 0, 0, 0, 0],  # D+ER
    15: [0, 1, 0, 0, 1, 0, 0, 0],  # D+L
    16: [0, 1, 0, 0, 0, 0, 1, 0],  # D+S
    17: [0, 0, 1, 0, 1, 0, 0, 0],  # EL+L
    18: [0, 0, 1, 0, 0, 0, 1, 0],  # EL+S
    19: [0, 0, 0, 1, 1, 0, 0, 0],  # ER+L
    20: [0, 0, 0, 1, 0, 0, 1, 0],  # ER+S
    21: [0, 0, 0, 0, 1, 0, 1, 0],  # L+S
    22: [1, 0, 1, 0, 1, 0, 0, 0],  # C+EL+L
    23: [1, 0, 1, 0, 0, 0, 1, 0],  # C+EL+S
    24: [1, 0, 0, 1, 1, 0, 0, 0],  # C+ER+L
    25: [1, 0, 0, 1, 0, 0, 1, 0],  # C+ER+S
    26: [1, 0, 0, 0, 1, 0, 1, 0],  # C+L+S
    27: [0, 1, 1, 0, 1, 0, 0, 0],  # D+EL+L
    28: [0, 1, 1, 0, 0, 0, 1, 0],  # D+EL+S
    29: [0, 1, 0, 1, 1, 0, 0, 0],  # D+ER+L
    30: [0, 1, 0, 1, 0, 0, 1, 0],  # D+ER+S
    31: [0, 1, 0, 0, 1, 0, 1, 0],  # D+L+S
    32: [0, 0, 1, 0, 1, 0, 1, 0],  # EL+L+S
    33: [0, 0, 0, 1, 1, 0, 1, 0],  # ER+L+S
    34: [1, 0, 1, 0, 1, 0, 1, 0],  # C+L+EL+S
    35: [1, 0, 0, 1, 1, 0, 1, 0],  # C+L+ER+S
    36: [0, 1, 1, 0, 1, 0, 1, 0],  # D+L+EL+S
    37: [0, 1, 0, 1, 1, 0, 1, 0],  # D+L+ER+S
}


def map_38_to_8(label_38: int) -> np.ndarray:
    """
    将 38 类标签映射到 8 类多标签
    
    Args:
        label_38: 38 类标签 ID (0-37)
        
    Returns:
        8 维二进制向量
    """
    if label_38 not in LABEL_38_TO_8:
        raise ValueError(f"Invalid label_38: {label_38}, must be in range [0, 37]")
    return np.array(LABEL_38_TO_8[label_38], dtype=np.float32)


def label_str_to_id(label_str: str) -> int:
    """
    将标签字符串转换为 38 类 ID
    
    Args:
        label_str: 标签字符串，如 "[1 0 0 0 0 0 0 0]"
        
    Returns:
        38 类标签 ID
    """
    if label_str not in CLASS_MAPPING:
        raise ValueError(f"Invalid label string: {label_str}")
    return CLASS_MAPPING[label_str]


def validate_mapping() -> Tuple[bool, Dict[str, any]]:
    """
    验证 38→8 映射的正确性
    
    Returns:
        (是否通过, 统计信息)
    """
    stats = {
        "total_classes": 38,
        "covered_classes": 0,
        "single_defect_classes": 0,
        "mixed_defect_classes": 0,
        "normal_class": 0,
        "errors": [],
    }
    
    for label_38, label_8 in LABEL_38_TO_8.items():
        stats["covered_classes"] += 1
        
        # 检查向量长度
        if len(label_8) != 8:
            stats["errors"].append(f"Class {label_38}: invalid length {len(label_8)}")
            continue
        
        # 统计缺陷数量
        defect_count = sum(label_8)
        if defect_count == 0:
            stats["normal_class"] += 1
        elif defect_count == 1:
            stats["single_defect_classes"] += 1
        else:
            stats["mixed_defect_classes"] += 1
    
    passed = (
        stats["covered_classes"] == 38 and
        stats["normal_class"] == 1 and
        stats["single_defect_classes"] == 8 and
        stats["mixed_defect_classes"] == 29 and
        len(stats["errors"]) == 0
    )
    
    return passed, stats


def print_mapping_stats():
    """打印映射统计信息"""
    passed, stats = validate_mapping()
    
    print("=" * 50)
    print("38→8 Label Mapping Statistics")
    print("=" * 50)
    print(f"Total classes: {stats['total_classes']}")
    print(f"Covered classes: {stats['covered_classes']}")
    print(f"Normal class: {stats['normal_class']}")
    print(f"Single defect classes: {stats['single_defect_classes']}")
    print(f"Mixed defect classes: {stats['mixed_defect_classes']}")
    
    if stats["errors"]:
        print("\nErrors:")
        for err in stats["errors"]:
            print(f"  - {err}")
    
    print(f"\nValidation: {'PASSED' if passed else 'FAILED'}")
    print("=" * 50)
    
    return passed


if __name__ == "__main__":
    print_mapping_stats()
