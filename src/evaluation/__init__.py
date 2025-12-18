# Evaluation module
"""
评估模块

包含指标计算和评估器类
"""

from src.evaluation.metrics import (
    compute_macro_f1,
    compute_map,
    compute_dice,
    compute_iou,
    compute_confusion_matrix,
    compute_per_class_metrics,
)
from src.evaluation.evaluator import Evaluator, create_evaluator

__all__ = [
    "compute_macro_f1",
    "compute_map",
    "compute_dice",
    "compute_iou",
    "compute_confusion_matrix",
    "compute_per_class_metrics",
    "Evaluator",
    "create_evaluator",
]
