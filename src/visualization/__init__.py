# -*- coding: utf-8 -*-
"""
可视化模块

提供训练和评估过程中的可视化功能：
- plots: 绘图函数（混淆矩阵、损失曲线、指标曲线）
- overlays: Overlay 可视化（分割、分离热力图、伪mask）
"""

from .plots import (
    plot_confusion_matrix,
    plot_loss_curves,
    plot_metric_curves,
    plot_training_summary,
)

from .overlays import (
    generate_seg_overlays,
    generate_separation_heatmaps,
    generate_pseudo_mask_overlays,
    generate_all_visualizations,
    COMPONENT_NAMES,
)

__all__ = [
    # plots
    "plot_confusion_matrix",
    "plot_loss_curves",
    "plot_metric_curves",
    "plot_training_summary",
    # overlays
    "generate_seg_overlays",
    "generate_separation_heatmaps",
    "generate_pseudo_mask_overlays",
    "generate_all_visualizations",
    "COMPONENT_NAMES",
]
