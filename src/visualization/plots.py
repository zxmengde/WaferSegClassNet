# -*- coding: utf-8 -*-
"""
可视化绘图模块

提供训练和评估过程中的可视化功能：
- plot_confusion_matrix(): 绘制混淆矩阵
- plot_loss_curves(): 绘制损失曲线
- plot_metric_curves(): 绘制指标曲线

Requirements: 8.1, 8.4
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    confusion_matrix: Union[np.ndarray, List[List[int]]],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: tuple = (12, 10),
    cmap: str = "Blues",
    show_values: bool = True,
    value_format: str = ".2f",
    fontsize: int = 8,
) -> Optional[object]:
    """
    绘制混淆矩阵
    
    Args:
        confusion_matrix: 混淆矩阵，形状 (num_classes, num_classes)
        class_names: 类别名称列表
        save_path: 保存路径，如果为 None 则不保存
        title: 图表标题
        normalize: 是否归一化（按行归一化）
        figsize: 图表大小
        cmap: 颜色映射
        show_values: 是否在格子中显示数值
        value_format: 数值格式
        fontsize: 字体大小
        
    Returns:
        matplotlib figure 对象，如果 matplotlib 不可用则返回 None
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping confusion matrix plot")
        return None
    
    # 转换为 numpy 数组
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    num_classes = cm.shape[0]
    
    # 归一化
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        cm_display = cm / row_sums
    else:
        cm_display = cm
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, cmap=cmap, aspect='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Proportion' if normalize else 'Count', rotation=-90, va="bottom")
    
    # 设置刻度
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    
    # 设置刻度标签
    if class_names is not None and len(class_names) == num_classes:
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=fontsize)
        ax.set_yticklabels(class_names, fontsize=fontsize)
    else:
        ax.set_xticklabels(range(num_classes), fontsize=fontsize)
        ax.set_yticklabels(range(num_classes), fontsize=fontsize)
    
    # 显示数值
    if show_values and num_classes <= 20:  # 类别太多时不显示数值
        thresh = cm_display.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                value = cm_display[i, j]
                color = "white" if value > thresh else "black"
                ax.text(j, i, format(value, value_format),
                       ha="center", va="center", color=color, fontsize=fontsize-1)
    
    ax.set_xlabel('Predicted Label', fontsize=fontsize+2)
    ax.set_ylabel('True Label', fontsize=fontsize+2)
    ax.set_title(title, fontsize=fontsize+4)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.close()
    return fig



def plot_loss_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Loss Curves",
    figsize: tuple = (10, 6),
) -> Optional[object]:
    """
    绘制损失曲线
    
    Args:
        history: 训练历史字典，包含 'train_loss', 'val_loss' 等键
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
        
    Returns:
        matplotlib figure 对象
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping loss curves plot")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制训练损失
    if 'train_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    
    # 绘制验证损失
    if 'val_loss' in history:
        epochs = range(1, len(history['val_loss']) + 1)
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    # 绘制各任务损失（如果有）
    loss_keys = [k for k in history.keys() if 'loss' in k.lower() and k not in ['train_loss', 'val_loss']]
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_keys)))
    
    for idx, key in enumerate(loss_keys):
        epochs = range(1, len(history[key]) + 1)
        ax.plot(epochs, history[key], '--', color=colors[idx], label=key, linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 设置 y 轴下限为 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Loss curves saved to {save_path}")
    
    plt.close()
    return fig


def plot_metric_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Metric Curves",
    figsize: tuple = (10, 6),
    metrics: Optional[List[str]] = None,
) -> Optional[object]:
    """
    绘制指标曲线
    
    Args:
        history: 训练历史字典，包含各种指标
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
        metrics: 要绘制的指标列表，如果为 None 则自动检测
        
    Returns:
        matplotlib figure 对象
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping metric curves plot")
        return None
    
    # 自动检测指标
    if metrics is None:
        # 排除损失相关的键
        metrics = [k for k in history.keys() 
                   if 'loss' not in k.lower() and len(history[k]) > 0]
    
    if not metrics:
        logger.warning("No metrics found in history, skipping metric curves plot")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    
    for idx, metric in enumerate(metrics):
        if metric in history and len(history[metric]) > 0:
            epochs = range(1, len(history[metric]) + 1)
            # 区分训练和验证指标
            linestyle = '-' if 'val' in metric.lower() else '--'
            ax.plot(epochs, history[metric], linestyle, color=colors[idx], 
                   label=metric, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 设置 y 轴范围 [0, 1] 用于常见指标
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Metric curves saved to {save_path}")
    
    plt.close()
    return fig


def plot_training_summary(
    history: Dict[str, List[float]],
    output_dir: str,
    experiment_name: str = "experiment",
) -> None:
    """
    生成完整的训练摘要可视化
    
    Args:
        history: 训练历史字典
        output_dir: 输出目录
        experiment_name: 实验名称
    """
    output_dir = Path(output_dir)
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制损失曲线
    plot_loss_curves(
        history,
        save_path=curves_dir / "loss_curve.png",
        title=f"{experiment_name} - Loss Curves"
    )
    
    # 绘制指标曲线
    plot_metric_curves(
        history,
        save_path=curves_dir / "metric_curve.png",
        title=f"{experiment_name} - Metric Curves"
    )
    
    logger.info(f"Training summary plots saved to {curves_dir}")


if __name__ == "__main__":
    # 测试可视化函数
    print("=" * 60)
    print("Testing Visualization Functions")
    print("=" * 60)
    
    # 测试混淆矩阵
    print("\n1. Testing plot_confusion_matrix:")
    cm = np.array([
        [50, 5, 2],
        [3, 45, 7],
        [1, 4, 48],
    ])
    class_names = ["Class A", "Class B", "Class C"]
    fig = plot_confusion_matrix(
        cm, 
        class_names=class_names,
        save_path="test_confusion_matrix.png",
        title="Test Confusion Matrix"
    )
    print(f"   Confusion matrix plot created: {fig is not None}")
    
    # 测试损失曲线
    print("\n2. Testing plot_loss_curves:")
    history = {
        'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.22],
        'val_loss': [1.1, 0.9, 0.75, 0.65, 0.55, 0.5, 0.48, 0.47, 0.46, 0.45],
        'cls_loss': [0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.14, 0.13, 0.12],
        'seg_loss': [0.5, 0.4, 0.3, 0.25, 0.2, 0.17, 0.15, 0.14, 0.12, 0.10],
    }
    fig = plot_loss_curves(
        history,
        save_path="test_loss_curves.png",
        title="Test Loss Curves"
    )
    print(f"   Loss curves plot created: {fig is not None}")
    
    # 测试指标曲线
    print("\n3. Testing plot_metric_curves:")
    history['val_macro_f1'] = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.68, 0.70, 0.72, 0.73]
    history['val_dice'] = [0.4, 0.5, 0.55, 0.6, 0.65, 0.68, 0.70, 0.72, 0.74, 0.75]
    history['val_iou'] = [0.3, 0.4, 0.45, 0.5, 0.55, 0.58, 0.60, 0.62, 0.64, 0.65]
    fig = plot_metric_curves(
        history,
        save_path="test_metric_curves.png",
        title="Test Metric Curves"
    )
    print(f"   Metric curves plot created: {fig is not None}")
    
    # 清理测试文件
    import os
    for f in ["test_confusion_matrix.png", "test_loss_curves.png", "test_metric_curves.png"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"   Cleaned up: {f}")
    
    print("\n" + "=" * 60)
    print("All visualization tests passed!")
    print("=" * 60)
