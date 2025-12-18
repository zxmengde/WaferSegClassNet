# -*- coding: utf-8 -*-
"""
评估指标模块

包含分类和分割任务的评估指标计算函数：
- compute_macro_f1(): 计算宏平均 F1 分数
- compute_map(): 计算多标签分类的 mAP
- compute_dice(): 计算 Dice 系数
- compute_iou(): 计算 IoU (Intersection over Union)

Requirements: 3.5
"""

from typing import Optional, Union, List, Tuple
import numpy as np
import torch


def compute_macro_f1(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    num_classes: Optional[int] = None,
    zero_division: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """
    计算宏平均 F1 分数
    
    Macro-F1 = (1/N) * Σ F1_i
    
    Args:
        y_true: 真实标签 (N,)，整数类别索引
        y_pred: 预测标签 (N,)，整数类别索引
        num_classes: 类别数量，如果为 None 则自动推断
        zero_division: 当某类别没有样本时的默认值
        
    Returns:
        macro_f1: 宏平均 F1 分数
        per_class_f1: 每类的 F1 分数数组
    """
    # 转换为 numpy 数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # 推断类别数
    if num_classes is None:
        num_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    
    per_class_f1 = np.zeros(num_classes, dtype=np.float64)
    
    for c in range(num_classes):
        # 计算 TP, FP, FN
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        # 计算 Precision 和 Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        
        # 计算 F1
        if precision + recall > 0:
            per_class_f1[c] = 2 * precision * recall / (precision + recall)
        else:
            per_class_f1[c] = zero_division
    
    macro_f1 = float(np.mean(per_class_f1))
    
    return macro_f1, per_class_f1



def compute_map(
    y_true: Union[np.ndarray, torch.Tensor],
    y_score: Union[np.ndarray, torch.Tensor],
) -> Tuple[float, np.ndarray]:
    """
    计算多标签分类的 mAP (mean Average Precision)
    
    mAP = (1/C) * Σ AP_c
    
    Args:
        y_true: 真实多标签 (N, C)，二值矩阵
        y_score: 预测分数 (N, C)，概率或 logits
        
    Returns:
        mAP: 平均 AP 分数
        per_class_ap: 每类的 AP 分数数组
    """
    # 转换为 numpy 数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # 确保是 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)
    
    num_classes = y_true.shape[1]
    per_class_ap = np.zeros(num_classes, dtype=np.float64)
    
    for c in range(num_classes):
        # 获取该类的真实标签和预测分数
        y_true_c = y_true[:, c]
        y_score_c = y_score[:, c]
        
        # 如果该类没有正样本，AP 为 0
        if np.sum(y_true_c) == 0:
            per_class_ap[c] = 0.0
            continue
        
        # 按预测分数降序排序
        sorted_indices = np.argsort(-y_score_c)
        y_true_sorted = y_true_c[sorted_indices]
        
        # 计算 AP (Average Precision)
        # AP = Σ (R_n - R_{n-1}) * P_n
        tp_cumsum = np.cumsum(y_true_sorted)
        precision_at_k = tp_cumsum / np.arange(1, len(y_true_sorted) + 1)
        
        # 只在正样本位置计算
        ap = np.sum(precision_at_k * y_true_sorted) / np.sum(y_true_c)
        per_class_ap[c] = ap
    
    mAP = float(np.mean(per_class_ap))
    
    return mAP, per_class_ap


def compute_dice(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-6,
    threshold: Optional[float] = 0.5,
) -> float:
    """
    计算 Dice 系数
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        pred: 预测 mask，形状 (N, H, W) 或 (N, 1, H, W)
        target: 真实 mask，形状 (N, H, W) 或 (N, 1, H, W)
        smooth: 平滑因子，防止除零
        threshold: 二值化阈值，如果为 None 则不进行二值化
        
    Returns:
        dice: Dice 系数 (0-1)
    """
    # 转换为 numpy 数组
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    
    # 展平到 (N, -1)
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    
    # 二值化
    if threshold is not None:
        pred = (pred > threshold).astype(np.float64)
    target = (target > 0.5).astype(np.float64)
    
    # 计算 Dice
    intersection = np.sum(pred * target, axis=1)
    union = np.sum(pred, axis=1) + np.sum(target, axis=1)
    
    dice_per_sample = (2.0 * intersection + smooth) / (union + smooth)
    dice = float(np.mean(dice_per_sample))
    
    return dice


def compute_iou(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-6,
    threshold: Optional[float] = 0.5,
) -> float:
    """
    计算 IoU (Intersection over Union)
    
    IoU = |A ∩ B| / |A ∪ B| = TP / (TP + FP + FN)
    
    Args:
        pred: 预测 mask，形状 (N, H, W) 或 (N, 1, H, W)
        target: 真实 mask，形状 (N, H, W) 或 (N, 1, H, W)
        smooth: 平滑因子，防止除零
        threshold: 二值化阈值，如果为 None 则不进行二值化
        
    Returns:
        iou: IoU 分数 (0-1)
    """
    # 转换为 numpy 数组
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    
    # 展平到 (N, -1)
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    
    # 二值化
    if threshold is not None:
        pred = (pred > threshold).astype(np.float64)
    target = (target > 0.5).astype(np.float64)
    
    # 计算 IoU
    intersection = np.sum(pred * target, axis=1)
    union = np.sum(pred, axis=1) + np.sum(target, axis=1) - intersection
    
    iou_per_sample = (intersection + smooth) / (union + smooth)
    iou = float(np.mean(iou_per_sample))
    
    return iou


def compute_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        y_true: 真实标签 (N,)
        y_pred: 预测标签 (N,)
        num_classes: 类别数量
        
    Returns:
        confusion_matrix: (num_classes, num_classes) 混淆矩阵
            cm[i, j] = 真实类别为 i，预测为 j 的样本数
    """
    # 转换为 numpy 数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    
    # 推断类别数
    if num_classes is None:
        num_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    
    # 构建混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    return cm


def compute_per_class_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    计算每类的详细指标
    
    Args:
        y_true: 真实标签 (N,)
        y_pred: 预测标签 (N,)
        num_classes: 类别数量
        class_names: 类别名称列表
        
    Returns:
        metrics_dict: 包含每类指标的字典
            - precision: 每类精确率
            - recall: 每类召回率
            - f1: 每类 F1 分数
            - support: 每类样本数
    """
    # 转换为 numpy 数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # 推断类别数
    if num_classes is None:
        num_classes = max(int(y_true.max()), int(y_pred.max())) + 1
    
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    f1 = np.zeros(num_classes, dtype=np.float64)
    support = np.zeros(num_classes, dtype=np.int64)
    
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        support[c] = np.sum(y_true == c)
        
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        else:
            f1[c] = 0.0
    
    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }
    
    if class_names is not None:
        result["class_names"] = class_names
    
    return result


if __name__ == "__main__":
    # 测试指标计算
    print("=" * 60)
    print("Testing Metrics Functions")
    print("=" * 60)
    
    # 测试 compute_macro_f1
    print("\n1. compute_macro_f1:")
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 2])
    macro_f1, per_class_f1 = compute_macro_f1(y_true, y_pred, num_classes=3)
    print(f"   Macro-F1: {macro_f1:.4f}")
    print(f"   Per-class F1: {per_class_f1}")
    
    # 测试 compute_map
    print("\n2. compute_map:")
    y_true_ml = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
    ])
    y_score_ml = np.array([
        [0.9, 0.1, 0.8],
        [0.2, 0.9, 0.7],
        [0.8, 0.7, 0.3],
        [0.1, 0.2, 0.9],
    ])
    mAP, per_class_ap = compute_map(y_true_ml, y_score_ml)
    print(f"   mAP: {mAP:.4f}")
    print(f"   Per-class AP: {per_class_ap}")
    
    # 测试 compute_dice
    print("\n3. compute_dice:")
    pred_mask = np.array([
        [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
    ]).astype(float)
    target_mask = np.array([
        [[1, 1, 1], [1, 0, 0], [0, 0, 0]],
        [[0, 1, 1], [0, 1, 0], [0, 0, 0]],
    ]).astype(float)
    dice = compute_dice(pred_mask, target_mask, threshold=0.5)
    print(f"   Dice: {dice:.4f}")
    
    # 测试 compute_iou
    print("\n4. compute_iou:")
    iou = compute_iou(pred_mask, target_mask, threshold=0.5)
    print(f"   IoU: {iou:.4f}")
    
    # 测试 compute_confusion_matrix
    print("\n5. compute_confusion_matrix:")
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
    print(f"   Confusion Matrix:\n{cm}")
    
    # 测试 compute_per_class_metrics
    print("\n6. compute_per_class_metrics:")
    metrics = compute_per_class_metrics(y_true, y_pred, num_classes=3)
    print(f"   Precision: {metrics['precision']}")
    print(f"   Recall: {metrics['recall']}")
    print(f"   F1: {metrics['f1']}")
    print(f"   Support: {metrics['support']}")
    
    print("\n" + "=" * 60)
    print("All metrics tests passed!")
    print("=" * 60)
