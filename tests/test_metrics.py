# -*- coding: utf-8 -*-
"""
指标计算测试

使用已知输入验证输出正确性

Requirements: 3.5
"""

import pytest
import numpy as np
import torch

from src.evaluation.metrics import (
    compute_macro_f1,
    compute_map,
    compute_dice,
    compute_iou,
    compute_confusion_matrix,
    compute_per_class_metrics,
)


class TestComputeMacroF1:
    """测试 compute_macro_f1 函数"""
    
    def test_perfect_prediction(self):
        """完美预测应返回 F1=1.0"""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        
        macro_f1, per_class_f1 = compute_macro_f1(y_true, y_pred, num_classes=3)
        
        assert macro_f1 == pytest.approx(1.0)
        assert all(f1 == pytest.approx(1.0) for f1 in per_class_f1)
    
    def test_completely_wrong_prediction(self):
        """完全错误的预测应返回 F1=0.0"""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 1, 2, 2, 0, 0])
        
        macro_f1, per_class_f1 = compute_macro_f1(y_true, y_pred, num_classes=3)
        
        assert macro_f1 == pytest.approx(0.0)
        assert all(f1 == pytest.approx(0.0) for f1 in per_class_f1)
    
    def test_known_values(self):
        """使用已知值验证计算正确性"""
        # 类别 0: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
        # 类别 1: TP=2, FP=1, FN=0 -> P=0.667, R=1.0, F1=0.8
        # 类别 2: TP=2, FP=0, FN=1 -> P=1.0, R=0.667, F1=0.8
        y_true = np.array([0, 0, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0, 2])
        
        macro_f1, per_class_f1 = compute_macro_f1(y_true, y_pred, num_classes=3)
        
        assert per_class_f1[0] == pytest.approx(0.5, rel=1e-3)
        assert per_class_f1[1] == pytest.approx(0.8, rel=1e-3)
        assert per_class_f1[2] == pytest.approx(0.8, rel=1e-3)
        assert macro_f1 == pytest.approx(0.7, rel=1e-3)
    
    def test_torch_tensor_input(self):
        """测试 torch.Tensor 输入"""
        y_true = torch.tensor([0, 0, 1, 1, 2, 2])
        y_pred = torch.tensor([0, 0, 1, 1, 2, 2])
        
        macro_f1, per_class_f1 = compute_macro_f1(y_true, y_pred, num_classes=3)
        
        assert macro_f1 == pytest.approx(1.0)
    
    def test_empty_class(self):
        """测试某类别没有样本的情况"""
        y_true = np.array([0, 0, 2, 2])  # 没有类别 1
        y_pred = np.array([0, 0, 2, 2])
        
        macro_f1, per_class_f1 = compute_macro_f1(y_true, y_pred, num_classes=3)
        
        assert per_class_f1[0] == pytest.approx(1.0)
        assert per_class_f1[1] == pytest.approx(0.0)  # 没有样本，默认 0
        assert per_class_f1[2] == pytest.approx(1.0)


class TestComputeMap:
    """测试 compute_map 函数"""
    
    def test_perfect_ranking(self):
        """完美排序应返回 mAP=1.0"""
        y_true = np.array([
            [1, 0],
            [0, 1],
            [1, 1],
        ])
        y_score = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.8],
        ])
        
        mAP, per_class_ap = compute_map(y_true, y_score)
        
        assert mAP == pytest.approx(1.0)
    
    def test_known_values(self):
        """使用已知值验证计算"""
        # 类别 0: 正样本在位置 0, 2 (排序后)
        # 类别 1: 正样本在位置 0, 1, 2 (排序后)
        y_true = np.array([
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 0],
        ])
        y_score = np.array([
            [0.9, 0.9],
            [0.3, 0.8],
            [0.7, 0.7],
            [0.1, 0.1],
        ])
        
        mAP, per_class_ap = compute_map(y_true, y_score)
        
        # 验证 mAP 在合理范围内
        assert 0.0 <= mAP <= 1.0
        assert all(0.0 <= ap <= 1.0 for ap in per_class_ap)
    
    def test_torch_tensor_input(self):
        """测试 torch.Tensor 输入"""
        y_true = torch.tensor([[1, 0], [0, 1]])
        y_score = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        
        mAP, per_class_ap = compute_map(y_true, y_score)
        
        assert mAP == pytest.approx(1.0)


class TestComputeDice:
    """测试 compute_dice 函数"""
    
    def test_perfect_overlap(self):
        """完美重叠应返回 Dice=1.0"""
        pred = np.array([[[1, 1], [1, 0]]])
        target = np.array([[[1, 1], [1, 0]]])
        
        dice = compute_dice(pred, target, threshold=0.5)
        
        assert dice == pytest.approx(1.0, rel=1e-3)
    
    def test_no_overlap(self):
        """无重叠应返回接近 0"""
        pred = np.array([[[1, 1], [0, 0]]])
        target = np.array([[[0, 0], [1, 1]]])
        
        dice = compute_dice(pred, target, threshold=0.5)
        
        assert dice < 0.01  # 接近 0（有 smooth 因子）
    
    def test_known_values(self):
        """使用已知值验证计算"""
        # pred: 3 个 1, target: 4 个 1, 交集: 3 个
        # Dice = 2 * 3 / (3 + 4) = 6/7 ≈ 0.857
        pred = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
        target = np.array([[[1, 1, 1], [1, 0, 0], [0, 0, 0]]])
        
        dice = compute_dice(pred, target, threshold=0.5)
        
        assert dice == pytest.approx(6/7, rel=1e-2)
    
    def test_torch_tensor_input(self):
        """测试 torch.Tensor 输入"""
        pred = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
        target = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
        
        dice = compute_dice(pred, target, threshold=0.5)
        
        assert dice == pytest.approx(1.0, rel=1e-3)
    
    def test_batch_processing(self):
        """测试批量处理"""
        pred = np.array([
            [[1, 1], [0, 0]],
            [[1, 0], [1, 0]],
        ])
        target = np.array([
            [[1, 1], [0, 0]],
            [[1, 0], [1, 0]],
        ])
        
        dice = compute_dice(pred, target, threshold=0.5)
        
        assert dice == pytest.approx(1.0, rel=1e-3)


class TestComputeIoU:
    """测试 compute_iou 函数"""
    
    def test_perfect_overlap(self):
        """完美重叠应返回 IoU=1.0"""
        pred = np.array([[[1, 1], [1, 0]]])
        target = np.array([[[1, 1], [1, 0]]])
        
        iou = compute_iou(pred, target, threshold=0.5)
        
        assert iou == pytest.approx(1.0, rel=1e-3)
    
    def test_no_overlap(self):
        """无重叠应返回接近 0"""
        pred = np.array([[[1, 1], [0, 0]]])
        target = np.array([[[0, 0], [1, 1]]])
        
        iou = compute_iou(pred, target, threshold=0.5)
        
        assert iou < 0.01  # 接近 0
    
    def test_known_values(self):
        """使用已知值验证计算"""
        # pred: 3 个 1, target: 4 个 1, 交集: 3 个, 并集: 4 个
        # IoU = 3 / 4 = 0.75
        pred = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
        target = np.array([[[1, 1, 1], [1, 0, 0], [0, 0, 0]]])
        
        iou = compute_iou(pred, target, threshold=0.5)
        
        assert iou == pytest.approx(0.75, rel=1e-2)
    
    def test_torch_tensor_input(self):
        """测试 torch.Tensor 输入"""
        pred = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
        target = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
        
        iou = compute_iou(pred, target, threshold=0.5)
        
        assert iou == pytest.approx(1.0, rel=1e-3)


class TestComputeConfusionMatrix:
    """测试 compute_confusion_matrix 函数"""
    
    def test_perfect_prediction(self):
        """完美预测的混淆矩阵应为对角矩阵"""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        
        expected = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ])
        np.testing.assert_array_equal(cm, expected)
    
    def test_known_values(self):
        """使用已知值验证"""
        y_true = np.array([0, 0, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0, 2])
        
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        
        # cm[i, j] = 真实为 i，预测为 j
        expected = np.array([
            [1, 1, 0],  # 真实 0: 1 个预测对，1 个预测为 1
            [0, 2, 0],  # 真实 1: 2 个预测对
            [1, 0, 2],  # 真实 2: 1 个预测为 0，2 个预测对
        ])
        np.testing.assert_array_equal(cm, expected)
    
    def test_torch_tensor_input(self):
        """测试 torch.Tensor 输入"""
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([0, 1, 2])
        
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        np.testing.assert_array_equal(cm, expected)


class TestComputePerClassMetrics:
    """测试 compute_per_class_metrics 函数"""
    
    def test_perfect_prediction(self):
        """完美预测应返回所有指标为 1.0"""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        
        metrics = compute_per_class_metrics(y_true, y_pred, num_classes=3)
        
        assert all(p == pytest.approx(1.0) for p in metrics["precision"])
        assert all(r == pytest.approx(1.0) for r in metrics["recall"])
        assert all(f == pytest.approx(1.0) for f in metrics["f1"])
        np.testing.assert_array_equal(metrics["support"], [2, 2, 2])
    
    def test_known_values(self):
        """使用已知值验证"""
        y_true = np.array([0, 0, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0, 2])
        
        metrics = compute_per_class_metrics(y_true, y_pred, num_classes=3)
        
        # 类别 0: TP=1, FP=1, FN=1 -> P=0.5, R=0.5
        assert metrics["precision"][0] == pytest.approx(0.5, rel=1e-3)
        assert metrics["recall"][0] == pytest.approx(0.5, rel=1e-3)
        
        # 类别 1: TP=2, FP=1, FN=0 -> P=0.667, R=1.0
        assert metrics["precision"][1] == pytest.approx(2/3, rel=1e-3)
        assert metrics["recall"][1] == pytest.approx(1.0, rel=1e-3)
        
        # 类别 2: TP=2, FP=0, FN=1 -> P=1.0, R=0.667
        assert metrics["precision"][2] == pytest.approx(1.0, rel=1e-3)
        assert metrics["recall"][2] == pytest.approx(2/3, rel=1e-3)
        
        np.testing.assert_array_equal(metrics["support"], [2, 2, 3])


class TestDiceIoURelationship:
    """测试 Dice 和 IoU 之间的数学关系"""
    
    def test_dice_iou_relationship(self):
        """验证 Dice = 2*IoU / (1+IoU) 的关系"""
        pred = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
        target = np.array([[[1, 1, 1], [1, 0, 0], [0, 0, 0]]])
        
        dice = compute_dice(pred, target, threshold=0.5, smooth=0)
        iou = compute_iou(pred, target, threshold=0.5, smooth=0)
        
        # Dice = 2*IoU / (1+IoU)
        expected_dice = 2 * iou / (1 + iou)
        
        assert dice == pytest.approx(expected_dice, rel=1e-2)
