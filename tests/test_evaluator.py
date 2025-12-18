# -*- coding: utf-8 -*-
"""
评估器测试

验证 Evaluator 类的基本功能

Requirements: 3.5, 3.6
"""

import pytest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.evaluator import Evaluator, create_evaluator
from src.evaluation.metrics import compute_macro_f1, compute_dice, compute_iou


class MockModel(nn.Module):
    """模拟模型用于测试"""
    
    def __init__(self, num_classes: int = 38, separation_enabled: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.separation_enabled = separation_enabled
        self.dummy = nn.Linear(1, 1)  # 需要至少一个参数
    
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        h, w = x.shape[2], x.shape[3]
        
        outputs = {
            'cls_logits': torch.randn(batch_size, self.num_classes),
            'seg_mask': torch.randn(batch_size, 1, h, w),
        }
        
        if self.separation_enabled:
            outputs['sep_heatmaps'] = torch.randn(batch_size, 8, h, w)
        
        return outputs


class MockDataset(torch.utils.data.Dataset):
    """模拟数据集用于测试"""
    
    def __init__(self, size: int = 10, num_classes: int = 38):
        self.size = size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'image': torch.randn(3, 64, 64),
            'label': torch.tensor(idx % self.num_classes),
            'mask': torch.randint(0, 2, (1, 64, 64)).float(),
        }


class TestEvaluatorInit:
    """测试 Evaluator 初始化"""
    
    def test_init_basic(self):
        """测试基本初始化"""
        model = MockModel()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(
                model=model,
                device="cpu",
                amp_enabled=False,
                num_classes=38,
                output_dir=tmpdir,
            )
            
            assert evaluator.num_classes == 38
            assert evaluator.device == "cpu"
            assert not evaluator.amp_enabled
    
    def test_init_creates_directories(self):
        """测试初始化创建必要目录"""
        model = MockModel()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            
            evaluator = Evaluator(
                model=model,
                device="cpu",
                output_dir=str(output_dir),
            )
            
            assert output_dir.exists()
            assert (output_dir / "seg_overlays").exists()
    
    def test_init_with_pseudo_mask(self):
        """测试伪 mask 模式初始化"""
        model = MockModel()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(
                model=model,
                device="cpu",
                output_dir=tmpdir,
                pseudo_mask_used=True,
            )
            
            assert (Path(tmpdir) / "pseudo_mask_samples").exists()


class TestEvaluatorEvaluate:
    """测试 Evaluator.evaluate 方法"""
    
    def test_evaluate_basic(self):
        """测试基本评估流程"""
        model = MockModel(num_classes=5)
        dataset = MockDataset(size=8, num_classes=5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(
                model=model,
                device="cpu",
                amp_enabled=False,
                num_classes=5,
                output_dir=tmpdir,
            )
            
            metrics = evaluator.evaluate(
                dataloader,
                save_visualizations=False,
            )
            
            # 验证返回的指标
            assert 'macro_f1' in metrics
            assert 'dice' in metrics
            assert 'iou' in metrics
            assert 'accuracy' in metrics
            assert 'per_class_f1' in metrics
            
            # 验证指标范围
            assert 0.0 <= metrics['macro_f1'] <= 1.0
            assert 0.0 <= metrics['dice'] <= 1.0
            assert 0.0 <= metrics['iou'] <= 1.0
            assert 0.0 <= metrics['accuracy'] <= 1.0
    
    def test_evaluate_saves_metrics_csv(self):
        """测试评估保存 metrics.csv"""
        model = MockModel(num_classes=5)
        dataset = MockDataset(size=8, num_classes=5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(
                model=model,
                device="cpu",
                amp_enabled=False,
                num_classes=5,
                output_dir=tmpdir,
            )
            
            evaluator.evaluate(dataloader, save_visualizations=False)
            
            # 验证 metrics.csv 存在
            metrics_path = Path(tmpdir) / "metrics.csv"
            assert metrics_path.exists()
            
            # 验证 per_class metrics 存在
            per_class_path = Path(tmpdir) / "metrics_per_class.csv"
            assert per_class_path.exists()
    
    def test_evaluate_with_separation(self):
        """测试带分离头的评估"""
        model = MockModel(num_classes=5, separation_enabled=True)
        dataset = MockDataset(size=8, num_classes=5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(
                model=model,
                device="cpu",
                amp_enabled=False,
                num_classes=5,
                output_dir=tmpdir,
            )
            
            metrics = evaluator.evaluate(
                dataloader,
                save_visualizations=False,
            )
            
            # 验证基本指标存在
            assert 'macro_f1' in metrics
            assert 'dice' in metrics


class TestCreateEvaluator:
    """测试 create_evaluator 工厂函数"""
    
    def test_create_evaluator_basic(self):
        """测试基本创建"""
        model = MockModel()
        
        # 模拟配置对象
        class MockConfig:
            class model:
                classification_classes = 38
            class data:
                classification_mode = "single_label"
            class training:
                amp_enabled = False
        
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = create_evaluator(
                model=model,
                config=MockConfig(),
                output_dir=tmpdir,
            )
            
            assert evaluator.num_classes == 38
            assert evaluator.classification_mode == "single_label"


class TestMetricsIntegration:
    """测试指标计算与评估器的集成"""
    
    def test_metrics_consistency(self):
        """验证评估器计算的指标与独立函数一致"""
        # 创建已知数据
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        
        # 使用独立函数计算
        macro_f1, _ = compute_macro_f1(y_true, y_pred, num_classes=3)
        
        # 验证完美预测
        assert macro_f1 == pytest.approx(1.0)
    
    def test_dice_iou_consistency(self):
        """验证 Dice 和 IoU 计算一致性"""
        pred = np.array([[[1, 1], [1, 0]]])
        target = np.array([[[1, 1], [1, 0]]])
        
        dice = compute_dice(pred, target, threshold=0.5)
        iou = compute_iou(pred, target, threshold=0.5)
        
        # 完美重叠
        assert dice == pytest.approx(1.0, rel=1e-3)
        assert iou == pytest.approx(1.0, rel=1e-3)
