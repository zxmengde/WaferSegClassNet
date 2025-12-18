# -*- coding: utf-8 -*-
"""
可视化模块测试

测试 plots.py 和 overlays.py 的功能
"""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch


class TestPlots:
    """测试绘图函数"""
    
    def test_plot_confusion_matrix_basic(self):
        """测试基本混淆矩阵绘制"""
        from src.visualization.plots import plot_confusion_matrix
        
        cm = np.array([
            [50, 5, 2],
            [3, 45, 7],
            [1, 4, 48],
        ])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "cm.png"
            fig = plot_confusion_matrix(cm, save_path=str(save_path))
            
            assert fig is not None
            assert save_path.exists()
    
    def test_plot_confusion_matrix_with_class_names(self):
        """测试带类别名称的混淆矩阵"""
        from src.visualization.plots import plot_confusion_matrix
        
        cm = np.array([
            [50, 5],
            [3, 45],
        ])
        class_names = ["Class A", "Class B"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "cm.png"
            fig = plot_confusion_matrix(
                cm, 
                class_names=class_names,
                save_path=str(save_path),
                normalize=True
            )
            
            assert fig is not None
            assert save_path.exists()
    
    def test_plot_loss_curves(self):
        """测试损失曲线绘制"""
        from src.visualization.plots import plot_loss_curves
        
        history = {
            'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
            'val_loss': [1.1, 0.9, 0.75, 0.65, 0.55],
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "loss.png"
            fig = plot_loss_curves(history, save_path=str(save_path))
            
            assert fig is not None
            assert save_path.exists()
    
    def test_plot_metric_curves(self):
        """测试指标曲线绘制"""
        from src.visualization.plots import plot_metric_curves
        
        history = {
            'val_macro_f1': [0.3, 0.4, 0.5, 0.55, 0.6],
            'val_dice': [0.4, 0.5, 0.55, 0.6, 0.65],
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "metrics.png"
            fig = plot_metric_curves(history, save_path=str(save_path))
            
            assert fig is not None
            assert save_path.exists()
    
    def test_plot_training_summary(self):
        """测试训练摘要生成"""
        from src.visualization.plots import plot_training_summary
        
        history = {
            'train_loss': [1.0, 0.8, 0.6],
            'val_loss': [1.1, 0.9, 0.75],
            'val_macro_f1': [0.3, 0.4, 0.5],
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_training_summary(history, temp_dir, "test_exp")
            
            curves_dir = Path(temp_dir) / "curves"
            assert curves_dir.exists()
            assert (curves_dir / "loss_curve.png").exists()
            assert (curves_dir / "metric_curve.png").exists()


class TestOverlays:
    """测试 overlay 可视化函数"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        batch_size = 5
        height, width = 32, 32
        
        images = np.random.rand(batch_size, 3, height, width).astype(np.float32)
        gt_masks = np.random.rand(batch_size, 1, height, width).astype(np.float32) > 0.5
        pred_masks = np.random.rand(batch_size, 1, height, width).astype(np.float32)
        sep_heatmaps = np.random.rand(batch_size, 8, height, width).astype(np.float32)
        
        return {
            'images': images,
            'gt_masks': gt_masks.astype(np.float32),
            'pred_masks': pred_masks,
            'sep_heatmaps': sep_heatmaps,
        }
    
    def test_generate_seg_overlays(self, sample_data):
        """测试分割 overlay 生成"""
        from src.visualization.overlays import generate_seg_overlays
        
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = generate_seg_overlays(
                sample_data['images'],
                sample_data['pred_masks'],
                sample_data['gt_masks'],
                output_dir=temp_dir,
                max_samples=3,
            )
            
            assert len(paths) == 3
            for path in paths:
                assert Path(path).exists()
    
    def test_generate_seg_overlays_with_torch(self, sample_data):
        """测试使用 torch tensor 的分割 overlay"""
        from src.visualization.overlays import generate_seg_overlays
        
        images = torch.from_numpy(sample_data['images'])
        pred_masks = torch.from_numpy(sample_data['pred_masks'])
        gt_masks = torch.from_numpy(sample_data['gt_masks'])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = generate_seg_overlays(
                images, pred_masks, gt_masks,
                output_dir=temp_dir,
                max_samples=2,
            )
            
            assert len(paths) == 2
    
    def test_generate_separation_heatmaps(self, sample_data):
        """测试分离热力图生成"""
        from src.visualization.overlays import generate_separation_heatmaps
        
        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths, tensor_paths = generate_separation_heatmaps(
                sample_data['images'],
                sample_data['sep_heatmaps'],
                output_dir=temp_dir,
                max_samples=3,
            )
            
            assert len(image_paths) == 3
            assert len(tensor_paths) == 3
            
            for path in image_paths:
                assert Path(path).exists()
            for path in tensor_paths:
                assert Path(path).exists()
                # 验证 tensor 可以加载
                loaded = torch.load(path, weights_only=True)
                assert loaded.shape == (8, 32, 32)
    
    def test_generate_pseudo_mask_overlays_min_samples(self, sample_data):
        """测试伪 mask overlay 最小样本数"""
        from src.visualization.overlays import generate_pseudo_mask_overlays
        
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = generate_pseudo_mask_overlays(
                sample_data['images'],
                sample_data['gt_masks'],
                output_dir=temp_dir,
                min_samples=10,  # 要求至少10张
                max_samples=20,
            )
            
            # 由于只有5个样本，应该生成5张
            assert len(paths) == 5
    
    def test_generate_pseudo_mask_overlays_enough_samples(self):
        """测试伪 mask overlay 足够样本时"""
        from src.visualization.overlays import generate_pseudo_mask_overlays
        
        np.random.seed(42)
        batch_size = 15
        height, width = 32, 32
        
        images = np.random.rand(batch_size, 3, height, width).astype(np.float32)
        masks = (np.random.rand(batch_size, 1, height, width) > 0.5).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = generate_pseudo_mask_overlays(
                images, masks,
                output_dir=temp_dir,
                min_samples=10,
                max_samples=12,
            )
            
            # 应该生成12张（max_samples）
            assert len(paths) == 12
    
    def test_generate_all_visualizations(self, sample_data):
        """测试生成所有可视化"""
        from src.visualization.overlays import generate_all_visualizations
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_all_visualizations(
                sample_data['images'],
                sample_data['pred_masks'],
                sample_data['gt_masks'],
                output_dir=temp_dir,
                sep_heatmaps=sample_data['sep_heatmaps'],
                pseudo_mask_used=True,
                max_samples=3,
            )
            
            assert 'seg_overlays' in result
            assert 'separation_images' in result
            assert 'separation_tensors' in result
            assert 'pseudo_mask_overlays' in result
            
            assert len(result['seg_overlays']) == 3
            assert len(result['separation_images']) == 3


class TestVisualizationIntegration:
    """集成测试"""
    
    def test_full_visualization_pipeline(self):
        """测试完整可视化流程"""
        from src.visualization import (
            plot_confusion_matrix,
            plot_loss_curves,
            plot_metric_curves,
            generate_seg_overlays,
            generate_separation_heatmaps,
        )
        
        np.random.seed(42)
        
        # 模拟训练历史
        history = {
            'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
            'val_loss': [1.1, 0.9, 0.75, 0.65, 0.55],
            'val_macro_f1': [0.3, 0.4, 0.5, 0.55, 0.6],
            'val_dice': [0.4, 0.5, 0.55, 0.6, 0.65],
        }
        
        # 模拟混淆矩阵
        cm = np.random.randint(0, 50, (10, 10))
        
        # 模拟图像和 mask
        images = np.random.rand(5, 3, 64, 64).astype(np.float32)
        masks = (np.random.rand(5, 1, 64, 64) > 0.5).astype(np.float32)
        heatmaps = np.random.rand(5, 8, 64, 64).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # 绘制混淆矩阵
            plot_confusion_matrix(cm, save_path=str(temp_dir / "cm.png"))
            assert (temp_dir / "cm.png").exists()
            
            # 绘制曲线
            plot_loss_curves(history, save_path=str(temp_dir / "loss.png"))
            assert (temp_dir / "loss.png").exists()
            
            plot_metric_curves(history, save_path=str(temp_dir / "metrics.png"))
            assert (temp_dir / "metrics.png").exists()
            
            # 生成 overlay
            seg_dir = temp_dir / "seg_overlays"
            generate_seg_overlays(images, masks, masks, output_dir=str(seg_dir))
            assert seg_dir.exists()
            
            # 生成热力图
            sep_dir = temp_dir / "separation_maps"
            generate_separation_heatmaps(images, heatmaps, output_dir=str(sep_dir))
            assert sep_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
