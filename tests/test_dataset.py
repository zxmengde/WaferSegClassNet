# -*- coding: utf-8 -*-
"""
数据集属性测试

Property 2: 分割mask二值性
Validates: Requirements 2.4
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MixedWM38Dataset, PseudoMaskGenerator


# ============================================================================
# 策略定义
# ============================================================================

# 有效的数据集索引（需要在测试中动态确定）
def valid_dataset_index(dataset):
    """生成有效的数据集索引策略"""
    return st.integers(min_value=0, max_value=len(dataset) - 1)


# ============================================================================
# Property 2: 分割mask二值性
# **Feature: mixed-wm38-recognition, Property 2: 分割mask二值性**
# **Validates: Requirements 2.4**
# ============================================================================

@pytest.mark.property
class TestMaskBinaryProperty:
    """Property 2: 分割mask二值性测试"""
    
    @pytest.fixture(scope="class")
    def dataset(self):
        """创建测试数据集"""
        return MixedWM38Dataset(
            data_root="data/processed",
            split="train",
            debug=True,
            max_per_class=3,
        )
    
    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_mask_values_are_binary(self, data, dataset):
        """
        **Feature: mixed-wm38-recognition, Property 2: 分割mask二值性**
        **Validates: Requirements 2.4**
        
        对于任意加载的分割mask，其像素值应仅包含0和1两种值。
        """
        assume(len(dataset) > 0)
        
        idx = data.draw(st.integers(min_value=0, max_value=len(dataset) - 1))
        sample = dataset[idx]
        mask = sample['mask']
        
        # 检查mask值是否为二值（0或1）
        unique_values = torch.unique(mask)
        for val in unique_values:
            assert val.item() in [0.0, 1.0], \
                f"Mask包含非二值像素: {val.item()}"
    
    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_mask_shape_matches_image(self, data, dataset):
        """
        **Feature: mixed-wm38-recognition, Property 2: 分割mask二值性**
        **Validates: Requirements 2.4**
        
        对于任意加载的分割mask，其空间尺寸应与输入图像一致。
        """
        assume(len(dataset) > 0)
        
        idx = data.draw(st.integers(min_value=0, max_value=len(dataset) - 1))
        sample = dataset[idx]
        
        image = sample['image']
        mask = sample['mask']
        
        # 图像形状: (C, H, W)
        # Mask形状: (1, H, W)
        assert mask.shape[1] == image.shape[1], \
            f"Mask高度{mask.shape[1]}与图像高度{image.shape[1]}不匹配"
        assert mask.shape[2] == image.shape[2], \
            f"Mask宽度{mask.shape[2]}与图像宽度{image.shape[2]}不匹配"
    
    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_mask_has_single_channel(self, data, dataset):
        """
        **Feature: mixed-wm38-recognition, Property 2: 分割mask二值性**
        **Validates: Requirements 2.4**
        
        对于任意加载的分割mask，应为单通道。
        """
        assume(len(dataset) > 0)
        
        idx = data.draw(st.integers(min_value=0, max_value=len(dataset) - 1))
        sample = dataset[idx]
        mask = sample['mask']
        
        assert mask.shape[0] == 1, \
            f"Mask应为单通道，实际为{mask.shape[0]}通道"
    
    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_mask_dtype_is_float(self, data, dataset):
        """
        **Feature: mixed-wm38-recognition, Property 2: 分割mask二值性**
        **Validates: Requirements 2.4**
        
        对于任意加载的分割mask，数据类型应为float。
        """
        assume(len(dataset) > 0)
        
        idx = data.draw(st.integers(min_value=0, max_value=len(dataset) - 1))
        sample = dataset[idx]
        mask = sample['mask']
        
        assert mask.dtype == torch.float32, \
            f"Mask数据类型应为float32，实际为{mask.dtype}"


@pytest.mark.property
class TestPseudoMaskGenerator:
    """伪Mask生成器测试"""
    
    @given(
        threshold=st.integers(min_value=1, max_value=254),
        kernel_size=st.integers(min_value=1, max_value=7),
    )
    @settings(max_examples=50, deadline=None)
    def test_pseudo_mask_is_binary(self, threshold, kernel_size):
        """
        **Feature: mixed-wm38-recognition, Property 2: 分割mask二值性**
        **Validates: Requirements 2.4**
        
        对于任意配置的伪mask生成器，生成的mask应为二值。
        """
        generator = PseudoMaskGenerator({
            'threshold': threshold,
            'morphology_kernel': kernel_size,
        })
        
        # 创建随机测试图像
        test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        mask = generator.generate(test_image)
        
        # 检查mask值是否为二值（0或255）
        unique_values = np.unique(mask)
        for val in unique_values:
            assert val in [0, 255], \
                f"伪Mask包含非二值像素: {val}"
    
    @given(
        height=st.integers(min_value=32, max_value=512),
        width=st.integers(min_value=32, max_value=512),
    )
    @settings(max_examples=50, deadline=None)
    def test_pseudo_mask_shape_matches_input(self, height, width):
        """
        **Feature: mixed-wm38-recognition, Property 2: 分割mask二值性**
        **Validates: Requirements 2.4**
        
        对于任意尺寸的输入图像，生成的伪mask尺寸应与输入一致。
        """
        generator = PseudoMaskGenerator()
        
        # 创建指定尺寸的测试图像
        test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        mask = generator.generate(test_image)
        
        # 检查形状（生成器输出为 (H, W, 1)）
        assert mask.shape[0] == height, \
            f"伪Mask高度{mask.shape[0]}与输入高度{height}不匹配"
        assert mask.shape[1] == width, \
            f"伪Mask宽度{mask.shape[1]}与输入宽度{width}不匹配"
        assert mask.shape[2] == 1, \
            f"伪Mask应为单通道，实际为{mask.shape[2]}通道"


# ============================================================================
# 单元测试
# ============================================================================

@pytest.mark.unit
class TestDatasetBasics:
    """数据集基础功能测试"""
    
    def test_dataset_loads_successfully(self):
        """数据集应能成功加载"""
        dataset = MixedWM38Dataset(
            data_root="data/processed",
            split="train",
            debug=True,
            max_per_class=2,
        )
        assert len(dataset) > 0, "数据集不应为空"
    
    def test_single_label_mode(self):
        """单标签模式应返回正确的标签格式"""
        dataset = MixedWM38Dataset(
            data_root="data/processed",
            split="train",
            classification_mode="single_label",
            debug=True,
            max_per_class=2,
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            assert sample['label'].dtype == torch.long, \
                "单标签模式下label应为long类型"
    
    def test_multi_label_mode(self):
        """多标签模式应返回正确的标签格式"""
        dataset = MixedWM38Dataset(
            data_root="data/processed",
            split="train",
            classification_mode="multi_label",
            debug=True,
            max_per_class=2,
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            assert sample['label'].dtype == torch.float32, \
                "多标签模式下label应为float32类型"
            assert sample['label'].shape == (8,), \
                f"多标签应为8维，实际为{sample['label'].shape}"
    
    def test_debug_mode_limits_samples(self):
        """Debug模式应限制每类样本数"""
        max_per_class = 2
        dataset = MixedWM38Dataset(
            data_root="data/processed",
            split="train",
            debug=True,
            max_per_class=max_per_class,
        )
        
        class_counts = dataset.get_class_counts()
        for cls, count in class_counts.items():
            assert count <= max_per_class, \
                f"类{cls}样本数{count}超过限制{max_per_class}"
    
    def test_sample_contains_required_keys(self):
        """样本应包含所有必需的键"""
        dataset = MixedWM38Dataset(
            data_root="data/processed",
            split="train",
            debug=True,
            max_per_class=2,
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            required_keys = ['image', 'mask', 'label_38', 'label_8', 'label']
            for key in required_keys:
                assert key in sample, f"样本缺少必需的键: {key}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ============================================================================
# 数据加载器测试
# ============================================================================

@pytest.mark.unit
class TestDataLoader:
    """数据加载器测试"""
    
    def test_get_dataloaders_returns_three_loaders(self):
        """get_dataloaders应返回三个数据加载器"""
        from src.data.dataloader import get_dataloaders
        
        train_loader, val_loader, test_loader = get_dataloaders(
            data_root="data/processed",
            batch_size=8,
            debug=True,
            max_per_class=2,
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
    
    def test_debug_mode_reduces_samples(self):
        """Debug模式应减少样本数"""
        from src.data.dataloader import get_dataloaders
        
        # Debug模式
        train_debug, _, _ = get_dataloaders(
            data_root="data/processed",
            batch_size=8,
            debug=True,
            max_per_class=2,
        )
        
        # 非Debug模式
        train_full, _, _ = get_dataloaders(
            data_root="data/processed",
            batch_size=8,
            debug=False,
        )
        
        # Debug模式应该有更少的批次
        assert len(train_debug) <= len(train_full)
    
    def test_batch_contains_required_keys(self):
        """批次应包含所有必需的键"""
        from src.data.dataloader import get_dataloaders
        
        train_loader, _, _ = get_dataloaders(
            data_root="data/processed",
            batch_size=4,
            debug=True,
            max_per_class=2,
        )
        
        for batch in train_loader:
            required_keys = ['image', 'mask', 'label_38', 'label_8', 'label']
            for key in required_keys:
                assert key in batch, f"批次缺少必需的键: {key}"
            break
    
    def test_get_class_weights(self):
        """get_class_weights应返回权重张量"""
        from src.data.dataloader import get_dataloaders, get_class_weights
        
        train_loader, _, _ = get_dataloaders(
            data_root="data/processed",
            batch_size=8,
            debug=True,
            max_per_class=2,
        )
        
        weights = get_class_weights(train_loader)
        
        assert weights is not None
        assert len(weights) == 38  # 38类
