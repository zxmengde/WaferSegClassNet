# -*- coding: utf-8 -*-
"""
数据增强测试

验证增强配置不包含黑名单操作
Requirements: 2.6
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentation import (
    WaferFriendlyAugmentation,
    WHITELIST_OPERATIONS,
    BLACKLIST_OPERATIONS,
    create_train_augmentation,
    create_val_augmentation,
    validate_augmentation_config,
)


# ============================================================================
# 白名单/黑名单验证测试
# ============================================================================

@pytest.mark.unit
class TestAugmentationWhitelist:
    """数据增强白名单/黑名单验证测试"""
    
    def test_whitelist_operations_defined(self):
        """白名单操作应已定义"""
        assert len(WHITELIST_OPERATIONS) > 0, "白名单操作不应为空"
        assert 'rotation' in WHITELIST_OPERATIONS
        assert 'flip' in WHITELIST_OPERATIONS
        assert 'morphological_noise' in WHITELIST_OPERATIONS
        assert 'color_jitter' in WHITELIST_OPERATIONS
    
    def test_blacklist_operations_defined(self):
        """黑名单操作应已定义"""
        assert len(BLACKLIST_OPERATIONS) > 0, "黑名单操作不应为空"
        assert 'random_resized_crop' in BLACKLIST_OPERATIONS
        assert 'center_crop' in BLACKLIST_OPERATIONS
        assert 'random_crop' in BLACKLIST_OPERATIONS
    
    def test_whitelist_blacklist_no_overlap(self):
        """白名单和黑名单不应有重叠"""
        overlap = WHITELIST_OPERATIONS & BLACKLIST_OPERATIONS
        assert len(overlap) == 0, f"白名单和黑名单有重叠: {overlap}"
    
    def test_blacklist_operation_rejected(self):
        """黑名单操作应被拒绝"""
        for op in BLACKLIST_OPERATIONS:
            with pytest.raises(ValueError) as exc_info:
                WaferFriendlyAugmentation({op: {}})
            assert "禁止使用" in str(exc_info.value) or op in str(exc_info.value)
    
    def test_whitelist_operation_accepted(self):
        """白名单操作应被接受"""
        # 测试各种白名单操作
        valid_configs = [
            {'rotation': [0, 90, 180, 270]},
            {'flip': True},
            {'horizontal_flip': True},
            {'vertical_flip': True},
            {'morphological_noise': 0.1},
            {'color_jitter': {'brightness': 0.1}},
            {'gaussian_noise': 0.01},
        ]
        
        for config in valid_configs:
            aug = WaferFriendlyAugmentation(config)
            is_valid, errors = aug.validate_config()
            assert is_valid, f"白名单配置 {config} 应被接受，但有错误: {errors}"
    
    def test_validate_augmentation_config_rejects_blacklist(self):
        """validate_augmentation_config应拒绝黑名单操作"""
        for op in BLACKLIST_OPERATIONS:
            is_valid, errors = validate_augmentation_config({op: {}})
            assert not is_valid, f"黑名单操作 {op} 应被拒绝"
            assert len(errors) > 0
    
    def test_validate_augmentation_config_accepts_whitelist(self):
        """validate_augmentation_config应接受白名单操作"""
        config = {
            'rotation': [0, 90],
            'flip': True,
            'morphological_noise': 0.1,
        }
        is_valid, errors = validate_augmentation_config(config)
        assert is_valid, f"白名单配置应被接受，但有错误: {errors}"


@pytest.mark.unit
class TestAugmentationFunctionality:
    """数据增强功能测试"""
    
    def test_train_augmentation_creates_valid_instance(self):
        """训练增强应创建有效实例"""
        aug = create_train_augmentation()
        assert aug is not None
        is_valid, errors = aug.validate_config()
        assert is_valid, f"默认训练增强配置应有效: {errors}"
    
    def test_val_augmentation_creates_valid_instance(self):
        """验证增强应创建有效实例"""
        aug = create_val_augmentation()
        assert aug is not None
        is_valid, errors = aug.validate_config()
        assert is_valid, f"默认验证增强配置应有效: {errors}"
    
    def test_augmentation_preserves_image_shape(self):
        """增强应保持图像形状"""
        aug = create_train_augmentation()
        
        # 测试不同尺寸
        for size in [(224, 224), (128, 128), (256, 256)]:
            image = np.random.rand(size[0], size[1], 3).astype(np.float32)
            mask = np.random.randint(0, 2, (size[0], size[1], 1)).astype(np.float32)
            
            result = aug(image, mask)
            
            assert result['image'].shape == image.shape, \
                f"图像形状应保持不变: {result['image'].shape} != {image.shape}"
            assert result['mask'].shape == mask.shape, \
                f"Mask形状应保持不变: {result['mask'].shape} != {mask.shape}"
    
    def test_augmentation_preserves_dtype(self):
        """增强应保持数据类型"""
        aug = create_train_augmentation()
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        mask = np.random.randint(0, 2, (224, 224, 1)).astype(np.float32)
        
        result = aug(image, mask)
        
        assert result['image'].dtype == image.dtype, \
            f"图像dtype应保持不变: {result['image'].dtype} != {image.dtype}"
        assert result['mask'].dtype == mask.dtype, \
            f"Mask dtype应保持不变: {result['mask'].dtype} != {mask.dtype}"
    
    def test_augmentation_handles_none_mask(self):
        """增强应处理None mask"""
        aug = create_train_augmentation()
        
        image = np.random.rand(224, 224, 3).astype(np.float32)
        
        result = aug(image, None)
        
        assert result['image'] is not None
        assert result['mask'] is None
    
    def test_rotation_angles_applied(self):
        """旋转角度应被应用"""
        # 使用固定旋转角度
        aug = WaferFriendlyAugmentation({
            'rotation': [90],
            'flip': False,
        })
        
        # 创建非对称图像以验证旋转
        image = np.zeros((4, 4, 1), dtype=np.float32)
        image[0, 0, 0] = 1.0  # 左上角标记
        
        result = aug(image, None)
        
        # 旋转90度后，左上角应变为左下角
        assert result['image'][3, 0, 0] == 1.0, "90度旋转应将左上角移到左下角"
    
    def test_get_config_returns_dict(self):
        """get_config应返回配置字典"""
        aug = create_train_augmentation()
        config = aug.get_config()
        
        assert isinstance(config, dict)
        assert 'rotation' in config
        assert 'flip' in config


@pytest.mark.unit
class TestCropRestrictions:
    """裁剪限制测试"""
    
    def test_large_crop_rejected(self):
        """大于20%的裁剪应被拒绝"""
        # 注意：当前实现通过黑名单直接拒绝crop操作
        # 如果未来支持小比例裁剪，需要验证min_scale >= 0.8
        
        # 测试random_crop被拒绝
        with pytest.raises(ValueError):
            WaferFriendlyAugmentation({'random_crop': {'min_scale': 0.5}})
        
        # 测试center_crop被拒绝
        with pytest.raises(ValueError):
            WaferFriendlyAugmentation({'center_crop': {'size': 100}})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
