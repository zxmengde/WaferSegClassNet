# -*- coding: utf-8 -*-
"""
多任务模型测试

测试 WaferMultiTaskModel
"""

import pytest
import torch

from src.models.multitask import WaferMultiTaskModel, create_model


class TestWaferMultiTaskModel:
    """多任务模型测试类
    
    _Requirements: 3.1_
    """
    
    def test_forward_returns_required_keys(self):
        """测试 forward() 返回 {cls_logits, seg_mask, sep_heatmaps}"""
        model = WaferMultiTaskModel(separation_enabled=True)
        x = torch.randn(2, 3, 224, 224)
        outputs = model(x)
        
        # 验证返回的字典包含所有必需的键
        assert 'cls_logits' in outputs, "outputs should contain 'cls_logits'"
        assert 'seg_mask' in outputs, "outputs should contain 'seg_mask'"
        assert 'sep_heatmaps' in outputs, "outputs should contain 'sep_heatmaps'"
    
    def test_forward_output_shapes(self):
        """测试输出形状"""
        model = WaferMultiTaskModel(
            classification_classes=38,
            segmentation_classes=1,
            separation_enabled=True,
            separation_channels=8,
        )
        x = torch.randn(2, 3, 224, 224)
        outputs = model(x)
        
        assert outputs['cls_logits'].shape == (2, 38)
        assert outputs['seg_mask'].shape == (2, 1, 224, 224)
        assert outputs['sep_heatmaps'].shape == (2, 8, 224, 224)
    
    def test_forward_without_separation(self):
        """测试不启用分离头时的输出"""
        model = WaferMultiTaskModel(separation_enabled=False)
        x = torch.randn(2, 3, 224, 224)
        outputs = model(x)
        
        assert 'cls_logits' in outputs
        assert 'seg_mask' in outputs
        assert 'sep_heatmaps' not in outputs
    
    def test_8_class_classification(self):
        """测试8类分类"""
        model = WaferMultiTaskModel(classification_classes=8)
        x = torch.randn(2, 3, 224, 224)
        outputs = model(x)
        
        assert outputs['cls_logits'].shape == (2, 8)
    
    def test_freeze_unfreeze_encoder(self):
        """测试冻结/解冻编码器"""
        model = WaferMultiTaskModel()
        
        # 冻结编码器
        model.freeze_encoder()
        for param in model.encoder.parameters():
            assert not param.requires_grad
        
        # 解冻编码器
        model.unfreeze_encoder()
        for param in model.encoder.parameters():
            assert param.requires_grad
    
    def test_different_input_sizes(self):
        """测试不同输入尺寸"""
        model = WaferMultiTaskModel(separation_enabled=True)
        
        for size in [128, 224, 256]:
            x = torch.randn(2, 3, size, size)
            outputs = model(x)
            
            assert outputs['cls_logits'].shape == (2, 38)
            assert outputs['seg_mask'].shape == (2, 1, size, size)
            assert outputs['sep_heatmaps'].shape == (2, 8, size, size)


class TestCreateModel:
    """create_model 工厂函数测试"""
    
    def test_create_custom_model(self):
        """测试创建自定义模型"""
        model = create_model(
            encoder="custom",
            classification_classes=38,
            separation_enabled=True,
        )
        
        assert isinstance(model, WaferMultiTaskModel)
    
    def test_create_model_with_invalid_encoder(self):
        """测试无效编码器类型"""
        with pytest.raises(NotImplementedError):
            create_model(encoder="resnet18")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
