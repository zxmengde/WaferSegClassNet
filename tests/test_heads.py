# -*- coding: utf-8 -*-
"""
任务头测试

测试 ClassificationHead 和 SeparationHead
"""

import pytest
import torch

from src.models.heads import ClassificationHead, SeparationHead


class TestClassificationHead:
    """分类头测试类"""
    
    def test_classification_head_38_classes(self):
        """测试38类分类头输出形状"""
        head = ClassificationHead(in_channels=64, num_classes=38)
        x = torch.randn(2, 64, 14, 14)
        out = head(x)
        
        assert out.shape == (2, 38)
    
    def test_classification_head_8_classes(self):
        """测试8类分类头输出形状"""
        head = ClassificationHead(in_channels=64, num_classes=8)
        x = torch.randn(2, 64, 14, 14)
        out = head(x)
        
        assert out.shape == (2, 8)
    
    def test_classification_head_different_input_sizes(self):
        """测试不同输入尺寸"""
        head = ClassificationHead(in_channels=64, num_classes=38)
        
        # 不同的空间尺寸应该都能工作（因为使用了GAP）
        for size in [(7, 7), (14, 14), (28, 28)]:
            x = torch.randn(2, 64, *size)
            out = head(x)
            assert out.shape == (2, 38)


class TestSeparationHead:
    """分离头测试类
    
    验证输出为8通道
    _Requirements: 6.1_
    """
    
    def test_separation_head_output_8_channels(self):
        """测试分离头输出为8通道
        
        验证输出为8通道，每个通道对应一种基础缺陷类型
        """
        head = SeparationHead(in_channels=64, num_components=8)
        x = torch.randn(2, 64, 14, 14)
        out = head(x)
        
        # 验证输出通道数为8
        assert out.shape[1] == 8, f"Expected 8 channels, got {out.shape[1]}"
        assert out.shape == (2, 8, 14, 14)
    
    def test_separation_head_output_range(self):
        """测试分离头输出范围在[0, 1]"""
        head = SeparationHead(in_channels=64, num_components=8)
        x = torch.randn(2, 64, 14, 14)
        out = head(x)
        
        # 输出应该在[0, 1]范围内（因为使用了sigmoid）
        assert out.min() >= 0.0
        assert out.max() <= 1.0
    
    def test_separation_head_different_input_channels(self):
        """测试不同输入通道数"""
        for in_channels in [32, 64, 128]:
            head = SeparationHead(in_channels=in_channels, num_components=8)
            x = torch.randn(2, in_channels, 14, 14)
            out = head(x)
            
            assert out.shape == (2, 8, 14, 14)
    
    def test_separation_head_preserves_spatial_size(self):
        """测试分离头保持空间尺寸"""
        head = SeparationHead(in_channels=64, num_components=8)
        
        for size in [(7, 7), (14, 14), (28, 28)]:
            x = torch.randn(2, 64, *size)
            out = head(x)
            
            assert out.shape[2:] == size, f"Expected spatial size {size}, got {out.shape[2:]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
