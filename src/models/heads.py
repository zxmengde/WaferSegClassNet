# -*- coding: utf-8 -*-
"""
任务头模块

包含分类头和分离头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    分类头
    
    支持 38 类单标签和 8 类多标签分类
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 38,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 编码器特征 (B, C, H, W)
            
        Returns:
            分类 logits (B, num_classes)
        """
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SeparationHead(nn.Module):
    """
    成分分离头
    
    输出 8 通道热力图，每个通道对应一种基础缺陷
    """
    
    def __init__(
        self,
        in_channels: int,
        num_components: int = 8,
        hidden_channels: int = 32,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, num_components, kernel_size=1)
        self.num_components = num_components
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 编码器特征 (B, C, H, W)
            
        Returns:
            分离热力图 (B, 8, H, W)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.sigmoid(self.conv2(x))
        return x


if __name__ == "__main__":
    # 测试
    cls_head = ClassificationHead(in_channels=64, num_classes=38)
    sep_head = SeparationHead(in_channels=64, num_components=8)
    
    x = torch.randn(2, 64, 14, 14)
    
    cls_out = cls_head(x)
    sep_out = sep_head(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Classification output shape: {cls_out.shape}")
    print(f"Separation output shape: {sep_out.shape}")
