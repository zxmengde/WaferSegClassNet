# -*- coding: utf-8 -*-
"""
晶圆图谱分割解码器

U-Net 风格的解码器，支持跳跃连接
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SeparableConv2d


class TransposeConvBlock(nn.Module):
    """
    上采样卷积块
    
    结构: TransposeConv -> Concat(skip) -> Conv -> ReLU -> BN -> Dropout -> SepConv -> ReLU -> BN
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        # 转置卷积上采样
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        
        # 合并后的卷积
        self.conv1 = nn.Conv2d(
            out_channels + skip_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.sep_conv = SeparableConv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, C, H, W)
            skip: 跳跃连接特征 (B, C', H*2, W*2)
            
        Returns:
            上采样后的特征 (B, out_channels, H*2, W*2)
        """
        x = self.upsample(x)
        
        # 处理尺寸不匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.sep_conv(x))
        x = self.bn2(x)
        
        return x


class WaferDecoder(nn.Module):
    """
    晶圆图谱分割解码器
    
    U-Net 风格，使用跳跃连接
    """
    
    def __init__(
        self,
        encoder_channels: int = 64,
        base_channels: int = 8,
        num_classes: int = 1,
    ):
        super().__init__()
        
        # 解码器层（与编码器对称）
        # encoder_channels=64, skips=[8, 16, 16, 32]
        self.up1 = TransposeConvBlock(encoder_channels, base_channels * 4, base_channels * 4)  # 14->28
        self.up2 = TransposeConvBlock(base_channels * 4, base_channels * 2, base_channels * 2)  # 28->56
        self.up3 = TransposeConvBlock(base_channels * 2, base_channels * 2, base_channels * 2)  # 56->112
        self.up4 = TransposeConvBlock(base_channels * 2, base_channels, base_channels)  # 112->224
        
        # 输出层
        self.output = nn.Conv2d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, features: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 编码器输出特征 (B, C, H, W)
            skips: 跳跃连接特征列表 [s1, s2, s3, s4]
            
        Returns:
            分割 mask (B, num_classes, H, W)
        """
        # skips: [s1(8ch), s2(16ch), s3(16ch), s4(32ch)]
        s1, s2, s3, s4 = skips
        
        x = self.up1(features, s4)  # 64->32, concat with s4(32)
        x = self.up2(x, s3)         # 32->16, concat with s3(16)
        x = self.up3(x, s2)         # 16->16, concat with s2(16)
        x = self.up4(x, s1)         # 16->8, concat with s1(8)
        
        x = torch.sigmoid(self.output(x))
        
        return x


if __name__ == "__main__":
    # 测试解码器
    from encoder import WaferEncoder
    
    encoder = WaferEncoder()
    decoder = WaferDecoder(encoder_channels=64, base_channels=8)
    
    x = torch.randn(2, 3, 224, 224)
    features, skips = encoder(x)
    mask = decoder(features, skips)
    
    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Mask shape: {mask.shape}")
