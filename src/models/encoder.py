# -*- coding: utf-8 -*-
"""
晶圆图谱编码器

基于原有 TensorFlow 设计迁移到 PyTorch
支持自定义轻量级架构和 ResNet backbone
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class SeparableConv2d(nn.Module):
    """深度可分离卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvBlock(nn.Module):
    """
    下采样卷积块
    
    结构: Conv -> ReLU -> BN -> Dropout -> SepConv -> ReLU -> BN -> MaxPool
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_stride: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.sep_conv = SeparableConv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_stride)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            skip: 用于跳跃连接的特征
            out: 下采样后的特征
        """
        skip = F.relu(self.conv1(x))
        x = self.bn1(skip)
        x = self.dropout(x)
        x = F.relu(self.sep_conv(x))
        x = self.bn2(x)
        out = self.pool(x)
        return skip, out


class GapConvBlock(nn.Module):
    """
    使用平均池化的下采样卷积块
    
    结构: Conv -> ReLU -> BN -> Dropout -> SepConv -> ReLU -> BN -> AvgPool
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_stride: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.sep_conv = SeparableConv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(pool_stride)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = F.relu(self.conv1(x))
        x = self.bn1(skip)
        x = self.dropout(x)
        x = F.relu(self.sep_conv(x))
        x = self.bn2(x)
        out = self.pool(x)
        return skip, out


class TerminalConvBlock(nn.Module):
    """
    终端卷积块（无池化）
    
    结构: Conv -> ReLU -> BN -> Dropout -> SepConv -> ReLU -> BN
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.sep_conv = SeparableConv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.sep_conv(x))
        x = self.bn2(x)
        return x


class WaferEncoder(nn.Module):
    """
    晶圆图谱特征编码器
    
    基于原有 TensorFlow 设计，支持:
    - 自定义轻量级架构
    - SSL 预训练权重加载
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 8):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # 编码器层
        self.block1 = ConvBlock(in_channels, base_channels)           # 224 -> 112
        self.block2 = ConvBlock(base_channels, base_channels * 2)     # 112 -> 56
        self.block3 = ConvBlock(base_channels * 2, base_channels * 2) # 56 -> 28
        self.block4 = GapConvBlock(base_channels * 2, base_channels * 4)  # 28 -> 14
        self.terminal = TerminalConvBlock(base_channels * 4, base_channels * 8)  # 14 -> 14
        
        # 输出通道数
        self.out_channels = base_channels * 8
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: 输入图像 (B, 3, H, W)
            
        Returns:
            features: 编码特征 (B, C, H', W')
            skips: 跳跃连接特征列表 [s1, s2, s3, s4]
        """
        skips = []
        
        s1, x = self.block1(x)
        skips.append(s1)
        
        s2, x = self.block2(x)
        skips.append(s2)
        
        s3, x = self.block3(x)
        skips.append(s3)
        
        s4, x = self.block4(x)
        skips.append(s4)
        
        features = self.terminal(x)
        
        return features, skips
    
    def load_pretrained(
        self,
        checkpoint_path: str,
        key_mapping: Optional[Dict] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        加载预训练权重
        
        Args:
            checkpoint_path: 检查点路径
            key_mapping: 可选的 key 映射规则
                - strip_prefix: 要剥离的前缀列表
                - extract_subtree: 要提取的子树
                - use_encoder_state_dict: 是否使用 encoder_state_dict（SSL checkpoint）
            output_dir: 输出目录（用于保存 weight_loading.json）
            
        Returns:
            加载统计信息
        """
        key_mapping = key_mapping or {}
        strip_prefixes = key_mapping.get('strip_prefix', [])
        extract_subtree = key_mapping.get('extract_subtree', None)
        use_encoder_state_dict = key_mapping.get('use_encoder_state_dict', False)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 提取状态字典
        # 优先使用 encoder_state_dict（SSL checkpoint 格式）
        if use_encoder_state_dict and 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
            logger.info("Using encoder_state_dict from SSL checkpoint")
        elif 'encoder_state_dict' in checkpoint:
            # 自动检测 SSL checkpoint 格式
            state_dict = checkpoint['encoder_state_dict']
            logger.info("Auto-detected encoder_state_dict from SSL checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 提取子树
        if extract_subtree:
            prefix = extract_subtree + '.'
            state_dict = {
                k[len(prefix):]: v for k, v in state_dict.items()
                if k.startswith(prefix)
            }
        
        # 剥离前缀
        for prefix in strip_prefixes:
            state_dict = {
                k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in state_dict.items()
            }
        
        # 获取当前模型的状态字典
        model_state = self.state_dict()
        
        # 统计
        matched = 0
        missing = 0
        unexpected = 0
        shape_mismatch = 0
        
        matched_keys = []
        missing_keys = []
        unexpected_keys = []
        shape_mismatch_keys = []
        
        # 匹配权重
        for key in model_state.keys():
            if key in state_dict:
                if model_state[key].shape == state_dict[key].shape:
                    model_state[key] = state_dict[key]
                    matched += 1
                    matched_keys.append(key)
                else:
                    shape_mismatch += 1
                    shape_mismatch_keys.append(key)
            else:
                missing += 1
                missing_keys.append(key)
        
        for key in state_dict.keys():
            if key not in model_state:
                unexpected += 1
                unexpected_keys.append(key)
        
        # 加载权重
        self.load_state_dict(model_state)
        
        # 统计信息
        stats = {
            'matched': matched,
            'missing': missing,
            'unexpected': unexpected,
            'shape_mismatch': shape_mismatch,
            'ignored_prefixes': strip_prefixes,
            'matched_keys': matched_keys,
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'shape_mismatch_keys': shape_mismatch_keys,
        }
        
        # 保存统计信息
        if output_dir:
            output_path = Path(output_dir) / 'weight_loading.json'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Weight loading stats saved to {output_path}")
        
        logger.info(f"Loaded pretrained weights: matched={matched}, missing={missing}, "
                   f"unexpected={unexpected}, shape_mismatch={shape_mismatch}")
        
        return stats


class ProjectionHead(nn.Module):
    """对比学习投影头"""
    
    def __init__(self, in_features: int, out_features: int = 128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


if __name__ == "__main__":
    # 测试编码器
    encoder = WaferEncoder()
    x = torch.randn(2, 3, 224, 224)
    features, skips = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Skip shapes: {[s.shape for s in skips]}")
