# -*- coding: utf-8 -*-
"""
多任务模型

整合编码器、解码器和任务头
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoder import WaferEncoder
from .decoder import WaferDecoder
from .heads import ClassificationHead, SeparationHead


class WaferMultiTaskModel(nn.Module):
    """
    晶圆图谱多任务模型
    
    包含:
    - 共享编码器
    - 分类头 (38 类 / 8 类)
    - 分割解码器 (U-Net 风格)
    - 分离头 (8 通道, E3 专用)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 8,
        classification_classes: int = 38,
        segmentation_classes: int = 1,
        separation_enabled: bool = False,
        separation_channels: int = 8,
    ):
        super().__init__()
        
        self.separation_enabled = separation_enabled
        
        # 编码器
        self.encoder = WaferEncoder(in_channels=in_channels, base_channels=base_channels)
        encoder_out_channels = self.encoder.out_channels  # 64
        
        # 分类头
        self.cls_head = ClassificationHead(
            in_channels=encoder_out_channels,
            num_classes=classification_classes,
        )
        
        # 分割解码器
        self.decoder = WaferDecoder(
            encoder_channels=encoder_out_channels,
            base_channels=base_channels,
            num_classes=segmentation_classes,
        )
        
        # 分离头（可选）
        if separation_enabled:
            self.sep_head = SeparationHead(
                in_channels=encoder_out_channels,
                num_components=separation_channels,
            )
        else:
            self.sep_head = None
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像 (B, 3, H, W)
            
        Returns:
            {
                'cls_logits': Tensor (B, num_classes),
                'seg_mask': Tensor (B, 1, H, W),
                'sep_heatmaps': Tensor (B, 8, H, W) (if enabled),
            }
        """
        # 编码
        features, skips = self.encoder(x)
        
        # 分类
        cls_logits = self.cls_head(features)
        
        # 分割
        seg_mask = self.decoder(features, skips)
        
        outputs = {
            'cls_logits': cls_logits,
            'seg_mask': seg_mask,
        }
        
        # 分离（可选）
        if self.separation_enabled and self.sep_head is not None:
            sep_heatmaps = self.sep_head(features)
            # 上采样到原始尺寸
            sep_heatmaps = nn.functional.interpolate(
                sep_heatmaps, size=x.shape[2:], mode='bilinear', align_corners=False
            )
            outputs['sep_heatmaps'] = sep_heatmaps
        
        return outputs
    
    def freeze_encoder(self):
        """冻结编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """解冻编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def load_encoder_weights(
        self,
        checkpoint_path: str,
        key_mapping: Optional[Dict] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        加载编码器预训练权重
        
        Args:
            checkpoint_path: 检查点路径
            key_mapping: key 映射规则
            output_dir: 输出目录
            
        Returns:
            加载统计信息
        """
        return self.encoder.load_pretrained(
            checkpoint_path=checkpoint_path,
            key_mapping=key_mapping,
            output_dir=output_dir,
        )


def create_model(
    encoder: str = "custom",
    classification_classes: int = 38,
    segmentation_classes: int = 1,
    separation_enabled: bool = False,
    separation_channels: int = 8,
    pretrained_weights: Optional[str] = None,
    key_mapping: Optional[Dict] = None,
    output_dir: Optional[str] = None,
) -> WaferMultiTaskModel:
    """
    创建多任务模型
    
    Args:
        encoder: 编码器类型 (custom | resnet18 | resnet34)
        classification_classes: 分类类别数
        segmentation_classes: 分割类别数
        separation_enabled: 是否启用分离头
        separation_channels: 分离通道数
        pretrained_weights: 预训练权重路径
        key_mapping: 权重 key 映射规则
        output_dir: 输出目录
        
    Returns:
        WaferMultiTaskModel 实例
    """
    if encoder == "custom":
        model = WaferMultiTaskModel(
            in_channels=3,
            base_channels=8,
            classification_classes=classification_classes,
            segmentation_classes=segmentation_classes,
            separation_enabled=separation_enabled,
            separation_channels=separation_channels,
        )
    else:
        # TODO: 支持 ResNet backbone
        raise NotImplementedError(f"Encoder {encoder} not implemented yet")
    
    # 加载预训练权重
    if pretrained_weights:
        model.load_encoder_weights(
            checkpoint_path=pretrained_weights,
            key_mapping=key_mapping,
            output_dir=output_dir,
        )
    
    return model


if __name__ == "__main__":
    # 测试模型
    model = WaferMultiTaskModel(
        classification_classes=38,
        segmentation_classes=1,
        separation_enabled=True,
    )
    
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Classification logits shape: {outputs['cls_logits'].shape}")
    print(f"Segmentation mask shape: {outputs['seg_mask'].shape}")
    print(f"Separation heatmaps shape: {outputs['sep_heatmaps'].shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
