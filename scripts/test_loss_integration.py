# -*- coding: utf-8 -*-
"""测试损失函数与模型的集成"""

import torch
from src.models import WaferMultiTaskModel, create_loss, create_loss_from_config
from src.config_schema import LossConfig

def test_integration():
    print("=" * 60)
    print("Model + Loss Integration Test")
    print("=" * 60)
    
    # 创建模型
    model = WaferMultiTaskModel(
        classification_classes=38,
        segmentation_classes=1,
        separation_enabled=True,
    )
    
    # 创建损失函数
    loss_fn = create_loss(
        classification_loss='focal',
        segmentation_loss='bce_dice',
        separation_loss='kl_divergence',
        weights=[1.0, 1.0, 0.5],
        focal_gamma=2.0,
    )
    
    # 模拟输入
    x = torch.randn(2, 3, 64, 64)
    
    # 前向传播
    outputs = model(x)
    
    # 模拟标签
    targets = {
        'cls_label': torch.randint(0, 38, (2,)),
        'seg_mask': torch.randint(0, 2, (2, 1, 64, 64)).float(),
        'sep_target': torch.softmax(torch.randn(2, 8, 64, 64), dim=1),
    }
    
    # 计算损失
    losses = loss_fn(outputs, targets)
    
    print(f"Input shape: {x.shape}")
    print(f"Classification logits: {outputs['cls_logits'].shape}")
    print(f"Segmentation mask: {outputs['seg_mask'].shape}")
    print(f"Separation heatmaps: {outputs['sep_heatmaps'].shape}")
    print()
    print("Losses:")
    print(f"  Total: {losses['total'].item():.4f}")
    print(f"  Classification: {losses['cls'].item():.4f}")
    print(f"  Segmentation: {losses['seg'].item():.4f}")
    print(f"  Separation: {losses['sep'].item():.4f}")
    
    # 测试从配置创建
    print()
    print("Testing create_loss_from_config...")
    loss_config = LossConfig(
        classification="class_balanced",
        segmentation="dice",
        separation="mse",
        weights=[1.0, 0.5, 0.3],
    )
    loss_fn2 = create_loss_from_config(loss_config, num_classes=38, num_per_class=[100]*38)
    losses2 = loss_fn2(outputs, targets)
    print(f"  Total (class_balanced + dice): {losses2['total'].item():.4f}")
    
    print()
    print("=" * 60)
    print("All integration tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_integration()
