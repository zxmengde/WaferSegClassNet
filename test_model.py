#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试模型是否可以正常工作
"""

import torch
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model():
    """测试模型前向传播"""
    print("=" * 60)
    print("模型测试")
    print("=" * 60)
    
    # 检查CUDA
    print(f"\n1. CUDA检查:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建模型
    print(f"\n2. 创建模型:")
    try:
        from src.models.multitask import WaferMultiTaskModel
        
        model = WaferMultiTaskModel(
            in_channels=3,
            base_channels=8,
            classification_classes=38,
            segmentation_classes=1,
            separation_enabled=False,
        )
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✅ 模型创建成功")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        return False
    
    # 测试前向传播
    print(f"\n3. 测试前向传播:")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 创建随机输入
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        
        print(f"   输入形状: {x.shape}")
        print(f"   设备: {device}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(x)
        
        print(f"   ✅ 前向传播成功")
        print(f"   分类输出: {outputs['cls_logits'].shape}")
        print(f"   分割输出: {outputs['seg_mask'].shape}")
        
        if 'sep_heatmaps' in outputs and outputs['sep_heatmaps'] is not None:
            print(f"   分离输出: {outputs['sep_heatmaps'].shape}")
        
    except Exception as e:
        print(f"   ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试损失计算
    print(f"\n4. 测试损失计算:")
    try:
        from src.models.losses import MultiTaskLoss
        
        criterion = MultiTaskLoss(
            classification_loss='cross_entropy',
            segmentation_loss='bce_dice',
            weights=(1.0, 1.0, 0.5),
            num_classes=38,
        )
        
        # 创建假标签（注意：键名必须与MultiTaskLoss.forward()期望的一致）
        cls_labels = torch.randint(0, 38, (batch_size,)).to(device)
        seg_labels = torch.randint(0, 2, (batch_size, 1, 224, 224)).float().to(device)
        
        # 准备targets字典（键名：label, mask, label_8）
        targets = {
            'label': cls_labels,      # 38类分类标签
            'mask': seg_labels,       # 分割掩码
        }
        
        # 计算损失
        loss_dict = criterion(outputs, targets)
        
        print(f"   ✅ 损失计算成功")
        print(f"   总损失: {loss_dict['total'].item():.4f}")
        print(f"   分类损失: {loss_dict['cls'].item():.4f}")
        print(f"   分割损失: {loss_dict['seg'].item():.4f}")
        
    except Exception as e:
        print(f"   ❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！模型可以正常工作。")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
