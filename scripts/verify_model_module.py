# -*- coding: utf-8 -*-
"""
模型模块验证脚本

验证所有模型组件的前向传播和反向传播
Checkpoint 7: 模型模块验证
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.encoder import WaferEncoder
from src.models.decoder import WaferDecoder
from src.models.heads import ClassificationHead, SeparationHead
from src.models.multitask import WaferMultiTaskModel, create_model
from src.models.losses import (
    DiceLoss, BCEDiceLoss, FocalLoss, ClassBalancedLoss,
    MultiTaskLoss, create_loss, create_loss_from_config
)


def print_section(title: str):
    """打印分节标题"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_encoder():
    """测试编码器前向传播"""
    print_section("1. 编码器测试 (WaferEncoder)")
    
    encoder = WaferEncoder(in_channels=3, base_channels=8)
    x = torch.randn(2, 3, 224, 224)
    
    features, skips = encoder(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  特征形状: {features.shape}")
    print(f"  跳跃连接形状: {[s.shape for s in skips]}")
    print(f"  输出通道数: {encoder.out_channels}")
    
    # 验证形状
    assert features.shape == (2, 64, 14, 14), f"特征形状错误: {features.shape}"
    assert len(skips) == 4, f"跳跃连接数量错误: {len(skips)}"
    
    print("  ✓ 编码器测试通过")
    return True


def test_decoder():
    """测试解码器前向传播"""
    print_section("2. 解码器测试 (WaferDecoder)")
    
    encoder = WaferEncoder()
    decoder = WaferDecoder(encoder_channels=64, base_channels=8, num_classes=1)
    
    x = torch.randn(2, 3, 224, 224)
    features, skips = encoder(x)
    mask = decoder(features, skips)
    
    print(f"  输入形状: {x.shape}")
    print(f"  编码特征形状: {features.shape}")
    print(f"  输出 mask 形状: {mask.shape}")
    print(f"  输出值范围: [{mask.min().item():.4f}, {mask.max().item():.4f}]")
    
    # 验证形状和值范围
    assert mask.shape == (2, 1, 224, 224), f"Mask 形状错误: {mask.shape}"
    assert 0 <= mask.min() <= mask.max() <= 1, "Mask 值范围应在 [0, 1]"
    
    print("  ✓ 解码器测试通过")
    return True


def test_classification_head():
    """测试分类头"""
    print_section("3. 分类头测试 (ClassificationHead)")
    
    head_38 = ClassificationHead(in_channels=64, num_classes=38)
    head_8 = ClassificationHead(in_channels=64, num_classes=8)
    
    x = torch.randn(2, 64, 14, 14)
    
    out_38 = head_38(x)
    out_8 = head_8(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  38 类输出形状: {out_38.shape}")
    print(f"  8 类输出形状: {out_8.shape}")
    
    assert out_38.shape == (2, 38), f"38 类输出形状错误: {out_38.shape}"
    assert out_8.shape == (2, 8), f"8 类输出形状错误: {out_8.shape}"
    
    print("  ✓ 分类头测试通过")
    return True


def test_separation_head():
    """测试分离头"""
    print_section("4. 分离头测试 (SeparationHead)")
    
    head = SeparationHead(in_channels=64, num_components=8)
    x = torch.randn(2, 64, 14, 14)
    
    out = head(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  输出值范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
    
    assert out.shape == (2, 8, 14, 14), f"分离头输出形状错误: {out.shape}"
    assert 0 <= out.min() <= out.max() <= 1, "分离头输出值范围应在 [0, 1]"
    
    print("  ✓ 分离头测试通过")
    return True


def test_multitask_model():
    """测试多任务模型"""
    print_section("5. 多任务模型测试 (WaferMultiTaskModel)")
    
    # 测试带分离头的模型
    model = WaferMultiTaskModel(
        classification_classes=38,
        segmentation_classes=1,
        separation_enabled=True,
        separation_channels=8,
    )
    
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  分类 logits 形状: {outputs['cls_logits'].shape}")
    print(f"  分割 mask 形状: {outputs['seg_mask'].shape}")
    print(f"  分离热力图形状: {outputs['sep_heatmaps'].shape}")
    
    assert outputs['cls_logits'].shape == (2, 38)
    assert outputs['seg_mask'].shape == (2, 1, 224, 224)
    assert outputs['sep_heatmaps'].shape == (2, 8, 224, 224)
    
    # 测试不带分离头的模型
    model_no_sep = WaferMultiTaskModel(separation_enabled=False)
    outputs_no_sep = model_no_sep(x)
    
    assert 'sep_heatmaps' not in outputs_no_sep
    print("  ✓ 无分离头模型测试通过")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    print("  ✓ 多任务模型测试通过")
    return True


def test_loss_functions():
    """测试损失函数"""
    print_section("6. 损失函数测试")
    
    batch_size = 4
    num_classes = 38
    
    # 模拟输出和标签
    outputs = {
        "cls_logits": torch.randn(batch_size, num_classes),
        "seg_mask": torch.randn(batch_size, 1, 64, 64),
        "sep_heatmaps": torch.randn(batch_size, 8, 64, 64),
    }
    targets = {
        "cls_label": torch.randint(0, num_classes, (batch_size,)),
        "seg_mask": torch.randint(0, 2, (batch_size, 1, 64, 64)).float(),
        "sep_target": torch.softmax(torch.randn(batch_size, 8, 64, 64), dim=1),
    }
    
    # 测试各种损失函数
    print("  测试 DiceLoss...")
    dice_loss = DiceLoss()
    pred_seg = torch.sigmoid(outputs["seg_mask"])
    loss = dice_loss(pred_seg, targets["seg_mask"])
    assert 0 <= loss.item() <= 1, f"DiceLoss 值范围错误: {loss.item()}"
    print(f"    DiceLoss: {loss.item():.4f}")
    
    print("  测试 BCEDiceLoss...")
    bce_dice = BCEDiceLoss()
    loss = bce_dice(outputs["seg_mask"], targets["seg_mask"])
    assert loss.item() > 0, f"BCEDiceLoss 应为正数: {loss.item()}"
    print(f"    BCEDiceLoss: {loss.item():.4f}")
    
    print("  测试 FocalLoss...")
    focal = FocalLoss(gamma=2.0)
    loss = focal(outputs["cls_logits"], targets["cls_label"])
    assert loss.item() > 0, f"FocalLoss 应为正数: {loss.item()}"
    print(f"    FocalLoss: {loss.item():.4f}")
    
    print("  测试 ClassBalancedLoss...")
    num_per_class = [100] * num_classes
    cb_loss = ClassBalancedLoss(num_per_class=num_per_class)
    loss = cb_loss(outputs["cls_logits"], targets["cls_label"])
    assert loss.item() > 0, f"ClassBalancedLoss 应为正数: {loss.item()}"
    print(f"    ClassBalancedLoss: {loss.item():.4f}")
    
    print("  测试 MultiTaskLoss...")
    mt_loss = MultiTaskLoss(
        classification_loss="cross_entropy",
        segmentation_loss="bce_dice",
        weights=[1.0, 1.0, 0.5],
    )
    losses = mt_loss(outputs, targets)
    assert "total" in losses
    assert "cls" in losses
    assert "seg" in losses
    print(f"    Total: {losses['total'].item():.4f}")
    print(f"    Classification: {losses['cls'].item():.4f}")
    print(f"    Segmentation: {losses['seg'].item():.4f}")
    
    print("  ✓ 损失函数测试通过")
    return True


def test_backward_pass():
    """测试反向传播"""
    print_section("7. 反向传播测试")
    
    model = WaferMultiTaskModel(
        classification_classes=38,
        separation_enabled=True,
    )
    
    loss_fn = MultiTaskLoss(
        classification_loss="focal",
        segmentation_loss="bce_dice",
        weights=[1.0, 1.0, 0.5],
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 前向传播
    x = torch.randn(2, 3, 128, 128)
    outputs = model(x)
    
    targets = {
        "cls_label": torch.randint(0, 38, (2,)),
        "seg_mask": torch.randint(0, 2, (2, 1, 128, 128)).float(),
        "sep_target": torch.softmax(torch.randn(2, 8, 128, 128), dim=1),
    }
    
    # 计算损失
    losses = loss_fn(outputs, targets)
    total_loss = losses["total"]
    
    print(f"  前向传播损失: {total_loss.item():.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    
    # 检查梯度
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    print(f"  参数梯度数量: {len(grad_norms)}")
    print(f"  梯度范数范围: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
    
    # 优化器步骤
    optimizer.step()
    
    # 再次前向传播，验证参数已更新
    outputs2 = model(x)
    losses2 = loss_fn(outputs2, targets)
    
    print(f"  优化后损失: {losses2['total'].item():.4f}")
    
    print("  ✓ 反向传播测试通过")
    return True


def test_different_input_sizes():
    """测试不同输入尺寸"""
    print_section("8. 不同输入尺寸测试")
    
    model = WaferMultiTaskModel(separation_enabled=True)
    
    for size in [64, 128, 224, 256]:
        x = torch.randn(2, 3, size, size)
        outputs = model(x)
        
        assert outputs['cls_logits'].shape == (2, 38)
        assert outputs['seg_mask'].shape == (2, 1, size, size)
        assert outputs['sep_heatmaps'].shape == (2, 8, size, size)
        
        print(f"  尺寸 {size}x{size}: ✓")
    
    print("  ✓ 不同输入尺寸测试通过")
    return True


def test_freeze_unfreeze():
    """测试冻结/解冻编码器"""
    print_section("9. 编码器冻结/解冻测试")
    
    model = WaferMultiTaskModel()
    
    # 冻结编码器
    model.freeze_encoder()
    frozen_count = sum(1 for p in model.encoder.parameters() if not p.requires_grad)
    total_encoder_params = sum(1 for p in model.encoder.parameters())
    print(f"  冻结后: {frozen_count}/{total_encoder_params} 参数被冻结")
    assert frozen_count == total_encoder_params
    
    # 解冻编码器
    model.unfreeze_encoder()
    unfrozen_count = sum(1 for p in model.encoder.parameters() if p.requires_grad)
    print(f"  解冻后: {unfrozen_count}/{total_encoder_params} 参数可训练")
    assert unfrozen_count == total_encoder_params
    
    print("  ✓ 编码器冻结/解冻测试通过")
    return True


def main():
    """主函数"""
    print()
    print("*" * 60)
    print("*" + " " * 18 + "模型模块验证" + " " * 18 + "*")
    print("*" + " " * 14 + "Checkpoint 7 验证脚本" + " " * 13 + "*")
    print("*" * 60)
    
    tests = [
        ("编码器", test_encoder),
        ("解码器", test_decoder),
        ("分类头", test_classification_head),
        ("分离头", test_separation_head),
        ("多任务模型", test_multitask_model),
        ("损失函数", test_loss_functions),
        ("反向传播", test_backward_pass),
        ("不同输入尺寸", test_different_input_sizes),
        ("冻结/解冻", test_freeze_unfreeze),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  ✗ {name} 测试失败: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"  验证结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    if failed == 0:
        print()
        print("  ✓ 所有模型模块验证通过！")
        print("  模型组件已准备好进入 Phase 4: 训练与评估模块")
        print()
        return 0
    else:
        print()
        print("  ✗ 部分测试失败，请检查上述错误")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
