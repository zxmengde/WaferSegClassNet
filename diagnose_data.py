#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断数据和模型问题
"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from collections import Counter
from data.dataloader import get_dataloaders
from models.multitask import create_model

def main():
    print("=" * 50)
    print("数据集诊断")
    print("=" * 50)
    
    # 加载数据
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root='data/processed',
        batch_size=8,
        debug=True,
        max_per_class=2,
    )
    
    print(f"\n训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    
    # 收集标签
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch['label'].numpy().tolist())
    
    val_labels = []
    for batch in val_loader:
        val_labels.extend(batch['label'].numpy().tolist())
    
    print(f"\n训练集标签范围: {min(train_labels)} - {max(train_labels)}")
    print(f"训练集唯一标签数: {len(set(train_labels))}")
    print(f"训练集标签分布: {Counter(train_labels)}")
    
    print(f"\n验证集标签范围: {min(val_labels)} - {max(val_labels)}")
    print(f"验证集唯一标签数: {len(set(val_labels))}")
    print(f"验证集标签分布: {Counter(val_labels)}")
    
    # 检查训练集和验证集有没有重叠的类别
    train_classes = set(train_labels)
    val_classes = set(val_labels)
    common_classes = train_classes & val_classes
    print(f"\n训练集和验证集共同类别数: {len(common_classes)}")
    print(f"共同类别: {sorted(common_classes)}")
    
    if len(common_classes) == 0:
        print("\n⚠️  警告: 训练集和验证集没有共同类别！这会导致验证accuracy为0！")
    
    print("\n" + "=" * 50)
    print("模型输出诊断")
    print("=" * 50)
    
    # 创建模型
    model = create_model(
        encoder='custom',
        classification_classes=38,
        segmentation_classes=1,
        separation_enabled=False,
    )
    model.eval()
    
    # 测试一个batch
    for batch in train_loader:
        images = batch['image']
        labels = batch['label']
        
        with torch.no_grad():
            outputs = model(images)
        
        cls_logits = outputs['cls_logits']
        preds = cls_logits.argmax(dim=1)
        
        print(f"\n输入形状: {images.shape}")
        print(f"输出logits形状: {cls_logits.shape}")
        print(f"真实标签: {labels.numpy()}")
        print(f"预测标签: {preds.numpy()}")
        print(f"Logits范围: [{cls_logits.min():.4f}, {cls_logits.max():.4f}]")
        break
    
    print("\n诊断完成！")

if __name__ == "__main__":
    main()
