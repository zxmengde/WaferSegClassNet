# -*- coding: utf-8 -*-
"""
数据加载器模块

提供 PyTorch DataLoader 创建功能
支持 debug 模式（每类最多5样本）
支持加权采样器（处理类别不平衡）
"""

import logging
from typing import Optional, Tuple, Dict, Any, List

import torch
from torch.utils.data import DataLoader, Subset

from .dataset import MixedWM38Dataset
from .augmentation import create_train_augmentation, create_val_augmentation
from .sampler import WeightedClassSampler

logger = logging.getLogger(__name__)


def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    classification_mode: str = "single_label",
    debug: bool = False,
    max_per_class: int = 5,
    augmentation_config: Optional[dict] = None,
    pin_memory: bool = True,
    sampler_mode: Optional[str] = None,
    sampler_beta: float = 0.9999,
    synthetic_root: Optional[str] = None,
    synthetic_only_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        classification_mode: 分类模式 (single_label | multi_label)
        debug: 是否为 debug 模式
        max_per_class: debug 模式下每类最大样本数
        augmentation_config: 数据增强配置
        pin_memory: 是否使用 pin_memory
        sampler_mode: 采样器模式 (None | uniform | inverse | sqrt_inverse | effective_num)
                      None 或 uniform 表示不使用加权采样
        sampler_beta: 加权采样器的 beta 参数（用于 effective_num 模式）
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Debug 模式调整
    if debug:
        num_workers = 0  # Debug 模式下使用单线程
        logger.info(f"Debug mode: max {max_per_class} samples per class")
    
    # 创建数据增强
    train_transform = create_train_augmentation(augmentation_config)
    val_transform = create_val_augmentation()
    
    # 创建数据集
    train_dataset = MixedWM38Dataset(
        data_root=data_root,
        split="train",
        classification_mode=classification_mode,
        transform=train_transform,
        debug=debug,
        max_per_class=max_per_class,
        synthetic_root=synthetic_root,
        synthetic_only_train=synthetic_only_train,
    )
    
    val_dataset = MixedWM38Dataset(
        data_root=data_root,
        split="val",
        classification_mode=classification_mode,
        transform=val_transform,
        debug=debug,
        max_per_class=max_per_class,
        synthetic_root=synthetic_root,
        synthetic_only_train=synthetic_only_train,
    )
    
    test_dataset = MixedWM38Dataset(
        data_root=data_root,
        split="test",
        classification_mode=classification_mode,
        transform=val_transform,
        debug=debug,
        max_per_class=max_per_class,
        synthetic_root=synthetic_root,
        synthetic_only_train=synthetic_only_train,
    )
    
    # 记录数据集大小
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # 创建训练数据加载器（可能使用加权采样器）
    train_sampler = None
    shuffle_train = True
    
    if sampler_mode is not None and sampler_mode not in ["uniform", "none"]:
        # 提取训练集标签
        train_labels = [train_dataset[i]['label_38'].item() for i in range(len(train_dataset))]
        train_sampler = WeightedClassSampler(
            labels=train_labels,
            mode=sampler_mode,
            beta=sampler_beta,
        )
        shuffle_train = False  # 使用采样器时不能 shuffle
        logger.info(f"Using weighted sampler with mode: {sampler_mode}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    
    return train_loader, val_loader, test_loader


def get_dataloaders_from_config(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    从配置字典创建数据加载器
    
    Args:
        config: 配置字典，应包含 data 和 experiment 部分
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_config = config.get('data', {})
    experiment_config = config.get('experiment', {})
    
    return get_dataloaders(
        data_root=data_config.get('data_root', 'data/processed'),
        batch_size=data_config.get('batch_size', 32),
        num_workers=data_config.get('num_workers', 4),
        classification_mode=data_config.get('classification_mode', 'single_label'),
        debug=experiment_config.get('debug', False),
        max_per_class=data_config.get('max_per_class', 5),
        augmentation_config=data_config.get('augmentation', None),
        synthetic_root=data_config.get('synthetic_root', None),
        synthetic_only_train=data_config.get('synthetic_only_train', True),
    )


def get_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """
    计算类别权重（用于处理类别不平衡）
    
    Args:
        train_loader: 训练数据加载器
        
    Returns:
        类别权重张量
    """
    dataset = train_loader.dataset
    
    # 获取类别计数
    if hasattr(dataset, 'get_class_counts'):
        class_counts = dataset.get_class_counts()
    else:
        # 手动统计
        class_counts = {}
        for sample in dataset:
            label = sample['label_38'].item()
            class_counts[label] = class_counts.get(label, 0) + 1
    
    # 计算权重（频率倒数）
    total = sum(class_counts.values())
    num_classes = max(class_counts.keys()) + 1
    
    weights = torch.zeros(num_classes)
    for cls, count in class_counts.items():
        if count > 0:
            weights[cls] = total / (num_classes * count)
        else:
            weights[cls] = 0.0
    
    return weights


def get_dataset_stats(data_root: str) -> Dict[str, Any]:
    """
    获取数据集统计信息
    
    Args:
        data_root: 数据根目录
        
    Returns:
        统计信息字典
    """
    # 加载完整数据集（不使用debug模式）
    dataset = MixedWM38Dataset(
        data_root=data_root,
        split="train",
        debug=False,
    )
    
    class_counts = dataset.get_class_counts()
    
    # 计算统计信息
    total_samples = len(dataset)
    num_classes = len([c for c in class_counts.values() if c > 0])
    
    # 找出头部和尾部类别
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    head_classes = [c[0] for c in sorted_counts[:5] if c[1] > 0]
    tail_classes = [c[0] for c in sorted_counts[-5:] if c[1] > 0]
    
    return {
        'total_samples': total_samples,
        'num_classes': num_classes,
        'class_counts': class_counts,
        'head_classes': head_classes,
        'tail_classes': tail_classes,
        'min_samples': min(c for c in class_counts.values() if c > 0),
        'max_samples': max(class_counts.values()),
        'pseudo_mask_used': dataset.pseudo_mask_used,
        'pseudo_mask_count': dataset.pseudo_mask_count,
    }


if __name__ == "__main__":
    # 测试数据加载器
    logging.basicConfig(level=logging.INFO)
    
    # 测试 debug 模式
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root="data/processed",
        batch_size=8,
        debug=True,
        max_per_class=2,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 测试一个批次
    for batch in train_loader:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch mask shape: {batch['mask'].shape}")
        print(f"Batch label shape: {batch['label'].shape}")
        break
    
    # 获取数据集统计
    stats = get_dataset_stats("data/processed")
    print(f"\nDataset stats:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Num classes: {stats['num_classes']}")
    print(f"  Min samples per class: {stats['min_samples']}")
    print(f"  Max samples per class: {stats['max_samples']}")
