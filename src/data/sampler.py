# -*- coding: utf-8 -*-
"""
加权采样器模块

提供处理类别不平衡的采样策略：
- WeightedClassSampler: 按类别频率加权采样
- 支持多种加权模式: inverse, sqrt_inverse, effective_num

Requirements: 5.1
"""

import logging
from typing import List, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import numpy as np

logger = logging.getLogger(__name__)


class WeightedClassSampler(Sampler[int]):
    """
    按类别频率加权的采样器
    
    用于处理长尾分布数据集，通过对少数类过采样来平衡训练
    
    支持三种加权模式:
    - inverse: 频率倒数加权，w_i = 1 / n_i
    - sqrt_inverse: 平方根倒数加权，w_i = 1 / sqrt(n_i)
    - effective_num: 有效样本数加权，w_i = 1 / E_n，其中 E_n = (1 - beta^n) / (1 - beta)
    
    Reference: 
    - Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
    """
    
    def __init__(
        self,
        labels: List[int],
        mode: str = "inverse",
        beta: float = 0.9999,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        """
        Args:
            labels: 所有样本的类别标签列表
            mode: 加权模式 (inverse | sqrt_inverse | effective_num)
            beta: effective_num 模式的 beta 参数
            num_samples: 每个 epoch 采样的样本数，默认为数据集大小
            replacement: 是否有放回采样
        """
        super().__init__(data_source=None)
        
        self.labels = np.array(labels)
        self.mode = mode
        self.beta = beta
        self.replacement = replacement
        
        # 计算类别统计
        self.num_classes = int(self.labels.max()) + 1
        self.class_counts = self._compute_class_counts()
        
        # 计算样本权重
        self.weights = self._compute_weights()
        
        # 设置采样数量
        self.num_samples = num_samples if num_samples is not None else len(labels)
        
        # 记录统计信息
        self._log_statistics()
    
    def _compute_class_counts(self) -> np.ndarray:
        """计算每个类别的样本数"""
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for label in self.labels:
            counts[label] += 1
        return counts
    
    def _compute_weights(self) -> torch.Tensor:
        """计算每个样本的采样权重"""
        # 计算类别权重
        class_weights = np.zeros(self.num_classes, dtype=np.float64)
        
        for i in range(self.num_classes):
            count = self.class_counts[i]
            if count > 0:
                if self.mode == "inverse":
                    # 频率倒数
                    class_weights[i] = 1.0 / count
                elif self.mode == "sqrt_inverse":
                    # 平方根倒数
                    class_weights[i] = 1.0 / np.sqrt(count)
                elif self.mode == "effective_num":
                    # 有效样本数
                    effective_num = (1.0 - self.beta ** count) / (1.0 - self.beta)
                    class_weights[i] = 1.0 / effective_num
                else:
                    raise ValueError(f"Unknown weighting mode: {self.mode}")
            else:
                class_weights[i] = 0.0
        
        # 归一化类别权重
        if class_weights.sum() > 0:
            class_weights = class_weights / class_weights.sum()
        
        # 为每个样本分配权重
        sample_weights = np.array([class_weights[label] for label in self.labels])
        
        return torch.from_numpy(sample_weights).double()
    
    def _log_statistics(self):
        """记录采样统计信息"""
        non_zero_classes = np.sum(self.class_counts > 0)
        min_count = np.min(self.class_counts[self.class_counts > 0]) if non_zero_classes > 0 else 0
        max_count = np.max(self.class_counts)
        
        logger.info(f"WeightedClassSampler initialized:")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Total samples: {len(self.labels)}")
        logger.info(f"  Non-zero classes: {non_zero_classes}/{self.num_classes}")
        logger.info(f"  Min samples per class: {min_count}")
        logger.info(f"  Max samples per class: {max_count}")
        logger.info(f"  Imbalance ratio: {max_count / max(min_count, 1):.2f}")
    
    def __iter__(self) -> Iterator[int]:
        """生成采样索引"""
        indices = torch.multinomial(
            self.weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.num_samples
    
    def get_class_counts(self) -> np.ndarray:
        """返回类别计数"""
        return self.class_counts.copy()
    
    def get_effective_class_distribution(self) -> np.ndarray:
        """
        计算采样后的有效类别分布
        
        Returns:
            每个类别的期望采样比例
        """
        # 计算每个类别的总权重
        class_total_weights = np.zeros(self.num_classes)
        for i, label in enumerate(self.labels):
            class_total_weights[label] += self.weights[i].item()
        
        # 归一化为概率分布
        if class_total_weights.sum() > 0:
            class_total_weights = class_total_weights / class_total_weights.sum()
        
        return class_total_weights


def create_weighted_sampler(
    dataset: Dataset,
    mode: str = "inverse",
    beta: float = 0.9999,
    label_key: str = "label_38",
) -> WeightedClassSampler:
    """
    从数据集创建加权采样器
    
    Args:
        dataset: PyTorch 数据集
        mode: 加权模式
        beta: effective_num 模式的 beta 参数
        label_key: 标签字段名
        
    Returns:
        WeightedClassSampler 实例
    """
    # 提取所有标签
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if isinstance(sample, dict):
            label = sample[label_key]
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels.append(label)
        else:
            # 假设 sample 是 (image, label) 元组
            labels.append(sample[1])
    
    return WeightedClassSampler(labels=labels, mode=mode, beta=beta)


def get_tail_class_indices(
    class_counts: np.ndarray,
    threshold: int = 100,
) -> List[int]:
    """
    获取尾部类别索引
    
    Args:
        class_counts: 每个类别的样本数
        threshold: 样本数少于此值的类为尾部类
        
    Returns:
        尾部类别索引列表
    """
    tail_classes = []
    for i, count in enumerate(class_counts):
        if 0 < count < threshold:
            tail_classes.append(i)
    return tail_classes


def compute_class_balance_weights(
    class_counts: List[int],
    beta: float = 0.9999,
) -> torch.Tensor:
    """
    计算类别平衡权重（用于损失函数）
    
    Args:
        class_counts: 每个类别的样本数
        beta: 有效样本数计算的 beta 参数
        
    Returns:
        类别权重张量
    """
    class_counts = np.array(class_counts, dtype=np.float64)
    
    # 计算有效样本数
    effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
    
    # 计算权重（有效样本数的倒数）
    weights = np.zeros_like(effective_num)
    non_zero_mask = effective_num > 0
    weights[non_zero_mask] = 1.0 / effective_num[non_zero_mask]
    
    # 归一化
    if weights.sum() > 0:
        weights = weights / weights.sum() * len(class_counts)
    
    return torch.from_numpy(weights).float()


if __name__ == "__main__":
    # 测试采样器
    logging.basicConfig(level=logging.INFO)
    
    # 模拟长尾分布标签
    np.random.seed(42)
    
    # 创建不平衡标签分布
    labels = []
    class_sizes = [1000, 500, 200, 100, 50, 20, 10, 5]  # 8 个类别，长尾分布
    for class_id, size in enumerate(class_sizes):
        labels.extend([class_id] * size)
    
    print(f"Total samples: {len(labels)}")
    print(f"Class distribution: {class_sizes}")
    
    # 测试不同模式
    for mode in ["inverse", "sqrt_inverse", "effective_num"]:
        print(f"\n{'='*60}")
        print(f"Testing mode: {mode}")
        print("="*60)
        
        sampler = WeightedClassSampler(labels=labels, mode=mode)
        
        # 采样并统计
        sampled_indices = list(sampler)
        sampled_labels = [labels[i] for i in sampled_indices]
        
        # 统计采样后的类别分布
        sampled_counts = np.zeros(8)
        for label in sampled_labels:
            sampled_counts[label] += 1
        
        print(f"Original distribution: {class_sizes}")
        print(f"Sampled distribution:  {sampled_counts.astype(int).tolist()}")
        
        # 计算有效分布
        effective_dist = sampler.get_effective_class_distribution()
        print(f"Effective distribution: {[f'{p:.3f}' for p in effective_dist]}")
    
    # 测试类别平衡权重
    print(f"\n{'='*60}")
    print("Testing class balance weights")
    print("="*60)
    
    weights = compute_class_balance_weights(class_sizes, beta=0.9999)
    print(f"Class balance weights: {weights.tolist()}")
    print(f"Weight sum: {weights.sum().item():.2f}")
