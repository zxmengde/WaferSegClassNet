# -*- coding: utf-8 -*-
"""
MixedWM38 数据集类

支持:
- 38 类单标签分类
- 8 类多标签分类
- 二值分割 mask
- 伪 mask 生成
"""

import os
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from .mappings import CLASS_MAPPING, LABEL_38_TO_8, map_38_to_8


logger = logging.getLogger(__name__)


class PseudoMaskGenerator:
    """
    伪 Mask 生成器
    
    当真实分割 mask 不可用时，基于阈值和形态学操作生成伪 mask
    """
    
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.threshold = config.get('threshold', 128)
        self.morphology_kernel = config.get('morphology_kernel', 3)
        self.min_area = config.get('min_area', 100)
    
    def generate(self, image: np.ndarray) -> np.ndarray:
        """
        基于图像生成伪 mask
        
        Args:
            image: RGB 图像 (H, W, 3)
            
        Returns:
            二值 mask (H, W, 1)，值为 0 或 255
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 阈值分割
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去噪
        kernel = np.ones((self.morphology_kernel, self.morphology_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 添加通道维度
        mask = binary[:, :, np.newaxis]
        
        return mask.astype(np.uint8)
    
    def get_config(self) -> dict:
        """返回生成器配置"""
        return {
            'threshold': self.threshold,
            'morphology_kernel': self.morphology_kernel,
            'min_area': self.min_area,
        }


class MixedWM38Dataset(Dataset):
    """
    MixedWM38 数据集加载器
    
    支持:
    - 38 类单标签分类
    - 8 类多标签分类（从 38 类映射构造）
    - 二值分割 mask
    - 伪 mask 生成（当真实 mask 不可用时）
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        classification_mode: str = "single_label",
        transform: Optional[Callable] = None,
        pseudo_mask_config: Optional[dict] = None,
        debug: bool = False,
        max_per_class: int = 5,
        synthetic_root: Optional[str] = None,
        synthetic_only_train: bool = True,
    ):
        """
        Args:
            data_root: 数据根目录
            split: 数据集划分 (train | val | test)
            classification_mode: 分类模式 (single_label | multi_label)
            transform: 数据增强变换
            pseudo_mask_config: 伪 mask 生成配置
            debug: 是否为 debug 模式
            max_per_class: debug 模式下每类最大样本数
        """
        self.data_root = Path(data_root)
        self.split = split
        self.classification_mode = classification_mode
        self.transform = transform
        self.debug = debug
        self.max_per_class = max_per_class
        self.synthetic_root = Path(synthetic_root) if synthetic_root else None
        self.synthetic_only_train = synthetic_only_train
        
        # 伪 mask 生成器
        self.pseudo_mask_generator = PseudoMaskGenerator(pseudo_mask_config)
        self.pseudo_mask_used = False
        self.pseudo_mask_count = 0
        
        # 目录路径
        self.images_dir = self.data_root / "Images"
        self.labels_dir = self.data_root / "Labels"
        self.masks_dir = self.data_root / "Masks"
        
        # 加载文件列表
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        if self.pseudo_mask_used:
            logger.warning(f"Using pseudo masks for {self.pseudo_mask_count} samples")

    def _load_synthetic_samples(self) -> List[Dict]:
        """加载合成样本列表（如果配置了 synthetic_root）"""
        if self.synthetic_root is None:
            return []
        
        images_dir = self.synthetic_root / "Images"
        labels_dir = self.synthetic_root / "Labels"
        masks_dir = self.synthetic_root / "Masks"
        
        if not images_dir.exists():
            logger.warning(f"Synthetic images dir not found: {images_dir}")
            return []
        
        synthetic_samples = []
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
        
        for img_file in image_files:
            label_path = labels_dir / img_file
            if not label_path.exists():
                logger.warning(f"Synthetic label not found for {img_file}, skipping")
                continue
            
            label_raw = np.load(label_path)
            label_str = str(label_raw)
            if label_str not in CLASS_MAPPING:
                logger.warning(f"Unknown synthetic label {label_str} for {img_file}, skipping")
                continue
            
            label_38 = CLASS_MAPPING[label_str]
            mask_path = masks_dir / img_file
            has_real_mask = mask_path.exists()
            
            if not has_real_mask:
                self.pseudo_mask_used = True
                self.pseudo_mask_count += 1
            
            synthetic_samples.append({
                'image_path': str(images_dir / img_file),
                'label_path': str(label_path),
                'mask_path': str(mask_path) if has_real_mask else None,
                'label_38': label_38,
                'has_real_mask': has_real_mask,
                'is_synthetic': True,
            })
        
        if synthetic_samples:
            logger.info(f"Loaded {len(synthetic_samples)} synthetic samples from {self.synthetic_root}")
        
        return synthetic_samples

    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        base_samples = []
        
        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.npy')])
        
        # 首先收集所有样本及其标签
        for img_file in image_files:
            # 加载标签
            label_path = self.labels_dir / img_file
            if not label_path.exists():
                logger.warning(f"Label not found for {img_file}, skipping")
                continue
            
            label_raw = np.load(label_path)
            label_str = str(label_raw)
            
            if label_str not in CLASS_MAPPING:
                logger.warning(f"Unknown label {label_str} for {img_file}, skipping")
                continue
            
            label_38 = CLASS_MAPPING[label_str]
            
            # 检查 mask 是否存在
            mask_path = self.masks_dir / img_file
            has_real_mask = mask_path.exists()
            
            if not has_real_mask:
                self.pseudo_mask_used = True
                self.pseudo_mask_count += 1
            
            base_samples.append({
                'image_path': str(self.images_dir / img_file),
                'label_path': str(label_path),
                'mask_path': str(mask_path) if has_real_mask else None,
                'label_38': label_38,
                'has_real_mask': has_real_mask,
                'is_synthetic': False,
            })
        
        synthetic_samples = self._load_synthetic_samples()
        if synthetic_samples and not self.synthetic_only_train:
            all_samples = base_samples + synthetic_samples
        else:
            all_samples = base_samples
        
        # 使用分层采样进行 train/val/test 划分（80/10/10）
        # 确保每个类别在各划分中都有样本
        from collections import defaultdict
        import random
        
        # 设置随机种子确保可复现
        random.seed(42)
        
        # 按类别分组
        class_samples = defaultdict(list)
        for sample in all_samples:
            class_samples[sample['label_38']].append(sample)
        
        # 对每个类别进行分层划分
        train_samples = []
        val_samples = []
        test_samples = []
        
        for label, samples_in_class in class_samples.items():
            # 打乱类内样本顺序
            random.shuffle(samples_in_class)
            
            n = len(samples_in_class)
            train_end = int(0.8 * n)
            val_end = int(0.9 * n)
            
            # 确保每个划分至少有一个样本
            if n >= 3:
                train_end = max(1, train_end)
                val_end = max(train_end + 1, val_end)
            elif n == 2:
                train_end = 1
                val_end = 2
            else:  # n == 1
                # 只有一个样本时放入训练集
                train_end = 1
                val_end = 1
            
            train_samples.extend(samples_in_class[:train_end])
            val_samples.extend(samples_in_class[train_end:val_end])
            test_samples.extend(samples_in_class[val_end:])
        
        # 根据 split 选择样本
        if self.split == "train":
            samples = train_samples
        elif self.split == "val":
            samples = val_samples
        elif self.split == "test":
            samples = test_samples
        else:
            samples = all_samples
        
        if synthetic_samples and self.synthetic_only_train and self.split == "train":
            samples = samples + synthetic_samples
        
        # Debug 模式：限制每类样本数
        if self.debug:
            class_counts = {i: 0 for i in range(38)}
            filtered_samples = []
            for sample in samples:
                label_38 = sample['label_38']
                if class_counts[label_38] < self.max_per_class:
                    filtered_samples.append(sample)
                    class_counts[label_38] += 1
            samples = filtered_samples
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            {
                'image': Tensor (3, H, W),
                'mask': Tensor (1, H, W),
                'label_38': Tensor (1,) 或 int,
                'label_8': Tensor (8,),
            }
        """
        sample = self.samples[idx]
        
        # 加载图像
        image = np.load(sample['image_path']).astype(np.float32) / 255.0
        
        # 加载或生成 mask
        if sample['has_real_mask']:
            mask = np.load(sample['mask_path']).astype(np.float32) / 255.0
        else:
            # 生成伪 mask
            image_uint8 = (image * 255).astype(np.uint8)
            mask = self.pseudo_mask_generator.generate(image_uint8).astype(np.float32) / 255.0
        
        # 确保 mask 形状正确
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        
        # 获取标签
        label_38 = sample['label_38']
        label_8 = map_38_to_8(label_38)
        
        # 应用数据增强
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 确保数组是连续的（翻转操作可能产生负步长）
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        if not mask.flags['C_CONTIGUOUS']:
            mask = np.ascontiguousarray(mask)
        
        # 转换为 PyTorch 张量
        # 图像: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Mask: (H, W, 1) -> (1, H, W)
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        
        # 标签
        if self.classification_mode == "single_label":
            label = torch.tensor(label_38, dtype=torch.long)
        else:
            label = torch.from_numpy(label_8).float()
        
        return {
            'image': image,
            'mask': mask,
            'label_38': torch.tensor(label_38, dtype=torch.long),
            'label_8': torch.from_numpy(label_8).float(),
            'label': label,
        }
    
    def get_pseudo_mask_config(self) -> Optional[dict]:
        """获取伪 mask 配置（如果使用了伪 mask）"""
        if self.pseudo_mask_used:
            return {
                'used': True,
                'count': self.pseudo_mask_count,
                'config': self.pseudo_mask_generator.get_config(),
            }
        return None
    
    def get_class_counts(self) -> Dict[int, int]:
        """获取各类别样本数量"""
        counts = {i: 0 for i in range(38)}
        for sample in self.samples:
            counts[sample['label_38']] += 1
        return counts


def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    classification_mode: str = "single_label",
    debug: bool = False,
    max_per_class: int = 5,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    创建数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        classification_mode: 分类模式
        debug: 是否为 debug 模式
        max_per_class: debug 模式下每类最大样本数
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 创建数据集
    train_dataset = MixedWM38Dataset(
        data_root=data_root,
        split="train",
        classification_mode=classification_mode,
        debug=debug,
        max_per_class=max_per_class,
    )
    
    val_dataset = MixedWM38Dataset(
        data_root=data_root,
        split="val",
        classification_mode=classification_mode,
        debug=debug,
        max_per_class=max_per_class,
    )
    
    test_dataset = MixedWM38Dataset(
        data_root=data_root,
        split="test",
        classification_mode=classification_mode,
        debug=debug,
        max_per_class=max_per_class,
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    logging.basicConfig(level=logging.INFO)
    
    dataset = MixedWM38Dataset(
        data_root="data/processed",
        split="train",
        debug=True,
        max_per_class=2,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Label 38: {sample['label_38']}")
        print(f"Label 8: {sample['label_8']}")
