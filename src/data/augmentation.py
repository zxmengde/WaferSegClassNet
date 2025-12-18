# -*- coding: utf-8 -*-
"""
数据增强模块

晶圆友好的数据增强，避免破坏中心/边缘语义
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# 白名单操作（允许使用）
WHITELIST_OPERATIONS = {
    'rotation',
    'flip',
    'horizontal_flip',
    'vertical_flip',
    'morphological_noise',
    'color_jitter',
    'brightness',
    'contrast',
    'gaussian_noise',
}

# 黑名单操作（禁止使用）
BLACKLIST_OPERATIONS = {
    'random_resized_crop',
    'center_crop',
    'random_crop',
    'crop',
    'cutout',
    'random_erasing',
    'coarse_dropout',
}


class WaferFriendlyAugmentation:
    """
    晶圆友好的数据增强
    
    白名单操作: rotation, flip, morphological_noise, color_jitter
    黑名单操作: RandomResizedCrop, CenterCrop, 大于20%的裁剪
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: 增强配置字典
                - rotation: 旋转角度列表，如 [0, 90, 180, 270]
                - flip: 是否启用翻转
                - horizontal_flip: 是否启用水平翻转
                - vertical_flip: 是否启用垂直翻转
                - morphological_noise: 形态学噪声强度 (0-1)
                - color_jitter: 颜色抖动配置
                - gaussian_noise: 高斯噪声标准差
        """
        config = config or {}
        
        # 验证配置
        self._validate_config(config)
        
        # 旋转配置
        self.rotation_angles = config.get('rotation', [0, 90, 180, 270])
        
        # 翻转配置
        self.flip = config.get('flip', True)
        self.horizontal_flip = config.get('horizontal_flip', True)
        self.vertical_flip = config.get('vertical_flip', True)
        
        # 形态学噪声
        self.morphological_noise = config.get('morphological_noise', 0.0)
        
        # 颜色抖动
        self.color_jitter = config.get('color_jitter', None)
        
        # 高斯噪声
        self.gaussian_noise = config.get('gaussian_noise', 0.0)
        
        # 保存原始配置
        self._config = config
    
    def _validate_config(self, config: dict) -> None:
        """
        验证增强配置不包含黑名单操作
        
        Raises:
            ValueError: 如果配置包含黑名单操作
        """
        errors = []
        
        for key in config.keys():
            key_lower = key.lower()
            if key_lower in BLACKLIST_OPERATIONS:
                errors.append(f"禁止使用的增强操作: {key}")
            
            # 检查裁剪比例
            if 'crop' in key_lower and key_lower not in BLACKLIST_OPERATIONS:
                crop_config = config[key]
                if isinstance(crop_config, dict):
                    min_scale = crop_config.get('min_scale', 1.0)
                    if min_scale < 0.8:
                        errors.append(
                            f"裁剪操作 {key} 的 min_scale={min_scale} 小于 0.8，"
                            "可能破坏晶圆中心/边缘语义"
                        )
        
        if errors:
            raise ValueError("增强配置验证失败:\n" + "\n".join(errors))
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        验证当前配置是否符合白名单/黑名单规则
        
        Returns:
            (是否通过, 错误列表)
        """
        errors = []
        
        for key in self._config.keys():
            key_lower = key.lower()
            if key_lower in BLACKLIST_OPERATIONS:
                errors.append(f"配置包含黑名单操作: {key}")
            elif key_lower not in WHITELIST_OPERATIONS:
                # 警告未知操作
                logger.warning(f"未知的增强操作: {key}")
        
        return len(errors) == 0, errors
    
    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        应用数据增强
        
        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            mask: 可选的分割mask (H, W, 1) 或 (H, W)
            
        Returns:
            {'image': augmented_image, 'mask': augmented_mask}
        """
        # 随机选择旋转角度
        if self.rotation_angles:
            angle = np.random.choice(self.rotation_angles)
            image = self._rotate(image, angle)
            if mask is not None:
                mask = self._rotate(mask, angle)
        
        # 随机翻转
        if self.flip:
            if self.horizontal_flip and np.random.random() > 0.5:
                image = self._horizontal_flip(image)
                if mask is not None:
                    mask = self._horizontal_flip(mask)
            
            if self.vertical_flip and np.random.random() > 0.5:
                image = self._vertical_flip(image)
                if mask is not None:
                    mask = self._vertical_flip(mask)
        
        # 形态学噪声（仅应用于图像）
        if self.morphological_noise > 0 and np.random.random() < self.morphological_noise:
            image = self._apply_morphological_noise(image)
        
        # 颜色抖动（仅应用于图像）
        if self.color_jitter is not None:
            image = self._apply_color_jitter(image, self.color_jitter)
        
        # 高斯噪声（仅应用于图像）
        if self.gaussian_noise > 0:
            image = self._apply_gaussian_noise(image, self.gaussian_noise)
        
        return {'image': image, 'mask': mask}
    
    def _rotate(self, img: np.ndarray, angle: int) -> np.ndarray:
        """旋转图像（仅支持90度倍数）"""
        if angle == 0:
            return img
        elif angle == 90:
            return np.rot90(img, k=1)
        elif angle == 180:
            return np.rot90(img, k=2)
        elif angle == 270:
            return np.rot90(img, k=3)
        else:
            # 对于非90度倍数，使用cv2旋转
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(img, M, (w, h))
    
    def _horizontal_flip(self, img: np.ndarray) -> np.ndarray:
        """水平翻转"""
        return np.flip(img, axis=1).copy()
    
    def _vertical_flip(self, img: np.ndarray) -> np.ndarray:
        """垂直翻转"""
        return np.flip(img, axis=0).copy()
    
    def _apply_morphological_noise(self, img: np.ndarray) -> np.ndarray:
        """应用形态学噪声"""
        # 随机选择腐蚀或膨胀
        kernel_size = np.random.choice([3, 5])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if np.random.random() > 0.5:
            return cv2.erode(img, kernel, iterations=1)
        else:
            return cv2.dilate(img, kernel, iterations=1)
    
    def _apply_color_jitter(
        self,
        img: np.ndarray,
        config: Dict[str, float]
    ) -> np.ndarray:
        """应用颜色抖动"""
        brightness = config.get('brightness', 0.0)
        contrast = config.get('contrast', 0.0)
        
        # 亮度调整
        if brightness > 0:
            factor = 1.0 + np.random.uniform(-brightness, brightness)
            img = np.clip(img * factor, 0, 1 if img.max() <= 1 else 255)
        
        # 对比度调整
        if contrast > 0:
            factor = 1.0 + np.random.uniform(-contrast, contrast)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 1 if img.max() <= 1 else 255)
        
        return img.astype(img.dtype)
    
    def _apply_gaussian_noise(
        self,
        img: np.ndarray,
        std: float
    ) -> np.ndarray:
        """应用高斯噪声"""
        noise = np.random.normal(0, std, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 1 if img.max() <= 1 else 255).astype(img.dtype)
    
    def get_config(self) -> dict:
        """返回当前配置"""
        return {
            'rotation': self.rotation_angles,
            'flip': self.flip,
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'morphological_noise': self.morphological_noise,
            'color_jitter': self.color_jitter,
            'gaussian_noise': self.gaussian_noise,
        }


def create_train_augmentation(config: Optional[dict] = None) -> WaferFriendlyAugmentation:
    """
    创建训练用数据增强
    
    Args:
        config: 可选的自定义配置
        
    Returns:
        WaferFriendlyAugmentation实例
    """
    default_config = {
        'rotation': [0, 90, 180, 270],
        'flip': True,
        'horizontal_flip': True,
        'vertical_flip': True,
        'morphological_noise': 0.1,
        'color_jitter': {
            'brightness': 0.1,
            'contrast': 0.1,
        },
        'gaussian_noise': 0.01,
    }
    
    if config:
        default_config.update(config)
    
    return WaferFriendlyAugmentation(default_config)


def create_val_augmentation() -> WaferFriendlyAugmentation:
    """
    创建验证/测试用数据增强（无增强）
    
    Returns:
        WaferFriendlyAugmentation实例（仅保持原样）
    """
    return WaferFriendlyAugmentation({
        'rotation': [0],
        'flip': False,
        'morphological_noise': 0.0,
    })


def validate_augmentation_config(config: dict) -> Tuple[bool, List[str]]:
    """
    验证增强配置是否符合白名单/黑名单规则
    
    Args:
        config: 增强配置字典
        
    Returns:
        (是否通过, 错误列表)
    """
    errors = []
    warnings = []
    
    for key in config.keys():
        key_lower = key.lower()
        
        # 检查黑名单
        if key_lower in BLACKLIST_OPERATIONS:
            errors.append(f"配置包含黑名单操作: {key}")
        
        # 检查是否在白名单
        elif key_lower not in WHITELIST_OPERATIONS:
            warnings.append(f"未知的增强操作: {key}")
        
        # 检查裁剪比例
        if 'crop' in key_lower:
            crop_config = config[key]
            if isinstance(crop_config, dict):
                min_scale = crop_config.get('min_scale', 1.0)
                if min_scale < 0.8:
                    errors.append(
                        f"裁剪操作 {key} 的 min_scale={min_scale} 小于 0.8"
                    )
    
    # 记录警告
    for warning in warnings:
        logger.warning(warning)
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    # 测试增强
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试图像
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    test_mask = np.random.randint(0, 2, (224, 224, 1)).astype(np.float32)
    
    # 测试训练增强
    aug = create_train_augmentation()
    result = aug(test_image, test_mask)
    
    print(f"Input image shape: {test_image.shape}")
    print(f"Output image shape: {result['image'].shape}")
    print(f"Output mask shape: {result['mask'].shape}")
    
    # 验证配置
    is_valid, errors = aug.validate_config()
    print(f"Config valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # 测试黑名单检测
    try:
        bad_aug = WaferFriendlyAugmentation({
            'random_resized_crop': {'min_scale': 0.5}
        })
    except ValueError as e:
        print(f"Correctly rejected blacklist operation: {e}")
