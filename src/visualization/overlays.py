# -*- coding: utf-8 -*-
"""
Overlay 可视化模块（伪mask overlay唯一责任模块）

提供分割和分离任务的可视化功能：
- generate_seg_overlays(): 生成分割 overlay 图
- generate_separation_heatmaps(): 生成 8 通道分离热力图
- generate_pseudo_mask_overlays(): 生成伪 mask 样例导出（至少10张）

统一负责所有 overlay 导出逻辑

Requirements: 8.2, 8.3, 2.5
"""

import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 8 类基础缺陷名称
COMPONENT_NAMES = ['Center', 'Donut', 'EL', 'ER', 'LOC', 'NF', 'S', 'Random']


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """
    归一化图像到 [0, 1] 范围
    
    Args:
        image: 输入图像
        
    Returns:
        归一化后的图像
    """
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min > 1e-8:
        return (image - img_min) / (img_max - img_min)
    return np.zeros_like(image)


def _to_numpy(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    将 tensor 转换为 numpy 数组
    
    Args:
        tensor: 输入 tensor 或 numpy 数组
        
    Returns:
        numpy 数组
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return np.asarray(tensor)


def _prepare_image_for_display(
    image: np.ndarray,
    channel_first: bool = True,
) -> np.ndarray:
    """
    准备图像用于显示
    
    Args:
        image: 输入图像，形状 (C, H, W) 或 (H, W, C) 或 (H, W)
        channel_first: 是否为 channel first 格式
        
    Returns:
        形状为 (H, W, C) 或 (H, W) 的归一化图像
    """
    if image.ndim == 2:
        return _normalize_image(image)
    
    if channel_first and image.ndim == 3:
        # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))
    
    # 如果是单通道，squeeze
    if image.shape[-1] == 1:
        image = image.squeeze(-1)
    
    return _normalize_image(image)



def generate_seg_overlays(
    images: Union[np.ndarray, torch.Tensor],
    pred_masks: Union[np.ndarray, torch.Tensor],
    gt_masks: Union[np.ndarray, torch.Tensor],
    output_dir: str,
    max_samples: int = 20,
    threshold: float = 0.5,
    overlay_alpha: float = 0.5,
) -> List[str]:
    """
    生成分割 overlay 图
    
    显示原图、真实 mask、预测 mask 的对比
    
    Args:
        images: 输入图像，形状 (N, C, H, W)
        pred_masks: 预测 mask，形状 (N, 1, H, W) 或 (N, H, W)
        gt_masks: 真实 mask，形状 (N, 1, H, W) 或 (N, H, W)
        output_dir: 输出目录
        max_samples: 最大样本数
        threshold: 二值化阈值
        overlay_alpha: overlay 透明度
        
    Returns:
        保存的文件路径列表
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping segmentation overlays")
        return []
    
    # 转换为 numpy
    images = _to_numpy(images)
    pred_masks = _to_numpy(pred_masks)
    gt_masks = _to_numpy(gt_masks)
    
    # 确保 mask 是 3D (N, H, W)
    if pred_masks.ndim == 4:
        pred_masks = pred_masks.squeeze(1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    num_samples = min(len(images), max_samples)
    
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # 准备图像
        img = _prepare_image_for_display(images[i], channel_first=True)
        gt_mask = (gt_masks[i] > 0.5).astype(float)
        pred_mask = (pred_masks[i] > threshold).astype(float)
        
        # 1. 原图
        if img.ndim == 2:
            axes[0].imshow(img, cmap='gray')
        else:
            axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # 2. 真实 mask
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # 3. 预测 mask
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # 4. Overlay 对比
        # 绿色: TP, 红色: FP, 蓝色: FN
        overlay = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
        tp = (gt_mask > 0.5) & (pred_mask > 0.5)
        fp = (gt_mask <= 0.5) & (pred_mask > 0.5)
        fn = (gt_mask > 0.5) & (pred_mask <= 0.5)
        
        overlay[tp, 1] = 1.0  # Green: True Positive
        overlay[fp, 0] = 1.0  # Red: False Positive
        overlay[fn, 2] = 1.0  # Blue: False Negative
        
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = img
        
        blended = img_rgb * (1 - overlay_alpha) + overlay * overlay_alpha
        blended = np.clip(blended, 0, 1)
        
        axes[3].imshow(blended)
        axes[3].set_title('Overlay (G:TP, R:FP, B:FN)')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        save_path = output_dir / f"sample_{i:03d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        saved_paths.append(str(save_path))
    
    logger.info(f"Segmentation overlays saved to {output_dir} ({num_samples} samples)")
    return saved_paths



def generate_separation_heatmaps(
    images: Optional[Union[np.ndarray, torch.Tensor]],
    heatmaps: Union[np.ndarray, torch.Tensor],
    output_dir: str,
    max_samples: int = 10,
    component_names: Optional[List[str]] = None,
    save_tensors: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    生成 8 通道分离热力图
    
    Args:
        images: 输入图像，形状 (N, C, H, W)，可选
        heatmaps: 分离热力图，形状 (N, 8, H, W)
        output_dir: 输出目录
        max_samples: 最大样本数
        component_names: 成分名称列表
        save_tensors: 是否保存原始 tensor
        
    Returns:
        (图片路径列表, tensor路径列表)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping separation heatmaps")
        return [], []
    
    # 转换为 numpy
    heatmaps = _to_numpy(heatmaps)
    if images is not None:
        images = _to_numpy(images)
    
    if component_names is None:
        component_names = COMPONENT_NAMES
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    tensor_paths = []
    num_samples = min(len(heatmaps), max_samples)
    num_channels = heatmaps.shape[1]
    
    for i in range(num_samples):
        # 保存原始 tensor
        if save_tensors:
            tensor_path = output_dir / f"sample_{i:03d}.pt"
            torch.save(torch.from_numpy(heatmaps[i]), tensor_path)
            tensor_paths.append(str(tensor_path))
        
        # 创建可视化
        # 布局: 第一行原图 + 4个通道，第二行 4个通道
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        # 原图
        if images is not None and i < len(images):
            img = _prepare_image_for_display(images[i], channel_first=True)
            if img.ndim == 2:
                axes[0, 0].imshow(img, cmap='gray')
            else:
                axes[0, 0].imshow(img)
            axes[0, 0].set_title('Input Image', fontsize=10)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Image', ha='center', va='center')
        axes[0, 0].axis('off')
        
        # 8 通道热力图
        for c in range(min(num_channels, 8)):
            row = (c + 1) // 5
            col = (c + 1) % 5
            
            heatmap = heatmaps[i, c]
            im = axes[row, col].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
            
            name = component_names[c] if c < len(component_names) else f'Ch{c}'
            axes[row, col].set_title(name, fontsize=10)
            axes[row, col].axis('off')
        
        # 隐藏多余的子图
        if num_channels < 9:
            axes[1, 4].axis('off')
        
        # 添加颜色条
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Activation')
        
        plt.suptitle(f'Separation Heatmaps - Sample {i}', fontsize=12)
        # 调整子图间距
        plt.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.05, wspace=0.1, hspace=0.2)
        
        image_path = output_dir / f"sample_{i:03d}.png"
        plt.savefig(image_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        image_paths.append(str(image_path))
    
    logger.info(f"Separation heatmaps saved to {output_dir} ({num_samples} samples)")
    return image_paths, tensor_paths



def generate_pseudo_mask_overlays(
    images: Union[np.ndarray, torch.Tensor],
    pseudo_masks: Union[np.ndarray, torch.Tensor],
    output_dir: str,
    min_samples: int = 10,
    max_samples: int = 20,
    overlay_alpha: float = 0.5,
    overlay_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> List[str]:
    """
    生成伪 mask 样例导出（至少10张）
    
    用于验证伪 mask 生成质量
    
    Args:
        images: 输入图像，形状 (N, C, H, W)
        pseudo_masks: 伪 mask，形状 (N, 1, H, W) 或 (N, H, W)
        output_dir: 输出目录
        min_samples: 最小样本数（至少10张）
        max_samples: 最大样本数
        overlay_alpha: overlay 透明度
        overlay_color: overlay 颜色 (R, G, B)
        
    Returns:
        保存的文件路径列表
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping pseudo mask overlays")
        return []
    
    # 转换为 numpy
    images = _to_numpy(images)
    pseudo_masks = _to_numpy(pseudo_masks)
    
    # 确保 mask 是 3D (N, H, W)
    if pseudo_masks.ndim == 4:
        pseudo_masks = pseudo_masks.squeeze(1)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保至少导出 min_samples 张
    num_samples = max(min(len(images), max_samples), min(len(images), min_samples))
    
    saved_paths = []
    
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 准备图像
        img = _prepare_image_for_display(images[i], channel_first=True)
        mask = (pseudo_masks[i] > 0.5).astype(float)
        
        # 1. 原图
        if img.ndim == 2:
            axes[0].imshow(img, cmap='gray')
        else:
            axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # 2. 伪 mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Pseudo Mask')
        axes[1].axis('off')
        
        # 3. Overlay
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = img
        
        # 创建彩色 overlay
        overlay = np.zeros_like(img_rgb)
        overlay[mask > 0.5, 0] = overlay_color[0]
        overlay[mask > 0.5, 1] = overlay_color[1]
        overlay[mask > 0.5, 2] = overlay_color[2]
        
        blended = img_rgb * (1 - overlay_alpha * mask[..., np.newaxis]) + \
                  overlay * overlay_alpha * mask[..., np.newaxis]
        blended = np.clip(blended, 0, 1)
        
        axes[2].imshow(blended)
        axes[2].set_title('Pseudo Mask Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        save_path = output_dir / f"sample_{i:03d}_overlay.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        saved_paths.append(str(save_path))
    
    logger.info(f"Pseudo mask overlays saved to {output_dir} ({num_samples} samples, min required: {min_samples})")
    return saved_paths


def generate_all_visualizations(
    images: Union[np.ndarray, torch.Tensor],
    pred_masks: Union[np.ndarray, torch.Tensor],
    gt_masks: Union[np.ndarray, torch.Tensor],
    output_dir: str,
    sep_heatmaps: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pseudo_mask_used: bool = False,
    max_samples: int = 20,
) -> dict:
    """
    生成所有可视化
    
    Args:
        images: 输入图像
        pred_masks: 预测 mask
        gt_masks: 真实 mask
        output_dir: 输出目录
        sep_heatmaps: 分离热力图（可选）
        pseudo_mask_used: 是否使用了伪 mask
        max_samples: 最大样本数
        
    Returns:
        包含所有保存路径的字典
    """
    output_dir = Path(output_dir)
    result = {}
    
    # 分割 overlay
    seg_overlay_dir = output_dir / "seg_overlays"
    result['seg_overlays'] = generate_seg_overlays(
        images, pred_masks, gt_masks,
        output_dir=str(seg_overlay_dir),
        max_samples=max_samples,
    )
    
    # 分离热力图
    if sep_heatmaps is not None:
        sep_dir = output_dir / "separation_maps"
        image_paths, tensor_paths = generate_separation_heatmaps(
            images, sep_heatmaps,
            output_dir=str(sep_dir),
            max_samples=min(max_samples, 10),
        )
        result['separation_images'] = image_paths
        result['separation_tensors'] = tensor_paths
    
    # 伪 mask 样例
    if pseudo_mask_used:
        pseudo_dir = output_dir / "pseudo_mask_samples"
        result['pseudo_mask_overlays'] = generate_pseudo_mask_overlays(
            images, gt_masks,
            output_dir=str(pseudo_dir),
            min_samples=10,
            max_samples=max_samples,
        )
    
    return result



if __name__ == "__main__":
    # 测试 overlay 可视化函数
    print("=" * 60)
    print("Testing Overlay Visualization Functions")
    print("=" * 60)
    
    import tempfile
    import shutil
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建测试数据
        np.random.seed(42)
        batch_size = 12
        height, width = 64, 64
        
        # 模拟图像 (N, C, H, W)
        images = np.random.rand(batch_size, 3, height, width).astype(np.float32)
        
        # 模拟 mask (N, 1, H, W)
        gt_masks = np.zeros((batch_size, 1, height, width), dtype=np.float32)
        pred_masks = np.zeros((batch_size, 1, height, width), dtype=np.float32)
        
        for i in range(batch_size):
            # 创建圆形 mask
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2 + np.random.randint(-10, 10), width // 2 + np.random.randint(-10, 10)
            radius = np.random.randint(10, 20)
            mask = ((y - center_y) ** 2 + (x - center_x) ** 2) <= radius ** 2
            gt_masks[i, 0] = mask.astype(float)
            
            # 预测 mask 有一些偏移
            pred_center_y = center_y + np.random.randint(-5, 5)
            pred_center_x = center_x + np.random.randint(-5, 5)
            pred_mask = ((y - pred_center_y) ** 2 + (x - pred_center_x) ** 2) <= radius ** 2
            pred_masks[i, 0] = pred_mask.astype(float)
        
        # 模拟分离热力图 (N, 8, H, W)
        sep_heatmaps = np.random.rand(batch_size, 8, height, width).astype(np.float32)
        
        # 测试 generate_seg_overlays
        print("\n1. Testing generate_seg_overlays:")
        seg_dir = Path(temp_dir) / "seg_overlays"
        paths = generate_seg_overlays(
            images, pred_masks, gt_masks,
            output_dir=str(seg_dir),
            max_samples=5,
        )
        print(f"   Generated {len(paths)} segmentation overlays")
        assert len(paths) == 5, f"Expected 5 overlays, got {len(paths)}"
        
        # 测试 generate_separation_heatmaps
        print("\n2. Testing generate_separation_heatmaps:")
        sep_dir = Path(temp_dir) / "separation_maps"
        image_paths, tensor_paths = generate_separation_heatmaps(
            images, sep_heatmaps,
            output_dir=str(sep_dir),
            max_samples=5,
        )
        print(f"   Generated {len(image_paths)} heatmap images and {len(tensor_paths)} tensors")
        assert len(image_paths) == 5, f"Expected 5 images, got {len(image_paths)}"
        assert len(tensor_paths) == 5, f"Expected 5 tensors, got {len(tensor_paths)}"
        
        # 测试 generate_pseudo_mask_overlays
        print("\n3. Testing generate_pseudo_mask_overlays:")
        pseudo_dir = Path(temp_dir) / "pseudo_mask_samples"
        paths = generate_pseudo_mask_overlays(
            images, gt_masks,
            output_dir=str(pseudo_dir),
            min_samples=10,
            max_samples=15,
        )
        print(f"   Generated {len(paths)} pseudo mask overlays")
        assert len(paths) >= 10, f"Expected at least 10 overlays, got {len(paths)}"
        
        # 测试 generate_all_visualizations
        print("\n4. Testing generate_all_visualizations:")
        all_dir = Path(temp_dir) / "all_viz"
        result = generate_all_visualizations(
            images, pred_masks, gt_masks,
            output_dir=str(all_dir),
            sep_heatmaps=sep_heatmaps,
            pseudo_mask_used=True,
            max_samples=5,
        )
        print(f"   Generated visualizations:")
        print(f"     - seg_overlays: {len(result.get('seg_overlays', []))} files")
        print(f"     - separation_images: {len(result.get('separation_images', []))} files")
        print(f"     - separation_tensors: {len(result.get('separation_tensors', []))} files")
        print(f"     - pseudo_mask_overlays: {len(result.get('pseudo_mask_overlays', []))} files")
        
        print("\n" + "=" * 60)
        print("All overlay visualization tests passed!")
        print("=" * 60)
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")
