# -*- coding: utf-8 -*-
"""
成分分离模块 (E3-Fallback)

基于 Prototype 相似度的弱监督成分分离
不需要重新训练，在 eval 阶段生成 separation_maps

Requirements: 6.1, 6.4
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..data.mappings import DEFECT_NAMES_8
except ImportError:
    from src.data.mappings import DEFECT_NAMES_8


logger = logging.getLogger(__name__)


class PrototypeSeparator:
    """
    基于 Prototype 相似度的成分分离器
    
    工作原理:
    1. 从训练集中提取所有单缺陷类样本（class 1-8）
    2. 对每类样本通过 encoder 提取 feature map
    3. 计算每类的 prototype: prototype_i = mean(features_class_i)
    4. 对输入图像计算与每个 prototype 的 cosine similarity
    5. 输出 8 通道热力图
    
    这是 E3-Fallback 方案，不需要训练分离头
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        device: str = "cuda",
        num_components: int = 8,
    ):
        """
        Args:
            encoder: 特征编码器
            device: 计算设备
            num_components: 成分数量（8 类基础缺陷）
        """
        self.encoder = encoder
        self.device = device
        self.num_components = num_components
        
        # Prototype 存储: {component_id: prototype_tensor}
        self.prototypes: Dict[int, torch.Tensor] = {}
        self.prototype_counts: Dict[int, int] = {}
        
        # 组件名称
        self.component_names = DEFECT_NAMES_8
        
        # 是否已构建 prototype
        self.prototypes_built = False
    
    @torch.no_grad()
    def build_prototypes(
        self,
        dataloader: DataLoader,
        single_defect_classes: Optional[List[int]] = None,
    ) -> Dict[str, any]:
        """
        从训练数据构建 Prototype
        
        Args:
            dataloader: 训练数据加载器
            single_defect_classes: 单缺陷类 ID 列表（默认 1-8）
            
        Returns:
            构建统计信息
        """
        if single_defect_classes is None:
            # 默认单缺陷类: 1=Center, 2=Donut, ..., 8=Random
            single_defect_classes = list(range(1, 9))
        
        self.encoder.eval()
        self.encoder.to(self.device)
        
        # 初始化累加器
        feature_accumulators: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(self.num_components)
        }
        
        logger.info("Building prototypes from training data...")
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            labels_38 = batch['label_38']
            
            # 提取特征
            features, _ = self.encoder(images)  # (B, C, H, W)
            
            # 对每个样本检查是否为单缺陷类
            for i, label in enumerate(labels_38):
                label_val = label.item()
                
                # 检查是否为单缺陷类
                if label_val in single_defect_classes:
                    # 映射到 0-7 的组件 ID
                    component_id = label_val - 1  # class 1 -> component 0
                    
                    # 保存特征（全局平均池化后的向量）
                    feat = features[i]  # (C, H, W)
                    feature_accumulators[component_id].append(feat.cpu())
        
        # 计算每个组件的 prototype
        stats = {
            'total_samples': 0,
            'per_component': {},
        }
        
        for comp_id in range(self.num_components):
            feats = feature_accumulators[comp_id]
            count = len(feats)
            stats['per_component'][self.component_names[comp_id]] = count
            stats['total_samples'] += count
            
            if count > 0:
                # 堆叠并计算均值
                stacked = torch.stack(feats, dim=0)  # (N, C, H, W)
                prototype = stacked.mean(dim=0)  # (C, H, W)
                self.prototypes[comp_id] = prototype.to(self.device)
                self.prototype_counts[comp_id] = count
                logger.info(f"  {self.component_names[comp_id]}: {count} samples")
            else:
                logger.warning(f"  {self.component_names[comp_id]}: 0 samples (no prototype)")
                # 使用零向量作为占位符
                self.prototypes[comp_id] = None
                self.prototype_counts[comp_id] = 0
        
        self.prototypes_built = True
        logger.info(f"Prototypes built: {stats['total_samples']} total samples")
        
        return stats
    
    @torch.no_grad()
    def compute_separation_maps(
        self,
        images: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        计算分离热力图
        
        Args:
            images: 输入图像 (B, 3, H, W)
            temperature: softmax 温度参数
            
        Returns:
            分离热力图 (B, 8, H, W)
        """
        if not self.prototypes_built:
            raise RuntimeError("Prototypes not built. Call build_prototypes() first.")
        
        self.encoder.eval()
        images = images.to(self.device)
        
        # 提取特征
        features, _ = self.encoder(images)  # (B, C, H', W')
        B, C, H, W = features.shape
        
        # 计算与每个 prototype 的相似度
        separation_maps = []
        
        for comp_id in range(self.num_components):
            prototype = self.prototypes[comp_id]
            
            if prototype is None:
                # 没有 prototype，输出零热力图
                sim_map = torch.zeros(B, 1, H, W, device=self.device)
            else:
                # 计算 cosine similarity
                # prototype: (C, H', W') -> 需要与 features 对齐
                # 使用空间位置的 cosine similarity
                
                # 归一化
                feat_norm = F.normalize(features, dim=1)  # (B, C, H, W)
                proto_norm = F.normalize(prototype.unsqueeze(0), dim=1)  # (1, C, H', W')
                
                # 逐位置计算相似度
                sim_map = (feat_norm * proto_norm).sum(dim=1, keepdim=True)  # (B, 1, H, W)
                
                # 应用温度缩放并转换到 [0, 1]
                sim_map = torch.sigmoid(sim_map / temperature)
            
            separation_maps.append(sim_map)
        
        # 合并所有通道
        separation_maps = torch.cat(separation_maps, dim=1)  # (B, 8, H, W)
        
        # 上采样到原始图像尺寸
        original_size = images.shape[2:]
        separation_maps = F.interpolate(
            separation_maps, size=original_size, mode='bilinear', align_corners=False
        )
        
        return separation_maps
    
    def save_prototypes(self, save_path: str):
        """保存 prototypes 到文件"""
        save_dict = {
            'prototypes': {k: v.cpu() if v is not None else None 
                          for k, v in self.prototypes.items()},
            'prototype_counts': self.prototype_counts,
            'component_names': self.component_names,
            'num_components': self.num_components,
        }
        torch.save(save_dict, save_path)
        logger.info(f"Prototypes saved to {save_path}")
    
    def load_prototypes(self, load_path: str):
        """从文件加载 prototypes"""
        save_dict = torch.load(load_path, map_location='cpu', weights_only=False)
        
        self.prototypes = {
            k: v.to(self.device) if v is not None else None
            for k, v in save_dict['prototypes'].items()
        }
        self.prototype_counts = save_dict['prototype_counts']
        self.component_names = save_dict.get('component_names', DEFECT_NAMES_8)
        self.num_components = save_dict.get('num_components', 8)
        self.prototypes_built = True
        
        logger.info(f"Prototypes loaded from {load_path}")



def save_separation_maps(
    separation_maps: torch.Tensor,
    images: Optional[torch.Tensor],
    output_dir: str,
    sample_indices: Optional[List[int]] = None,
    max_samples: int = 10,
) -> List[str]:
    """
    保存分离热力图（图片 + tensor）
    
    Args:
        separation_maps: 分离热力图 (B, 8, H, W)
        images: 原始图像 (B, 3, H, W)，可选
        output_dir: 输出目录
        sample_indices: 样本索引列表
        max_samples: 最大保存样本数
        
    Returns:
        保存的文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    num_samples = min(len(separation_maps), max_samples)
    
    if sample_indices is None:
        sample_indices = list(range(num_samples))
    else:
        sample_indices = sample_indices[:num_samples]
    
    for i, idx in enumerate(sample_indices):
        # 保存原始 tensor
        tensor_path = output_dir / f"sample_{i:03d}.pt"
        torch.save(separation_maps[idx].cpu(), tensor_path)
        saved_files.append(str(tensor_path))
        
        # 生成可视化
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            
            # 原图（如果有）
            if images is not None and idx < len(images):
                img = images[idx].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                axes[0, 0].imshow(img)
                axes[0, 0].set_title('Input Image')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Image', ha='center', va='center')
            axes[0, 0].axis('off')
            
            # 8 通道热力图
            heatmap = separation_maps[idx].cpu().numpy()
            for c in range(8):
                row = (c + 1) // 5
                col = (c + 1) % 5
                axes[row, col].imshow(heatmap[c], cmap='hot', vmin=0, vmax=1)
                axes[row, col].set_title(DEFECT_NAMES_8[c] if c < len(DEFECT_NAMES_8) else f'Ch{c}')
                axes[row, col].axis('off')
            
            # 隐藏多余的子图
            axes[1, 4].axis('off')
            
            plt.tight_layout()
            img_path = output_dir / f"sample_{i:03d}.png"
            plt.savefig(img_path, dpi=100)
            plt.close()
            saved_files.append(str(img_path))
            
        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")
    
    logger.info(f"Saved {num_samples} separation maps to {output_dir}")
    return saved_files


class SeparationEvaluator:
    """
    分离评估器
    
    在 eval 阶段使用 PrototypeSeparator 生成 separation_maps
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        output_dir: str = "results/e3",
        temperature: float = 0.1,
    ):
        """
        Args:
            model: 多任务模型（需要有 encoder 属性）
            device: 计算设备
            output_dir: 输出目录
            temperature: 相似度计算的温度参数
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.temperature = temperature
        
        # 创建 PrototypeSeparator
        if hasattr(model, 'encoder'):
            self.separator = PrototypeSeparator(
                encoder=model.encoder,
                device=device,
                num_components=8,
            )
        else:
            raise ValueError("Model must have an 'encoder' attribute")
        
        # 创建输出目录
        (self.output_dir / "separation_maps").mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def build_prototypes_from_dataloader(
        self,
        train_dataloader: DataLoader,
    ) -> Dict[str, any]:
        """
        从训练数据构建 prototypes
        
        Args:
            train_dataloader: 训练数据加载器
            
        Returns:
            构建统计信息
        """
        return self.separator.build_prototypes(train_dataloader)
    
    @torch.no_grad()
    def evaluate_with_separation(
        self,
        dataloader: DataLoader,
        max_vis_samples: int = 10,
    ) -> Dict[str, any]:
        """
        评估并生成分离热力图
        
        Args:
            dataloader: 评估数据加载器
            max_vis_samples: 最大可视化样本数
            
        Returns:
            评估结果
        """
        self.model.eval()
        
        all_images = []
        all_separation_maps = []
        all_labels = []
        
        logger.info("Generating separation maps...")
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            labels = batch['label_38']
            
            # 计算分离热力图
            sep_maps = self.separator.compute_separation_maps(
                images, temperature=self.temperature
            )
            
            all_images.append(images.cpu())
            all_separation_maps.append(sep_maps.cpu())
            all_labels.append(labels)
            
            # 限制样本数
            total_samples = sum(len(x) for x in all_images)
            if total_samples >= max_vis_samples:
                break
        
        # 合并结果
        all_images = torch.cat(all_images)[:max_vis_samples]
        all_separation_maps = torch.cat(all_separation_maps)[:max_vis_samples]
        all_labels = torch.cat(all_labels)[:max_vis_samples]
        
        # 保存分离热力图
        sep_dir = self.output_dir / "separation_maps"
        saved_files = save_separation_maps(
            separation_maps=all_separation_maps,
            images=all_images,
            output_dir=str(sep_dir),
            max_samples=max_vis_samples,
        )
        
        # 保存 prototypes
        proto_path = self.output_dir / "prototypes.pt"
        self.separator.save_prototypes(str(proto_path))
        
        results = {
            'num_samples': len(all_separation_maps),
            'saved_files': saved_files,
            'prototype_path': str(proto_path),
            'prototype_counts': self.separator.prototype_counts,
        }
        
        logger.info(f"Separation evaluation complete. Results saved to {self.output_dir}")
        
        return results


def create_separation_evaluator(
    model: nn.Module,
    config: any,
    output_dir: str,
) -> SeparationEvaluator:
    """
    从配置创建分离评估器
    
    Args:
        model: 多任务模型
        config: 配置对象
        output_dir: 输出目录
        
    Returns:
        SeparationEvaluator 实例
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 从配置获取温度参数
    temperature = 0.1
    if hasattr(config, 'model') and hasattr(config.model, 'separation_temperature'):
        temperature = config.model.separation_temperature
    
    return SeparationEvaluator(
        model=model,
        device=device,
        output_dir=output_dir,
        temperature=temperature,
    )


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.models.encoder import WaferEncoder
    
    # 创建测试编码器
    encoder = WaferEncoder()
    
    # 创建分离器
    separator = PrototypeSeparator(
        encoder=encoder,
        device="cpu",
        num_components=8,
    )
    
    print("PrototypeSeparator created successfully")
    print(f"Component names: {separator.component_names}")
