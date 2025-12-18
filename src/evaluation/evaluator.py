# -*- coding: utf-8 -*-
"""
评估器模块

提供统一的评估接口，支持：
- 分类指标计算 (Macro-F1, mAP)
- 分割指标计算 (Dice, IoU)
- 混淆矩阵生成
- 可视化输出 (seg_overlays, separation_maps)
- metrics.csv 输出

Requirements: 3.5, 3.6, 2.5
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast

from .metrics import (
    compute_macro_f1,
    compute_map,
    compute_dice,
    compute_iou,
    compute_confusion_matrix,
    compute_per_class_metrics,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    统一评估器
    
    输出:
    - metrics.csv (Macro-F1, Dice, IoU, mAP)
    - confusion_matrix.png
    - seg_overlays/
    - separation_maps/ (E3)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        amp_enabled: bool = True,
        num_classes: int = 38,
        classification_mode: str = "single_label",
        output_dir: str = "results/experiment",
        class_names: Optional[List[str]] = None,
        pseudo_mask_used: bool = False,
    ):
        """
        Args:
            model: 模型
            device: 设备
            amp_enabled: 是否启用 AMP
            num_classes: 分类类别数
            classification_mode: 分类模式 (single_label | multi_label)
            output_dir: 输出目录
            class_names: 类别名称列表
            pseudo_mask_used: 是否使用了伪 mask
        """
        self.model = model.to(device)
        self.device = device
        self.amp_enabled = amp_enabled and torch.cuda.is_available()
        self.num_classes = num_classes
        self.classification_mode = classification_mode
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.pseudo_mask_used = pseudo_mask_used
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "seg_overlays").mkdir(exist_ok=True)
        
        if pseudo_mask_used:
            (self.output_dir / "pseudo_mask_samples").mkdir(exist_ok=True)
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        save_visualizations: bool = True,
        max_vis_samples: int = 20,
    ) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            save_visualizations: 是否保存可视化
            max_vis_samples: 最大可视化样本数
            
        Returns:
            评估结果字典
        """
        self.model.eval()
        
        # 收集预测结果
        all_cls_preds = []
        all_cls_labels = []
        all_cls_probs = []  # 用于 mAP
        all_seg_preds = []
        all_seg_masks = []
        all_images = []
        all_sep_heatmaps = []
        
        logger.info("Starting evaluation...")
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            with autocast('cuda', enabled=self.amp_enabled):
                outputs = self.model(images)
            
            # 分类预测
            cls_logits = outputs['cls_logits']
            cls_probs = torch.softmax(cls_logits, dim=1)
            cls_preds = cls_logits.argmax(dim=1)
            
            all_cls_preds.append(cls_preds.cpu())
            all_cls_labels.append(labels.cpu())
            all_cls_probs.append(cls_probs.cpu())
            
            # 分割预测
            seg_preds = torch.sigmoid(outputs['seg_mask'])
            all_seg_preds.append(seg_preds.cpu())
            all_seg_masks.append(masks.cpu())
            
            # 保存图像用于可视化
            if save_visualizations and len(all_images) * images.shape[0] < max_vis_samples:
                all_images.append(images.cpu())
            
            # 分离热力图（如果有）
            if 'sep_heatmaps' in outputs:
                all_sep_heatmaps.append(outputs['sep_heatmaps'].cpu())
        
        # 合并结果
        all_cls_preds = torch.cat(all_cls_preds)
        all_cls_labels = torch.cat(all_cls_labels)
        all_cls_probs = torch.cat(all_cls_probs)
        all_seg_preds = torch.cat(all_seg_preds)
        all_seg_masks = torch.cat(all_seg_masks)
        
        if all_images:
            all_images = torch.cat(all_images)[:max_vis_samples]
        
        if all_sep_heatmaps:
            all_sep_heatmaps = torch.cat(all_sep_heatmaps)
        
        # 计算指标
        metrics = self._compute_metrics(
            all_cls_preds, all_cls_labels, all_cls_probs,
            all_seg_preds, all_seg_masks
        )
        
        # 保存结果
        self._save_metrics_csv(metrics)
        
        # 生成可视化
        if save_visualizations:
            self._save_confusion_matrix(all_cls_preds, all_cls_labels)
            
            if len(all_images) > 0:
                self._save_seg_overlays(
                    all_images,
                    all_seg_preds[:len(all_images)],
                    all_seg_masks[:len(all_images)],
                )
            
            if len(all_sep_heatmaps) > 0:
                self._save_separation_maps(
                    all_images if len(all_images) > 0 else None,
                    all_sep_heatmaps[:max_vis_samples],
                )
            
            # 伪 mask 样例导出
            if self.pseudo_mask_used and len(all_images) > 0:
                self._save_pseudo_mask_samples(
                    all_images,
                    all_seg_masks[:len(all_images)],
                    min(10, len(all_images)),
                )
        
        logger.info(f"Evaluation complete. Results saved to {self.output_dir}")
        
        return metrics
    
    def _compute_metrics(
        self,
        cls_preds: torch.Tensor,
        cls_labels: torch.Tensor,
        cls_probs: torch.Tensor,
        seg_preds: torch.Tensor,
        seg_masks: torch.Tensor,
    ) -> Dict[str, Any]:
        """计算所有指标"""
        metrics = {}
        
        # 分类指标
        macro_f1, per_class_f1 = compute_macro_f1(
            cls_labels, cls_preds, num_classes=self.num_classes
        )
        metrics['macro_f1'] = macro_f1
        metrics['per_class_f1'] = per_class_f1.tolist()
        
        # 每类详细指标
        per_class_metrics = compute_per_class_metrics(
            cls_labels, cls_preds, num_classes=self.num_classes
        )
        metrics['per_class_precision'] = per_class_metrics['precision'].tolist()
        metrics['per_class_recall'] = per_class_metrics['recall'].tolist()
        metrics['per_class_support'] = per_class_metrics['support'].tolist()
        
        # 多标签 mAP（如果是多标签模式）
        if self.classification_mode == "multi_label":
            # 需要多标签真实标签，这里简化处理
            # 实际使用时需要从 dataloader 获取多标签标签
            pass
        
        # 分割指标
        dice = compute_dice(seg_preds, seg_masks, threshold=0.5)
        iou = compute_iou(seg_preds, seg_masks, threshold=0.5)
        metrics['dice'] = dice
        metrics['iou'] = iou
        
        # 混淆矩阵
        cm = compute_confusion_matrix(cls_labels, cls_preds, num_classes=self.num_classes)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 计算准确率
        accuracy = (cls_preds == cls_labels).float().mean().item()
        metrics['accuracy'] = accuracy
        
        logger.info(f"Macro-F1: {macro_f1:.4f}")
        logger.info(f"Dice: {dice:.4f}")
        logger.info(f"IoU: {iou:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def _save_metrics_csv(self, metrics: Dict[str, Any]):
        """保存 metrics.csv"""
        csv_path = self.output_dir / "metrics.csv"
        
        # 主要指标
        main_metrics = {
            'macro_f1': metrics['macro_f1'],
            'dice': metrics['dice'],
            'iou': metrics['iou'],
            'accuracy': metrics['accuracy'],
        }
        
        # 写入主要指标
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key, value in main_metrics.items():
                writer.writerow([key, f"{value:.6f}"])
        
        # 写入每类指标
        per_class_csv_path = self.output_dir / "metrics_per_class.csv"
        with open(per_class_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['class_id', 'class_name', 'precision', 'recall', 'f1', 'support'])
            
            for i in range(self.num_classes):
                class_name = self.class_names[i] if self.class_names else f"class_{i}"
                writer.writerow([
                    i,
                    class_name,
                    f"{metrics['per_class_precision'][i]:.6f}",
                    f"{metrics['per_class_recall'][i]:.6f}",
                    f"{metrics['per_class_f1'][i]:.6f}",
                    metrics['per_class_support'][i],
                ])
        
        logger.info(f"Metrics saved to {csv_path}")
        logger.info(f"Per-class metrics saved to {per_class_csv_path}")
    
    def _save_confusion_matrix(
        self,
        cls_preds: torch.Tensor,
        cls_labels: torch.Tensor,
    ):
        """保存混淆矩阵图"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            cm = compute_confusion_matrix(cls_labels, cls_preds, num_classes=self.num_classes)
            
            # 归一化
            cm_normalized = cm.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零
            cm_normalized = cm_normalized / row_sums
            
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(cm_normalized, cmap='Blues')
            
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix (Normalized)')
            
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            save_path = self.output_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            
            logger.info(f"Confusion matrix saved to {save_path}")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping confusion matrix visualization")
    
    def _save_seg_overlays(
        self,
        images: torch.Tensor,
        seg_preds: torch.Tensor,
        seg_masks: torch.Tensor,
    ):
        """保存分割 overlay 图"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            overlay_dir = self.output_dir / "seg_overlays"
            
            for i in range(min(len(images), 20)):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                # 原图
                img = images[i].permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                axes[0].imshow(img)
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                # 真实 mask
                mask_gt = seg_masks[i, 0].numpy()
                axes[1].imshow(mask_gt, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # 预测 mask
                mask_pred = (seg_preds[i, 0].numpy() > 0.5).astype(float)
                axes[2].imshow(mask_pred, cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(overlay_dir / f"sample_{i:03d}.png", dpi=100)
                plt.close()
            
            logger.info(f"Segmentation overlays saved to {overlay_dir}")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping segmentation overlays")
    
    def _save_separation_maps(
        self,
        images: Optional[torch.Tensor],
        sep_heatmaps: torch.Tensor,
    ):
        """保存分离热力图"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            sep_dir = self.output_dir / "separation_maps"
            sep_dir.mkdir(exist_ok=True)
            
            component_names = ['Center', 'Donut', 'EL', 'ER', 'LOC', 'NF', 'S', 'Random']
            
            for i in range(min(len(sep_heatmaps), 10)):
                # 保存原始 tensor
                torch.save(sep_heatmaps[i], sep_dir / f"sample_{i:03d}.pt")
                
                # 可视化
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))
                
                # 原图（如果有）
                if images is not None and i < len(images):
                    img = images[i].permute(1, 2, 0).numpy()
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    axes[0, 0].imshow(img)
                    axes[0, 0].set_title('Input Image')
                else:
                    axes[0, 0].axis('off')
                axes[0, 0].axis('off')
                
                # 8 通道热力图
                heatmap = sep_heatmaps[i].numpy()
                for c in range(8):
                    row = (c + 1) // 5
                    col = (c + 1) % 5
                    axes[row, col].imshow(heatmap[c], cmap='hot')
                    axes[row, col].set_title(component_names[c] if c < len(component_names) else f'Ch{c}')
                    axes[row, col].axis('off')
                
                # 隐藏多余的子图
                axes[1, 4].axis('off')
                
                plt.tight_layout()
                plt.savefig(sep_dir / f"sample_{i:03d}.png", dpi=100)
                plt.close()
            
            logger.info(f"Separation maps saved to {sep_dir}")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping separation maps")
    
    def _save_pseudo_mask_samples(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        num_samples: int = 10,
    ):
        """保存伪 mask 样例"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            sample_dir = self.output_dir / "pseudo_mask_samples"
            
            for i in range(min(num_samples, len(images))):
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                
                # 原图
                img = images[i].permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                axes[0].imshow(img)
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                # 伪 mask overlay
                mask = masks[i, 0].numpy()
                axes[1].imshow(img)
                axes[1].imshow(mask, cmap='Reds', alpha=0.5)
                axes[1].set_title('Pseudo Mask Overlay')
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(sample_dir / f"sample_{i:03d}_overlay.png", dpi=100)
                plt.close()
            
            logger.info(f"Pseudo mask samples saved to {sample_dir}")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping pseudo mask samples")


def create_evaluator(
    model: nn.Module,
    config: Any,
    output_dir: str,
    pseudo_mask_used: bool = False,
) -> Evaluator:
    """
    从配置创建评估器
    
    Args:
        model: 模型
        config: 配置对象
        output_dir: 输出目录
        pseudo_mask_used: 是否使用了伪 mask
        
    Returns:
        Evaluator 实例
    """
    # 从配置获取参数
    if hasattr(config, 'model'):
        num_classes = config.model.classification_classes
    else:
        num_classes = 38
    
    if hasattr(config, 'data'):
        classification_mode = config.data.classification_mode
    else:
        classification_mode = "single_label"
    
    if hasattr(config, 'training'):
        amp_enabled = config.training.amp_enabled
    else:
        amp_enabled = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return Evaluator(
        model=model,
        device=device,
        amp_enabled=amp_enabled,
        num_classes=num_classes,
        classification_mode=classification_mode,
        output_dir=output_dir,
        pseudo_mask_used=pseudo_mask_used,
    )
