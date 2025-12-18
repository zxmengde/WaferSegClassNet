# -*- coding: utf-8 -*-
"""
损失函数模块

包含分类、分割和分离任务的损失函数
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coef(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    """
    计算 Dice 系数
    
    Args:
        y_pred: 预测值 (B, C, H, W)
        y_true: 真实值 (B, C, H, W)
        smooth: 平滑项
        
    Returns:
        Dice 系数
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    
    return dice


class DiceLoss(nn.Module):
    """Dice 损失"""
    
    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return 1.0 - dice_coef(y_pred, y_true, self.smooth)


class BCEDiceLoss(nn.Module):
    """BCE + Dice 组合损失"""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-7):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # 禁用 autocast 以避免 BCE 的数值不稳定问题
        with torch.cuda.amp.autocast(enabled=False):
            y_pred_float = y_pred.float()
            y_true_float = y_true.float()
            bce_loss = self.bce(y_pred_float, y_true_float)
        
        dice_loss = self.dice(y_pred, y_true)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测 logits (B, C)
            targets: 真实标签 (B,) 或 (B, C)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples
    
    CB(p, y) = (1 - beta) / (1 - beta^n_y) * L(p, y)
    """
    
    def __init__(
        self,
        beta: float = 0.9999,
        num_per_class: Optional[torch.Tensor] = None,
        num_classes: int = 38,
        loss_type: str = 'focal',
        gamma: float = 2.0,
    ):
        super().__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.gamma = gamma
        
        if num_per_class is not None:
            self._compute_weights(num_per_class)
        else:
            self.weights = None
    
    def _compute_weights(self, num_per_class: torch.Tensor):
        """计算类别权重"""
        effective_num = 1.0 - torch.pow(self.beta, num_per_class)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * self.num_classes
        self.register_buffer('weights', weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'focal':
            return FocalLoss(gamma=self.gamma, alpha=self.weights)(inputs, targets)
        else:
            return F.cross_entropy(inputs, targets, weight=self.weights)


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    
    组合分类、分割和分离损失
    """
    
    def __init__(
        self,
        classification_loss: str = 'cross_entropy',
        segmentation_loss: str = 'bce_dice',
        separation_loss: str = 'kl_divergence',
        weights: tuple = (1.0, 1.0, 0.5),
        focal_gamma: float = 2.0,
        focal_alpha: Optional[torch.Tensor] = None,
        class_balanced_beta: float = 0.9999,
        num_per_class: Optional[torch.Tensor] = None,
        num_classes: int = 38,
    ):
        super().__init__()
        
        self.weights = weights
        
        # 分类损失
        if classification_loss == 'cross_entropy':
            self.cls_loss = nn.CrossEntropyLoss()
        elif classification_loss == 'focal':
            self.cls_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        elif classification_loss == 'class_balanced':
            self.cls_loss = ClassBalancedLoss(
                beta=class_balanced_beta,
                num_per_class=num_per_class,
                num_classes=num_classes,
                gamma=focal_gamma,
            )
        else:
            raise ValueError(f"Unknown classification loss: {classification_loss}")
        
        # 分割损失
        if segmentation_loss == 'bce_dice':
            self.seg_loss = BCEDiceLoss()
        elif segmentation_loss == 'dice':
            self.seg_loss = DiceLoss()
        elif segmentation_loss == 'bce':
            self.seg_loss = nn.BCELoss()
        else:
            raise ValueError(f"Unknown segmentation loss: {segmentation_loss}")
        
        # 分离损失
        self.separation_loss = separation_loss
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: 模型输出 {cls_logits, seg_mask, sep_heatmaps}
            targets: 目标 {label, mask, label_8}
            
        Returns:
            {total, cls, seg, sep}
        """
        losses = {}
        
        # 分类损失
        cls_loss = self.cls_loss(outputs['cls_logits'], targets['label'])
        losses['cls'] = cls_loss
        
        # 分割损失
        seg_loss = self.seg_loss(outputs['seg_mask'], targets['mask'])
        losses['seg'] = seg_loss
        
        # 总损失
        total_loss = self.weights[0] * cls_loss + self.weights[1] * seg_loss
        
        # 分离损失（可选）
        if 'sep_heatmaps' in outputs and 'label_8' in targets:
            if self.separation_loss == 'kl_divergence':
                # 使用 label_8 作为软目标
                sep_target = targets['label_8'].unsqueeze(-1).unsqueeze(-1)
                sep_target = sep_target.expand_as(outputs['sep_heatmaps'])
                sep_loss = F.kl_div(
                    outputs['sep_heatmaps'].log(),
                    sep_target,
                    reduction='batchmean',
                )
            else:
                sep_loss = torch.tensor(0.0, device=outputs['cls_logits'].device)
            
            losses['sep'] = sep_loss
            total_loss = total_loss + self.weights[2] * sep_loss
        
        losses['total'] = total_loss
        
        return losses


if __name__ == "__main__":
    # 测试损失函数
    batch_size = 4
    num_classes = 38
    
    # 分类损失测试
    cls_logits = torch.randn(batch_size, num_classes)
    cls_targets = torch.randint(0, num_classes, (batch_size,))
    
    ce_loss = nn.CrossEntropyLoss()(cls_logits, cls_targets)
    focal_loss = FocalLoss()(cls_logits, cls_targets)
    
    print(f"CrossEntropy Loss: {ce_loss.item():.4f}")
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    # 分割损失测试
    seg_pred = torch.sigmoid(torch.randn(batch_size, 1, 224, 224))
    seg_target = torch.randint(0, 2, (batch_size, 1, 224, 224)).float()
    
    dice_loss = DiceLoss()(seg_pred, seg_target)
    bce_dice_loss = BCEDiceLoss()(seg_pred, seg_target)
    
    print(f"Dice Loss: {dice_loss.item():.4f}")
    print(f"BCE+Dice Loss: {bce_dice_loss.item():.4f}")
