# -*- coding: utf-8 -*-
"""
损失函数模块

包含多任务学习所需的各种损失函数：
- DiceLoss: 分割任务的 Dice 损失
- BCEDiceLoss: BCE + Dice 组合损失
- FocalLoss: 处理类别不平衡的 Focal 损失
- ClassBalancedLoss: 基于有效样本数的类别平衡损失
- MultiTaskLoss: 多任务组合损失

Requirements: 3.1, 5.1, 5.2
"""

from typing import Optional, List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coef(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    计算 Dice 系数
    
    Args:
        pred: 预测值 (B, C, H, W) 或 (B, 1, H, W)，值域 [0, 1]
        target: 真实值 (B, C, H, W) 或 (B, 1, H, W)，二值
        smooth: 平滑因子
        
    Returns:
        Dice 系数（标量）
    """
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.mean()


class DiceLoss(nn.Module):
    """
    Dice 损失函数
    
    用于分割任务，衡量预测 mask 与真实 mask 的重叠度
    
    公式: Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: 平滑因子，防止除零
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (B, C, H, W) 或 (B, 1, H, W)，经过 sigmoid
            target: 真实值 (B, C, H, W) 或 (B, 1, H, W)，二值
            
        Returns:
            Dice 损失值
        """
        # 确保 pred 经过 sigmoid
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # 展平
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # 计算 Dice 系数
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice 组合损失
    
    结合二元交叉熵和 Dice 损失，用于分割任务
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        """
        Args:
            bce_weight: BCE 损失权重
            dice_weight: Dice 损失权重
            smooth: Dice 损失的平滑因子
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (B, 1, H, W)，logits（未经 sigmoid）
            target: 真实值 (B, 1, H, W)，二值
            
        Returns:
            组合损失值
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(torch.sigmoid(pred), target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    用于处理类别不平衡问题，降低易分类样本的损失权重
    
    公式: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            gamma: 聚焦参数，gamma > 0 降低易分类样本的权重
            alpha: 类别权重，可以是标量或每类权重向量
            reduction: 损失聚合方式 (mean | sum | none)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测 logits (B, C) 或 (B, C, H, W)
            target: 真实标签 (B,) 或 (B, H, W)，整数类别索引
            
        Returns:
            Focal 损失值
        """
        # 计算交叉熵（不聚合）
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        
        # 计算 p_t
        if pred.dim() == 2:
            # 分类任务 (B, C)
            p = F.softmax(pred, dim=1)
            p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # 分割任务 (B, C, H, W)
            p = F.softmax(pred, dim=1)
            p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # 计算 focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # 应用 alpha 权重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha 是每类权重向量
                alpha = self.alpha.to(pred.device)
                if pred.dim() == 2:
                    alpha_t = alpha.gather(0, target)
                else:
                    alpha_t = alpha.gather(0, target.view(-1)).view_as(target)
            focal_weight = alpha_t * focal_weight
        
        # 计算 focal loss
        focal_loss = focal_weight * ce_loss
        
        # 聚合
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss
    
    基于有效样本数的类别平衡损失
    
    公式: 
        有效样本数 E_n = (1 - beta^n) / (1 - beta)
        权重 w_i = 1 / E_{n_i}
    
    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
    """
    
    def __init__(
        self,
        num_per_class: List[int],
        beta: float = 0.9999,
        loss_type: str = "focal",
        gamma: float = 2.0,
    ):
        """
        Args:
            num_per_class: 每类样本数列表
            beta: 有效样本数计算的 beta 参数
            loss_type: 基础损失类型 (focal | softmax)
            gamma: Focal Loss 的 gamma 参数
        """
        super().__init__()
        self.beta = beta
        self.loss_type = loss_type
        self.gamma = gamma
        
        # 计算有效样本数和权重
        effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(num_per_class, dtype=torch.float))
        effective_num = effective_num / (1.0 - beta)
        
        # 计算类别权重（有效样本数的倒数）
        weights = 1.0 / effective_num
        weights = weights / weights.sum() * len(num_per_class)  # 归一化
        
        self.register_buffer("weights", weights)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测 logits (B, C)
            target: 真实标签 (B,)
            
        Returns:
            Class-Balanced 损失值
        """
        if self.loss_type == "focal":
            # 使用 Focal Loss 作为基础
            return self._focal_loss(pred, target)
        else:
            # 使用加权交叉熵
            return F.cross_entropy(pred, target, weight=self.weights)
    
    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """计算带类别权重的 Focal Loss"""
        # 计算交叉熵
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        
        # 计算 p_t
        p = F.softmax(pred, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # 计算 focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # 应用类别权重
        class_weights = self.weights.gather(0, target)
        
        # 计算最终损失
        loss = class_weights * focal_weight * ce_loss
        
        return loss.mean()


class KLDivergenceLoss(nn.Module):
    """
    KL 散度损失
    
    用于成分分离任务，衡量预测分布与目标分布的差异
    """
    
    def __init__(self, temperature: float = 1.0, reduction: str = "batchmean"):
        """
        Args:
            temperature: 温度参数，用于软化分布
            reduction: 损失聚合方式
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测 logits (B, C, H, W)
            target: 目标分布 (B, C, H, W)，已经是概率分布
            
        Returns:
            KL 散度损失值
        """
        # 对预测应用 log_softmax
        pred_log_prob = F.log_softmax(pred / self.temperature, dim=1)
        
        # 确保 target 是有效的概率分布
        target = target.clamp(min=1e-8)
        target = target / target.sum(dim=1, keepdim=True)
        
        # 计算 KL 散度
        loss = F.kl_div(pred_log_prob, target, reduction=self.reduction)
        
        return loss * (self.temperature ** 2)


class MSELoss(nn.Module):
    """
    MSE 损失
    
    用于成分分离任务的替代损失
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (B, C, H, W)
            target: 目标值 (B, C, H, W)
            
        Returns:
            MSE 损失值
        """
        # 对预测应用 sigmoid 归一化到 [0, 1]
        pred_normalized = torch.sigmoid(pred)
        return self.mse(pred_normalized, target)



class MultiTaskLoss(nn.Module):
    """
    多任务组合损失
    
    整合分类、分割和分离任务的损失函数
    
    总损失 = w_cls * L_cls + w_seg * L_seg + w_sep * L_sep
    """
    
    def __init__(
        self,
        classification_loss: str = "cross_entropy",
        segmentation_loss: str = "bce_dice",
        separation_loss: str = "kl_divergence",
        weights: List[float] = None,
        num_classes: int = 38,
        num_per_class: Optional[List[int]] = None,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[Union[float, List[float]]] = None,
        class_balanced_beta: float = 0.9999,
        separation_temperature: float = 1.0,
    ):
        """
        Args:
            classification_loss: 分类损失类型 (cross_entropy | focal | class_balanced)
            segmentation_loss: 分割损失类型 (bce_dice | dice | bce)
            separation_loss: 分离损失类型 (kl_divergence | mse)
            weights: 多任务损失权重 [cls, seg, sep]
            num_classes: 分类类别数
            num_per_class: 每类样本数（用于 class_balanced 损失）
            focal_gamma: Focal Loss 的 gamma 参数
            focal_alpha: Focal Loss 的 alpha 参数
            class_balanced_beta: Class-Balanced Loss 的 beta 参数
            separation_temperature: KL 散度的温度参数
        """
        super().__init__()
        
        # 损失权重
        if weights is None:
            weights = [1.0, 1.0, 0.5]
        self.weights = weights
        
        # 分类损失
        self.cls_loss_type = classification_loss
        if classification_loss == "cross_entropy":
            self.cls_loss = nn.CrossEntropyLoss()
        elif classification_loss == "focal":
            alpha = None
            if focal_alpha is not None:
                if isinstance(focal_alpha, list):
                    alpha = torch.tensor(focal_alpha)
                else:
                    alpha = focal_alpha
            self.cls_loss = FocalLoss(gamma=focal_gamma, alpha=alpha)
        elif classification_loss == "class_balanced":
            if num_per_class is None:
                # 默认均匀分布
                num_per_class = [100] * num_classes
            self.cls_loss = ClassBalancedLoss(
                num_per_class=num_per_class,
                beta=class_balanced_beta,
                gamma=focal_gamma,
            )
        else:
            raise ValueError(f"Unknown classification loss: {classification_loss}")
        
        # 分割损失
        self.seg_loss_type = segmentation_loss
        if segmentation_loss == "bce_dice":
            self.seg_loss = BCEDiceLoss()
        elif segmentation_loss == "dice":
            self.seg_loss = DiceLoss()
        elif segmentation_loss == "bce":
            self.seg_loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown segmentation loss: {segmentation_loss}")
        
        # 分离损失
        self.sep_loss_type = separation_loss
        if separation_loss == "kl_divergence":
            self.sep_loss = KLDivergenceLoss(temperature=separation_temperature)
        elif separation_loss == "mse":
            self.sep_loss = MSELoss()
        else:
            raise ValueError(f"Unknown separation loss: {separation_loss}")
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        Args:
            outputs: 模型输出
                - cls_logits: (B, num_classes) 分类 logits
                - seg_mask: (B, 1, H, W) 分割预测
                - sep_heatmaps: (B, 8, H, W) 分离热力图（可选）
            targets: 真实标签
                - cls_label: (B,) 分类标签
                - seg_mask: (B, 1, H, W) 分割 mask
                - sep_target: (B, 8, H, W) 分离目标（可选）
                
        Returns:
            损失字典:
                - total: 总损失
                - cls: 分类损失
                - seg: 分割损失
                - sep: 分离损失（如果有）
        """
        losses = {}
        total_loss = 0.0
        
        # 分类损失
        if "cls_logits" in outputs and "cls_label" in targets:
            cls_loss = self.cls_loss(outputs["cls_logits"], targets["cls_label"])
            losses["cls"] = cls_loss
            total_loss = total_loss + self.weights[0] * cls_loss
        
        # 分割损失
        if "seg_mask" in outputs and "seg_mask" in targets:
            seg_pred = outputs["seg_mask"]
            seg_target = targets["seg_mask"]
            
            # 确保 target 是 float 类型
            if seg_target.dtype != seg_pred.dtype:
                seg_target = seg_target.float()
            
            # 对于 DiceLoss，需要先 sigmoid
            if self.seg_loss_type == "dice":
                seg_loss = self.seg_loss(torch.sigmoid(seg_pred), seg_target)
            else:
                seg_loss = self.seg_loss(seg_pred, seg_target)
            
            losses["seg"] = seg_loss
            total_loss = total_loss + self.weights[1] * seg_loss
        
        # 分离损失（可选）
        if "sep_heatmaps" in outputs and "sep_target" in targets:
            sep_loss = self.sep_loss(outputs["sep_heatmaps"], targets["sep_target"])
            losses["sep"] = sep_loss
            if len(self.weights) > 2:
                total_loss = total_loss + self.weights[2] * sep_loss
        
        losses["total"] = total_loss
        
        return losses


def create_loss(
    classification_loss: str = "cross_entropy",
    segmentation_loss: str = "bce_dice",
    separation_loss: str = "kl_divergence",
    weights: List[float] = None,
    num_classes: int = 38,
    num_per_class: Optional[List[int]] = None,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[Union[float, List[float]]] = None,
    class_balanced_beta: float = 0.9999,
) -> MultiTaskLoss:
    """
    创建多任务损失函数
    
    Args:
        classification_loss: 分类损失类型
        segmentation_loss: 分割损失类型
        separation_loss: 分离损失类型
        weights: 多任务损失权重
        num_classes: 分类类别数
        num_per_class: 每类样本数
        focal_gamma: Focal Loss gamma
        focal_alpha: Focal Loss alpha
        class_balanced_beta: Class-Balanced Loss beta
        
    Returns:
        MultiTaskLoss 实例
    """
    return MultiTaskLoss(
        classification_loss=classification_loss,
        segmentation_loss=segmentation_loss,
        separation_loss=separation_loss,
        weights=weights,
        num_classes=num_classes,
        num_per_class=num_per_class,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        class_balanced_beta=class_balanced_beta,
    )


def create_loss_from_config(loss_config, num_classes: int = 38, num_per_class: Optional[List[int]] = None) -> MultiTaskLoss:
    """
    从配置创建损失函数
    
    Args:
        loss_config: LossConfig 实例或字典
        num_classes: 分类类别数
        num_per_class: 每类样本数
        
    Returns:
        MultiTaskLoss 实例
    """
    if hasattr(loss_config, '__dict__'):
        # LossConfig dataclass
        return MultiTaskLoss(
            classification_loss=loss_config.classification,
            segmentation_loss=loss_config.segmentation,
            separation_loss=loss_config.separation,
            weights=loss_config.weights,
            num_classes=num_classes,
            num_per_class=num_per_class,
            focal_gamma=loss_config.focal_gamma,
            focal_alpha=loss_config.focal_alpha,
            class_balanced_beta=loss_config.class_balanced_beta,
        )
    else:
        # 字典
        return MultiTaskLoss(
            classification_loss=loss_config.get("classification", "cross_entropy"),
            segmentation_loss=loss_config.get("segmentation", "bce_dice"),
            separation_loss=loss_config.get("separation", "kl_divergence"),
            weights=loss_config.get("weights", [1.0, 1.0, 0.5]),
            num_classes=num_classes,
            num_per_class=num_per_class,
            focal_gamma=loss_config.get("focal_gamma", 2.0),
            focal_alpha=loss_config.get("focal_alpha"),
            class_balanced_beta=loss_config.get("class_balanced_beta", 0.9999),
        )


if __name__ == "__main__":
    # 测试损失函数
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)
    
    batch_size = 4
    num_classes = 38
    height, width = 224, 224
    
    # 模拟模型输出
    outputs = {
        "cls_logits": torch.randn(batch_size, num_classes),
        "seg_mask": torch.randn(batch_size, 1, height, width),
        "sep_heatmaps": torch.randn(batch_size, 8, height, width),
    }
    
    # 模拟真实标签
    targets = {
        "cls_label": torch.randint(0, num_classes, (batch_size,)),
        "seg_mask": torch.randint(0, 2, (batch_size, 1, height, width)).float(),
        "sep_target": torch.softmax(torch.randn(batch_size, 8, height, width), dim=1),
    }
    
    # 测试 DiceLoss
    print("\n1. DiceLoss:")
    dice_loss = DiceLoss()
    pred_seg = torch.sigmoid(outputs["seg_mask"])
    loss = dice_loss(pred_seg, targets["seg_mask"])
    print(f"   Loss value: {loss.item():.4f}")
    
    # 测试 BCEDiceLoss
    print("\n2. BCEDiceLoss:")
    bce_dice_loss = BCEDiceLoss()
    loss = bce_dice_loss(outputs["seg_mask"], targets["seg_mask"])
    print(f"   Loss value: {loss.item():.4f}")
    
    # 测试 FocalLoss
    print("\n3. FocalLoss:")
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(outputs["cls_logits"], targets["cls_label"])
    print(f"   Loss value: {loss.item():.4f}")
    
    # 测试 ClassBalancedLoss
    print("\n4. ClassBalancedLoss:")
    num_per_class = [100 + i * 10 for i in range(num_classes)]  # 模拟不平衡分布
    cb_loss = ClassBalancedLoss(num_per_class=num_per_class, beta=0.9999)
    loss = cb_loss(outputs["cls_logits"], targets["cls_label"])
    print(f"   Loss value: {loss.item():.4f}")
    
    # 测试 MultiTaskLoss
    print("\n5. MultiTaskLoss (cross_entropy + bce_dice):")
    mt_loss = MultiTaskLoss(
        classification_loss="cross_entropy",
        segmentation_loss="bce_dice",
        weights=[1.0, 1.0, 0.5],
    )
    losses = mt_loss(outputs, targets)
    print(f"   Total loss: {losses['total'].item():.4f}")
    print(f"   Classification loss: {losses['cls'].item():.4f}")
    print(f"   Segmentation loss: {losses['seg'].item():.4f}")
    if "sep" in losses:
        print(f"   Separation loss: {losses['sep'].item():.4f}")
    
    # 测试 MultiTaskLoss with Focal
    print("\n6. MultiTaskLoss (focal + bce_dice):")
    mt_loss_focal = MultiTaskLoss(
        classification_loss="focal",
        segmentation_loss="bce_dice",
        weights=[1.0, 1.0, 0.5],
        focal_gamma=2.0,
    )
    losses = mt_loss_focal(outputs, targets)
    print(f"   Total loss: {losses['total'].item():.4f}")
    print(f"   Classification loss: {losses['cls'].item():.4f}")
    print(f"   Segmentation loss: {losses['seg'].item():.4f}")
    
    # 测试 MultiTaskLoss with ClassBalanced
    print("\n7. MultiTaskLoss (class_balanced + bce_dice):")
    mt_loss_cb = MultiTaskLoss(
        classification_loss="class_balanced",
        segmentation_loss="bce_dice",
        weights=[1.0, 1.0, 0.5],
        num_per_class=num_per_class,
    )
    losses = mt_loss_cb(outputs, targets)
    print(f"   Total loss: {losses['total'].item():.4f}")
    print(f"   Classification loss: {losses['cls'].item():.4f}")
    print(f"   Segmentation loss: {losses['seg'].item():.4f}")
    
    print("\n" + "=" * 60)
    print("All loss function tests passed!")
    print("=" * 60)
