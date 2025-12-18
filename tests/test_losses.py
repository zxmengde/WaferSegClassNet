# -*- coding: utf-8 -*-
"""
损失函数测试

测试 DiceLoss, BCEDiceLoss, FocalLoss, ClassBalancedLoss, MultiTaskLoss
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.losses import (
    DiceLoss,
    BCEDiceLoss,
    FocalLoss,
    ClassBalancedLoss,
    KLDivergenceLoss,
    MSELoss,
    MultiTaskLoss,
    dice_coef,
    create_loss,
    create_loss_from_config,
)


class TestDiceLoss:
    """DiceLoss 测试"""
    
    def test_perfect_prediction_returns_zero_loss(self):
        """完美预测应返回接近 0 的损失"""
        dice_loss = DiceLoss()
        pred = torch.ones(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        loss = dice_loss(pred, target)
        assert loss.item() < 0.01
    
    def test_completely_wrong_prediction_returns_high_loss(self):
        """完全错误的预测应返回接近 1 的损失"""
        dice_loss = DiceLoss()
        pred = torch.zeros(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        loss = dice_loss(pred, target)
        assert loss.item() > 0.99
    
    def test_loss_is_symmetric(self):
        """Dice 损失应该是对称的"""
        dice_loss = DiceLoss()
        pred = torch.rand(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        loss1 = dice_loss(pred, target)
        loss2 = dice_loss(target, pred)
        
        # 由于 smooth 因子，可能有微小差异
        assert abs(loss1.item() - loss2.item()) < 0.1
    
    def test_loss_in_valid_range(self):
        """损失值应在 [0, 1] 范围内"""
        dice_loss = DiceLoss()
        pred = torch.rand(4, 1, 64, 64)
        target = torch.randint(0, 2, (4, 1, 64, 64)).float()
        loss = dice_loss(pred, target)
        assert 0 <= loss.item() <= 1


class TestBCEDiceLoss:
    """BCEDiceLoss 测试"""
    
    def test_loss_is_positive(self):
        """损失值应为正数"""
        bce_dice = BCEDiceLoss()
        pred = torch.randn(2, 1, 32, 32)  # logits
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        loss = bce_dice(pred, target)
        assert loss.item() > 0
    
    def test_perfect_prediction_low_loss(self):
        """完美预测应有较低损失"""
        bce_dice = BCEDiceLoss()
        # 使用大的正值 logits 表示预测为 1
        pred = torch.ones(2, 1, 32, 32) * 10
        target = torch.ones(2, 1, 32, 32)
        loss = bce_dice(pred, target)
        assert loss.item() < 0.1
    
    def test_custom_weights(self):
        """自定义权重应影响损失值"""
        bce_dice_equal = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        bce_dice_bce_heavy = BCEDiceLoss(bce_weight=0.9, dice_weight=0.1)
        
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        loss1 = bce_dice_equal(pred, target)
        loss2 = bce_dice_bce_heavy(pred, target)
        
        # 两个损失应该不同
        assert loss1.item() != loss2.item()


class TestFocalLoss:
    """FocalLoss 测试"""
    
    def test_focal_loss_reduces_easy_sample_weight(self):
        """Focal Loss 应降低易分类样本的权重"""
        focal = FocalLoss(gamma=2.0)
        ce = torch.nn.CrossEntropyLoss()
        
        # 创建一个"容易"的样本（高置信度正确预测）
        pred_easy = torch.tensor([[10.0, -10.0, -10.0]])  # 高置信度预测类别 0
        target_easy = torch.tensor([0])
        
        # 创建一个"困难"的样本（低置信度）
        pred_hard = torch.tensor([[0.1, 0.0, 0.0]])  # 低置信度
        target_hard = torch.tensor([0])
        
        focal_easy = focal(pred_easy, target_easy)
        focal_hard = focal(pred_hard, target_hard)
        
        # Focal Loss 对容易样本的惩罚应该更小
        assert focal_easy.item() < focal_hard.item()
    
    def test_gamma_zero_equals_cross_entropy(self):
        """gamma=0 时应等同于交叉熵"""
        focal = FocalLoss(gamma=0.0)
        ce = torch.nn.CrossEntropyLoss()
        
        pred = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        
        focal_loss = focal(pred, target)
        ce_loss = ce(pred, target)
        
        assert abs(focal_loss.item() - ce_loss.item()) < 1e-5
    
    def test_loss_is_positive(self):
        """损失值应为正数"""
        focal = FocalLoss(gamma=2.0)
        pred = torch.randn(4, 38)
        target = torch.randint(0, 38, (4,))
        loss = focal(pred, target)
        assert loss.item() > 0


class TestClassBalancedLoss:
    """ClassBalancedLoss 测试"""
    
    def test_imbalanced_classes_get_higher_weight(self):
        """样本少的类别应获得更高权重"""
        # 创建不平衡的类别分布
        num_per_class = [1000, 100, 10]  # 类别 2 样本最少
        cb_loss = ClassBalancedLoss(num_per_class=num_per_class, beta=0.9999)
        
        # 检查权重
        weights = cb_loss.weights
        assert weights[2] > weights[1] > weights[0]
    
    def test_loss_is_positive(self):
        """损失值应为正数"""
        num_per_class = [100] * 38
        cb_loss = ClassBalancedLoss(num_per_class=num_per_class)
        
        pred = torch.randn(4, 38)
        target = torch.randint(0, 38, (4,))
        loss = cb_loss(pred, target)
        assert loss.item() > 0
    
    def test_beta_affects_weights(self):
        """不同的 beta 值应产生不同的权重"""
        num_per_class = [1000, 100, 10]
        
        cb_low_beta = ClassBalancedLoss(num_per_class=num_per_class, beta=0.9)
        cb_high_beta = ClassBalancedLoss(num_per_class=num_per_class, beta=0.9999)
        
        # 高 beta 应该产生更极端的权重差异
        low_ratio = cb_low_beta.weights[2] / cb_low_beta.weights[0]
        high_ratio = cb_high_beta.weights[2] / cb_high_beta.weights[0]
        
        assert high_ratio > low_ratio


class TestMultiTaskLoss:
    """MultiTaskLoss 测试"""
    
    @pytest.fixture
    def sample_outputs(self):
        """创建示例模型输出"""
        return {
            "cls_logits": torch.randn(4, 38),
            "seg_mask": torch.randn(4, 1, 64, 64),
            "sep_heatmaps": torch.randn(4, 8, 64, 64),
        }
    
    @pytest.fixture
    def sample_targets(self):
        """创建示例真实标签"""
        return {
            "cls_label": torch.randint(0, 38, (4,)),
            "seg_mask": torch.randint(0, 2, (4, 1, 64, 64)).float(),
            "sep_target": F.softmax(torch.randn(4, 8, 64, 64), dim=1),
        }
    
    def test_returns_all_loss_components(self, sample_outputs, sample_targets):
        """应返回所有损失组件"""
        mt_loss = MultiTaskLoss()
        losses = mt_loss(sample_outputs, sample_targets)
        
        assert "total" in losses
        assert "cls" in losses
        assert "seg" in losses
        assert "sep" in losses
    
    def test_total_loss_is_weighted_sum(self, sample_outputs, sample_targets):
        """总损失应为加权和"""
        weights = [1.0, 2.0, 0.5]
        mt_loss = MultiTaskLoss(weights=weights)
        losses = mt_loss(sample_outputs, sample_targets)
        
        expected_total = (
            weights[0] * losses["cls"] +
            weights[1] * losses["seg"] +
            weights[2] * losses["sep"]
        )
        
        assert abs(losses["total"].item() - expected_total.item()) < 1e-5
    
    def test_cross_entropy_classification(self, sample_outputs, sample_targets):
        """测试交叉熵分类损失"""
        mt_loss = MultiTaskLoss(classification_loss="cross_entropy")
        losses = mt_loss(sample_outputs, sample_targets)
        assert losses["cls"].item() > 0
    
    def test_focal_classification(self, sample_outputs, sample_targets):
        """测试 Focal 分类损失"""
        mt_loss = MultiTaskLoss(classification_loss="focal", focal_gamma=2.0)
        losses = mt_loss(sample_outputs, sample_targets)
        assert losses["cls"].item() > 0
    
    def test_class_balanced_classification(self, sample_outputs, sample_targets):
        """测试 Class-Balanced 分类损失"""
        num_per_class = [100] * 38
        mt_loss = MultiTaskLoss(
            classification_loss="class_balanced",
            num_per_class=num_per_class,
        )
        losses = mt_loss(sample_outputs, sample_targets)
        assert losses["cls"].item() > 0
    
    def test_bce_dice_segmentation(self, sample_outputs, sample_targets):
        """测试 BCE+Dice 分割损失"""
        mt_loss = MultiTaskLoss(segmentation_loss="bce_dice")
        losses = mt_loss(sample_outputs, sample_targets)
        assert losses["seg"].item() > 0
    
    def test_dice_only_segmentation(self, sample_outputs, sample_targets):
        """测试纯 Dice 分割损失"""
        mt_loss = MultiTaskLoss(segmentation_loss="dice")
        losses = mt_loss(sample_outputs, sample_targets)
        assert losses["seg"].item() > 0
    
    def test_bce_only_segmentation(self, sample_outputs, sample_targets):
        """测试纯 BCE 分割损失"""
        mt_loss = MultiTaskLoss(segmentation_loss="bce")
        losses = mt_loss(sample_outputs, sample_targets)
        assert losses["seg"].item() > 0
    
    def test_without_separation(self, sample_outputs, sample_targets):
        """测试无分离任务时的损失"""
        outputs_no_sep = {k: v for k, v in sample_outputs.items() if k != "sep_heatmaps"}
        targets_no_sep = {k: v for k, v in sample_targets.items() if k != "sep_target"}
        
        mt_loss = MultiTaskLoss()
        losses = mt_loss(outputs_no_sep, targets_no_sep)
        
        assert "total" in losses
        assert "cls" in losses
        assert "seg" in losses
        assert "sep" not in losses


class TestDiceCoef:
    """dice_coef 函数测试"""
    
    def test_perfect_overlap_returns_one(self):
        """完美重叠应返回 1"""
        pred = torch.ones(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        dice = dice_coef(pred, target)
        assert dice.item() > 0.99
    
    def test_no_overlap_returns_zero(self):
        """无重叠应返回接近 0"""
        pred = torch.zeros(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        dice = dice_coef(pred, target)
        assert dice.item() < 0.01
    
    def test_partial_overlap(self):
        """部分重叠应返回中间值"""
        pred = torch.zeros(2, 1, 32, 32)
        pred[:, :, :16, :] = 1.0
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, :] = 1.0
        
        dice = dice_coef(pred, target)
        assert 0.3 < dice.item() < 0.7


class TestCreateLoss:
    """create_loss 和 create_loss_from_config 测试"""
    
    def test_create_loss_default(self):
        """测试默认参数创建损失"""
        loss = create_loss()
        assert isinstance(loss, MultiTaskLoss)
    
    def test_create_loss_with_focal(self):
        """测试使用 Focal Loss 创建"""
        loss = create_loss(classification_loss="focal", focal_gamma=3.0)
        assert isinstance(loss, MultiTaskLoss)
        assert loss.cls_loss_type == "focal"
    
    def test_create_loss_from_config_dict(self):
        """测试从字典配置创建"""
        config = {
            "classification": "focal",
            "segmentation": "dice",
            "separation": "mse",
            "weights": [1.0, 0.5, 0.3],
            "focal_gamma": 2.5,
        }
        loss = create_loss_from_config(config)
        assert isinstance(loss, MultiTaskLoss)
        assert loss.cls_loss_type == "focal"
        assert loss.seg_loss_type == "dice"
        assert loss.sep_loss_type == "mse"


class TestKLDivergenceLoss:
    """KLDivergenceLoss 测试"""
    
    def test_same_distribution_low_loss(self):
        """相同分布应有低损失"""
        kl_loss = KLDivergenceLoss()
        target = F.softmax(torch.randn(2, 8, 16, 16), dim=1)
        # 使用相同的分布作为预测（转换为 logits）
        pred = torch.log(target + 1e-8)
        
        loss = kl_loss(pred * 1.0, target)  # temperature=1
        assert loss.item() < 0.1
    
    def test_loss_is_non_negative(self):
        """KL 散度应非负"""
        kl_loss = KLDivergenceLoss()
        pred = torch.randn(2, 8, 16, 16)
        target = F.softmax(torch.randn(2, 8, 16, 16), dim=1)
        
        loss = kl_loss(pred, target)
        assert loss.item() >= 0


class TestMSELoss:
    """MSELoss 测试"""
    
    def test_same_values_zero_loss(self):
        """相同值应有零损失"""
        mse_loss = MSELoss()
        # 使用 sigmoid 后相同的值
        pred = torch.zeros(2, 8, 16, 16)  # sigmoid(0) = 0.5
        target = torch.ones(2, 8, 16, 16) * 0.5
        
        loss = mse_loss(pred, target)
        assert loss.item() < 0.01
    
    def test_loss_is_positive(self):
        """损失应为正数"""
        mse_loss = MSELoss()
        pred = torch.randn(2, 8, 16, 16)
        target = torch.rand(2, 8, 16, 16)
        
        loss = mse_loss(pred, target)
        assert loss.item() > 0
