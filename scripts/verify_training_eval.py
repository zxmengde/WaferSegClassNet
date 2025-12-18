# -*- coding: utf-8 -*-
"""
训练评估模块验证脚本

Checkpoint 11: 验证训练和评估模块的端到端集成

Requirements: 3.2, 3.3, 3.4, 3.5, 3.6
"""

import sys
import tempfile
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.multitask import WaferMultiTaskModel, create_model
from src.models.losses import create_loss
from src.training.trainer import Trainer
from src.training.utils import set_seed, save_meta_json, load_meta_json
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_macro_f1, compute_dice, compute_iou
from src.visualization.plots import plot_confusion_matrix, plot_loss_curves

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockDataset(torch.utils.data.Dataset):
    """模拟数据集"""
    def __init__(self, size: int = 50, num_classes: int = 10):
        self.size = size
        self.num_classes = num_classes
        np.random.seed(42)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'image': torch.randn(3, 64, 64),
            'label': torch.tensor(idx % self.num_classes),
            'mask': (torch.rand(1, 64, 64) > 0.5).float(),
        }


def test_trainer_basic():
    """测试训练器基本功能"""
    print("\n" + "=" * 60)
    print("1. 测试训练器基本功能")
    print("=" * 60)
    
    set_seed(42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建模型和损失函数
        model = WaferMultiTaskModel(
            classification_classes=10,
            separation_enabled=False,
        )
        
        criterion = create_loss(
            classification_loss="cross_entropy",
            segmentation_loss="bce_dice",
            num_classes=10,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 创建数据加载器
        train_dataset = MockDataset(size=32, num_classes=10)
        val_dataset = MockDataset(size=16, num_classes=10)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
            amp_enabled=False,
            output_dir=tmpdir,
            seed=42,
        )
        
        # 训练 2 个 epoch
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            save_every=1,
        )
        
        # 验证输出
        output_dir = Path(tmpdir)
        assert (output_dir / "checkpoints" / "last.pt").exists(), "last.pt 应该存在"
        assert (output_dir / "history.json").exists(), "history.json 应该存在"
        assert (output_dir / "meta.json").exists(), "meta.json 应该存在"
        
        # 验证历史记录
        assert len(history['train_loss']) == 2, "应该有 2 个 epoch 的训练损失"
        assert len(history['val_loss']) == 2, "应该有 2 个 epoch 的验证损失"
        
        print("  ✓ 训练器基本功能测试通过")
        return True


def test_checkpoint_resume():
    """测试断点续训"""
    print("\n" + "=" * 60)
    print("2. 测试断点续训")
    print("=" * 60)
    
    set_seed(42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 第一次训练
        model1 = WaferMultiTaskModel(
            classification_classes=10,
            separation_enabled=False,
        )
        
        criterion = create_loss(
            classification_loss="cross_entropy",
            segmentation_loss="bce_dice",
            num_classes=10,
        )
        
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        
        train_dataset = MockDataset(size=32, num_classes=10)
        val_dataset = MockDataset(size=16, num_classes=10)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
        
        trainer1 = Trainer(
            model=model1,
            criterion=criterion,
            optimizer=optimizer1,
            device="cpu",
            amp_enabled=False,
            output_dir=tmpdir,
        )
        
        # 训练 2 个 epoch
        trainer1.train(train_loader, val_loader, epochs=2, save_every=1)
        
        # 第二次训练（续训）
        model2 = WaferMultiTaskModel(
            classification_classes=10,
            separation_enabled=False,
        )
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        
        trainer2 = Trainer(
            model=model2,
            criterion=criterion,
            optimizer=optimizer2,
            device="cpu",
            amp_enabled=False,
            output_dir=tmpdir,
        )
        
        # 从检查点恢复并继续训练
        ckpt_path = Path(tmpdir) / "checkpoints" / "last.pt"
        trainer2.train(
            train_loader, val_loader,
            epochs=4,
            resume_from=str(ckpt_path),
            save_every=1,
        )
        
        # 验证续训后的 epoch
        assert trainer2.current_epoch == 3, f"当前 epoch 应该是 3，实际是 {trainer2.current_epoch}"
        assert len(trainer2.history['train_loss']) == 4, "应该有 4 个 epoch 的训练损失"
        
        print("  ✓ 断点续训测试通过")
        return True


def test_evaluator_basic():
    """测试评估器基本功能"""
    print("\n" + "=" * 60)
    print("3. 测试评估器基本功能")
    print("=" * 60)
    
    set_seed(42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建模型
        model = WaferMultiTaskModel(
            classification_classes=10,
            separation_enabled=False,
        )
        
        # 创建数据加载器
        test_dataset = MockDataset(size=20, num_classes=10)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
        
        # 创建评估器
        evaluator = Evaluator(
            model=model,
            device="cpu",
            amp_enabled=False,
            num_classes=10,
            output_dir=tmpdir,
        )
        
        # 运行评估
        metrics = evaluator.evaluate(test_loader, save_visualizations=False)
        
        # 验证指标
        assert 'macro_f1' in metrics, "应该包含 macro_f1"
        assert 'dice' in metrics, "应该包含 dice"
        assert 'iou' in metrics, "应该包含 iou"
        assert 'accuracy' in metrics, "应该包含 accuracy"
        
        # 验证指标范围
        assert 0.0 <= metrics['macro_f1'] <= 1.0, "macro_f1 应该在 [0, 1] 范围内"
        assert 0.0 <= metrics['dice'] <= 1.0, "dice 应该在 [0, 1] 范围内"
        assert 0.0 <= metrics['iou'] <= 1.0, "iou 应该在 [0, 1] 范围内"
        
        # 验证输出文件
        output_dir = Path(tmpdir)
        assert (output_dir / "metrics.csv").exists(), "metrics.csv 应该存在"
        
        print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"  Dice: {metrics['dice']:.4f}")
        print(f"  IoU: {metrics['iou']:.4f}")
        print("  ✓ 评估器基本功能测试通过")
        return True


def test_visualization():
    """测试可视化功能"""
    print("\n" + "=" * 60)
    print("4. 测试可视化功能")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 测试混淆矩阵
        cm = np.random.randint(0, 50, (10, 10))
        fig = plot_confusion_matrix(cm, save_path=str(output_dir / "cm.png"))
        assert (output_dir / "cm.png").exists(), "混淆矩阵图应该存在"
        
        # 测试损失曲线
        history = {
            'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
            'val_loss': [1.1, 0.9, 0.75, 0.65, 0.55],
        }
        fig = plot_loss_curves(history, save_path=str(output_dir / "loss.png"))
        assert (output_dir / "loss.png").exists(), "损失曲线图应该存在"
        
        print("  ✓ 可视化功能测试通过")
        return True


def test_metrics_calculation():
    """测试指标计算"""
    print("\n" + "=" * 60)
    print("5. 测试指标计算")
    print("=" * 60)
    
    # 测试 Macro-F1
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    macro_f1, per_class_f1 = compute_macro_f1(y_true, y_pred, num_classes=3)
    assert macro_f1 == 1.0, "完美预测的 Macro-F1 应该是 1.0"
    
    # 测试 Dice
    pred = np.array([[[1, 1], [1, 0]]])
    target = np.array([[[1, 1], [1, 0]]])
    dice = compute_dice(pred, target, threshold=0.5)
    assert abs(dice - 1.0) < 0.01, "完美重叠的 Dice 应该接近 1.0"
    
    # 测试 IoU
    iou = compute_iou(pred, target, threshold=0.5)
    assert abs(iou - 1.0) < 0.01, "完美重叠的 IoU 应该接近 1.0"
    
    print("  ✓ 指标计算测试通过")
    return True


def test_meta_json():
    """测试 meta.json 保存和加载"""
    print("\n" + "=" * 60)
    print("6. 测试 meta.json 功能")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 保存 meta.json
        save_meta_json(
            output_dir=tmpdir,
            seed=42,
            extra_info={'test_key': 'test_value'}
        )
        
        # 加载 meta.json
        meta = load_meta_json(tmpdir)
        
        # 验证字段
        assert 'git_commit' in meta, "应该包含 git_commit"
        assert 'seed' in meta, "应该包含 seed"
        assert meta['seed'] == 42, "seed 应该是 42"
        assert 'timestamp' in meta, "应该包含 timestamp"
        assert meta.get('test_key') == 'test_value', "应该包含额外信息"
        
        print("  ✓ meta.json 功能测试通过")
        return True


def test_end_to_end():
    """端到端测试：训练 -> 保存 -> 加载 -> 评估"""
    print("\n" + "=" * 60)
    print("7. 端到端测试")
    print("=" * 60)
    
    set_seed(42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 1. 创建模型和训练
        model = WaferMultiTaskModel(
            classification_classes=10,
            separation_enabled=False,
        )
        
        criterion = create_loss(
            classification_loss="cross_entropy",
            segmentation_loss="bce_dice",
            num_classes=10,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        train_dataset = MockDataset(size=32, num_classes=10)
        val_dataset = MockDataset(size=16, num_classes=10)
        test_dataset = MockDataset(size=20, num_classes=10)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
        
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
            amp_enabled=False,
            output_dir=str(output_dir),
            seed=42,
        )
        
        # 训练
        print("  训练中...")
        history = trainer.train(train_loader, val_loader, epochs=2, save_every=1)
        
        # 2. 加载最佳模型并评估
        print("  加载模型并评估...")
        model_eval = WaferMultiTaskModel(
            classification_classes=10,
            separation_enabled=False,
        )
        
        ckpt = torch.load(output_dir / "checkpoints" / "last.pt", weights_only=False)
        model_eval.load_state_dict(ckpt['model_state_dict'])
        
        evaluator = Evaluator(
            model=model_eval,
            device="cpu",
            amp_enabled=False,
            num_classes=10,
            output_dir=str(output_dir),
        )
        
        metrics = evaluator.evaluate(test_loader, save_visualizations=False)
        
        # 3. 验证输出结构
        print("  验证输出结构...")
        assert (output_dir / "checkpoints" / "last.pt").exists()
        assert (output_dir / "history.json").exists()
        assert (output_dir / "meta.json").exists()
        assert (output_dir / "metrics.csv").exists()
        
        print(f"  最终 Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"  最终 Dice: {metrics['dice']:.4f}")
        print("  ✓ 端到端测试通过")
        return True


def main():
    """运行所有验证测试"""
    print("=" * 60)
    print("训练评估模块验证 - Checkpoint 11")
    print("=" * 60)
    
    tests = [
        ("训练器基本功能", test_trainer_basic),
        ("断点续训", test_checkpoint_resume),
        ("评估器基本功能", test_evaluator_basic),
        ("可视化功能", test_visualization),
        ("指标计算", test_metrics_calculation),
        ("meta.json 功能", test_meta_json),
        ("端到端测试", test_end_to_end),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {name} 失败")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"验证结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ 所有训练评估模块验证通过！")
        print("  可以进入 Phase 5: 统一入口与Debug模式")
        return 0
    else:
        print("\n✗ 部分验证失败，请检查问题")
        return 1


if __name__ == "__main__":
    sys.exit(main())
