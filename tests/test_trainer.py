# -*- coding: utf-8 -*-
"""
训练器测试

测试 Trainer 类的检查点保存/加载功能

Requirements: 3.4
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.training.trainer import Trainer
from src.training.utils import set_seed, get_git_commit, save_meta_json, load_meta_json


class SimpleModel(nn.Module):
    """用于测试的简单模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return {'cls_logits': self.fc(x), 'seg_mask': x.unsqueeze(1)}


class SimpleLoss(nn.Module):
    """用于测试的简单损失函数"""
    def forward(self, outputs, targets):
        cls_loss = nn.functional.cross_entropy(outputs['cls_logits'], targets['cls_label'])
        return {'total': cls_loss, 'cls': cls_loss}


class TestCheckpointSaveLoad:
    """检查点保存/加载测试
    
    _Requirements: 3.4_
    """
    
    def test_save_checkpoint_creates_file(self):
        """测试保存检查点创建文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            criterion = SimpleLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device="cpu",
                amp_enabled=False,
                output_dir=tmpdir,
            )
            
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer.save_checkpoint(ckpt_path)
            
            assert ckpt_path.exists(), "Checkpoint file should be created"
    
    def test_checkpoint_contains_required_keys(self):
        """测试检查点包含必需的键"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            criterion = SimpleLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device="cpu",
                amp_enabled=False,
                output_dir=tmpdir,
            )
            
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer.save_checkpoint(ckpt_path)
            
            checkpoint = torch.load(ckpt_path, weights_only=False)
            
            required_keys = [
                'epoch',
                'global_step',
                'model_state_dict',
                'optimizer_state_dict',
                'best_metric',
                'history',
                'config',
            ]
            
            for key in required_keys:
                assert key in checkpoint, f"Checkpoint should contain '{key}'"
    
    def test_load_checkpoint_restores_model_weights(self):
        """测试加载检查点恢复模型权重"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建并保存第一个 trainer
            set_seed(42)
            model1 = SimpleModel()
            criterion = SimpleLoss()
            optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
            
            trainer1 = Trainer(
                model=model1,
                criterion=criterion,
                optimizer=optimizer1,
                device="cpu",
                amp_enabled=False,
                output_dir=tmpdir,
            )
            
            # 修改模型权重
            with torch.no_grad():
                model1.fc.weight.fill_(1.0)
                model1.fc.bias.fill_(0.5)
            
            trainer1.current_epoch = 5
            trainer1.global_step = 100
            trainer1.best_metric = 0.95
            
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer1.save_checkpoint(ckpt_path)
            
            # 创建新的 trainer 并加载检查点
            set_seed(123)  # 不同的种子
            model2 = SimpleModel()
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
            
            trainer2 = Trainer(
                model=model2,
                criterion=criterion,
                optimizer=optimizer2,
                device="cpu",
                amp_enabled=False,
                output_dir=tmpdir,
            )
            
            trainer2.load_checkpoint(ckpt_path)
            
            # 验证模型权重恢复
            assert torch.allclose(model2.fc.weight, torch.ones_like(model2.fc.weight)), \
                "Model weights should be restored"
            assert torch.allclose(model2.fc.bias, torch.full_like(model2.fc.bias, 0.5)), \
                "Model bias should be restored"
    
    def test_load_checkpoint_restores_optimizer_state(self):
        """测试加载检查点恢复优化器状态"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model1 = SimpleModel()
            criterion = SimpleLoss()
            optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
            
            trainer1 = Trainer(
                model=model1,
                criterion=criterion,
                optimizer=optimizer1,
                device="cpu",
                amp_enabled=False,
                output_dir=tmpdir,
            )
            
            # 执行一些优化步骤以更新优化器状态
            x = torch.randn(4, 10)
            targets = {'cls_label': torch.randint(0, 2, (4,))}
            outputs = model1(x)
            loss = criterion(outputs, targets)['total']
            loss.backward()
            optimizer1.step()
            
            # 保存检查点
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer1.save_checkpoint(ckpt_path)
            
            # 获取原始优化器状态
            original_state = optimizer1.state_dict()
            
            # 创建新的 trainer 并加载检查点
            model2 = SimpleModel()
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
            
            trainer2 = Trainer(
                model=model2,
                criterion=criterion,
                optimizer=optimizer2,
                device="cpu",
                amp_enabled=False,
                output_dir=tmpdir,
            )
            
            trainer2.load_checkpoint(ckpt_path)
            
            # 验证优化器状态恢复
            loaded_state = optimizer2.state_dict()
            assert len(loaded_state['state']) == len(original_state['state']), \
                "Optimizer state should be restored"
    
    def test_load_checkpoint_restores_training_state(self):
        """测试加载检查点恢复训练状态（epoch, global_step, best_metric）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            criterion = SimpleLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            trainer1 = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device="cpu",
                amp_enabled=False,
                output_dir=tmpdir,
            )
            
            # 设置训练状态
            trainer1.current_epoch = 10
            trainer1.global_step = 500
            trainer1.best_metric = 0.88
            
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer1.save_checkpoint(ckpt_path)
            
            # 创建新的 trainer 并加载检查点
            model2 = SimpleModel()
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
            
            trainer2 = Trainer(
                model=model2,
                criterion=criterion,
                optimizer=optimizer2,
                device="cpu",
                amp_enabled=False,
                output_dir=tmpdir,
            )
            
            trainer2.load_checkpoint(ckpt_path)
            
            # 验证训练状态恢复（注意：load_checkpoint 会将 epoch + 1）
            assert trainer2.current_epoch == 11, "Epoch should be restored (+ 1 for resume)"
            assert trainer2.global_step == 500, "Global step should be restored"
            assert trainer2.best_metric == 0.88, "Best metric should be restored"
    
    def test_checkpoint_with_scheduler(self):
        """测试带调度器的检查点保存/加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            criterion = SimpleLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            
            trainer1 = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device="cpu",
                amp_enabled=False,
                output_dir=tmpdir,
            )
            
            # 执行几步调度器
            for _ in range(5):
                scheduler.step()
            
            ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer1.save_checkpoint(ckpt_path)
            
            # 验证检查点包含调度器状态
            checkpoint = torch.load(ckpt_path, weights_only=False)
            assert 'scheduler_state_dict' in checkpoint, \
                "Checkpoint should contain scheduler state"
            
            # 创建新的 trainer 并加载检查点
            model2 = SimpleModel()
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.1)
            
            trainer2 = Trainer(
                model=model2,
                criterion=criterion,
                optimizer=optimizer2,
                scheduler=scheduler2,
                device="cpu",
                amp_enabled=False,
                output_dir=tmpdir,
            )
            
            trainer2.load_checkpoint(ckpt_path)
            
            # 验证调度器状态恢复
            assert scheduler2.last_epoch == scheduler.last_epoch, \
                "Scheduler state should be restored"


class TestTrainingUtils:
    """训练工具函数测试
    
    _Requirements: 7.3, 7.4_
    """
    
    def test_set_seed_reproducibility(self):
        """测试 set_seed 确保可复现性"""
        set_seed(42)
        tensor1 = torch.randn(10)
        
        set_seed(42)
        tensor2 = torch.randn(10)
        
        assert torch.allclose(tensor1, tensor2), "Same seed should produce same random values"
    
    def test_get_git_commit_returns_string(self):
        """测试 get_git_commit 返回字符串"""
        commit = get_git_commit()
        assert isinstance(commit, str), "Git commit should be a string"
        # 应该返回 8 字符的短 hash 或 "unknown"
        assert len(commit) == 8 or commit == "unknown", \
            "Git commit should be 8 chars or 'unknown'"
    
    def test_save_and_load_meta_json(self):
        """测试 save_meta_json 和 load_meta_json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_meta_json(
                output_dir=tmpdir,
                seed=42,
                extra_info={'test_key': 'test_value'}
            )
            
            meta = load_meta_json(tmpdir)
            
            assert 'git_commit' in meta, "Meta should contain git_commit"
            assert 'seed' in meta, "Meta should contain seed"
            assert meta['seed'] == 42, "Seed should be 42"
            assert 'timestamp' in meta, "Meta should contain timestamp"
            assert meta.get('test_key') == 'test_value', "Extra info should be saved"
    
    def test_meta_json_contains_required_fields(self):
        """测试 meta.json 包含必需字段"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_meta_json(output_dir=tmpdir, seed=123)
            
            meta = load_meta_json(tmpdir)
            
            required_fields = ['git_commit', 'seed', 'timestamp', 'pytorch_version']
            for field in required_fields:
                assert field in meta, f"Meta should contain '{field}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
