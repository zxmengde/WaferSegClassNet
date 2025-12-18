# -*- coding: utf-8 -*-
"""
训练器模块

提供统一的训练接口，支持：
- AMP 混合精度训练
- 断点续训
- 可配置的 best checkpoint 指标
- 保存 meta.json（git_commit, seed）

Requirements: 3.2, 3.3, 3.4, 7.3
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from .utils import get_git_commit, save_config_snapshot, save_meta_json

logger = logging.getLogger(__name__)


class Trainer:
    """
    统一训练器
    
    功能:
    - AMP 混合精度
    - 断点续训
    - 最佳模型保存（可配置指标）
    - 训练曲线记录
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        amp_enabled: bool = True,
        checkpoint_metric: str = "macro_f1",
        checkpoint_metric_mode: str = "max",
        grad_accum_steps: int = 1,
        output_dir: str = "results/experiment",
        seed: int = 42,
        config: Optional[Any] = None,
    ):
        """
        Args:
            model: 模型
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            amp_enabled: 是否启用 AMP
            checkpoint_metric: 最佳模型选择指标
            checkpoint_metric_mode: 指标模式 (max | min)
            grad_accum_steps: 梯度累积步数
            output_dir: 输出目录
            seed: 随机种子
            config: 配置对象（用于保存 config_snapshot）
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp_enabled = amp_enabled and torch.cuda.is_available()
        self.checkpoint_metric = checkpoint_metric
        self.checkpoint_metric_mode = checkpoint_metric_mode
        self.grad_accum_steps = grad_accum_steps
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.config = config
        
        # AMP scaler
        self.scaler = GradScaler() if self.amp_enabled else None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf') if checkpoint_metric_mode == "max" else float('inf')
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_cls_loss': [],
            'train_seg_loss': [],
            'val_cls_loss': [],
            'val_seg_loss': [],
            'learning_rate': [],
        }
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "curves").mkdir(exist_ok=True)
        
        # 保存配置和元信息
        self._save_initial_meta()
    
    def _save_initial_meta(self):
        """保存初始元信息"""
        # 保存 config_snapshot
        if self.config is not None:
            save_config_snapshot(self.config, self.output_dir / "config_snapshot.yaml")
        
        # 保存 meta.json
        save_meta_json(
            output_dir=self.output_dir,
            seed=self.seed,
            extra_info={
                'amp_enabled': self.amp_enabled,
                'checkpoint_metric': self.checkpoint_metric,
                'grad_accum_steps': self.grad_accum_steps,
            }
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        resume_from: Optional[str] = None,
        val_metric_fn: Optional[Callable] = None,
        save_every: int = 10,
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            resume_from: 断点续训检查点路径
            val_metric_fn: 验证指标计算函数
            save_every: 每隔多少轮保存检查点
            
        Returns:
            训练历史
        """
        # 断点续训
        if resume_from:
            self.load_checkpoint(resume_from)
            logger.info(f"Resumed from epoch {self.current_epoch}")
        
        start_epoch = self.current_epoch
        total_epochs = epochs
        
        logger.info(f"Starting training from epoch {start_epoch} to {total_epochs}")
        logger.info(f"AMP enabled: {self.amp_enabled}")
        logger.info(f"Gradient accumulation steps: {self.grad_accum_steps}")
        
        for epoch in range(start_epoch, total_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个 epoch
            train_metrics = self._train_epoch(train_loader)
            
            # 验证
            val_metrics = self._validate_epoch(val_loader, val_metric_fn)
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', 0))
                else:
                    self.scheduler.step()
            
            # 记录历史
            self._update_history(train_metrics, val_metrics, current_lr)
            
            # 检查是否为最佳模型
            current_metric = val_metrics.get(self.checkpoint_metric, val_metrics.get('loss', 0))
            is_best = self._is_better(current_metric)
            if is_best:
                self.best_metric = current_metric
            
            # 保存检查点
            if (epoch + 1) % save_every == 0 or epoch == total_epochs - 1:
                self.save_checkpoint(
                    self.output_dir / "checkpoints" / "last.pt",
                    is_best=False
                )
            
            if is_best:
                self.save_checkpoint(
                    self.output_dir / "checkpoints" / "best.pt",
                    is_best=True
                )
            
            # 日志
            epoch_time = time.time() - epoch_start_time
            self._log_epoch(epoch, total_epochs, train_metrics, val_metrics, current_lr, epoch_time, is_best)
        
        # 保存训练历史
        self._save_history()
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_seg_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动数据到设备
            images = batch['image'].to(self.device)
            targets = {
                'cls_label': batch['label'].to(self.device),
                'seg_mask': batch['mask'].to(self.device),
            }
            
            # 前向传播（AMP）
            with autocast(enabled=self.amp_enabled):
                outputs = self.model(images)
                losses = self.criterion(outputs, targets)
                loss = losses['total'] / self.grad_accum_steps
            
            # 反向传播
            if self.amp_enabled:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.amp_enabled:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # 累计损失
            total_loss += losses['total'].item()
            if 'cls' in losses:
                total_cls_loss += losses['cls'].item()
            if 'seg' in losses:
                total_seg_loss += losses['seg'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'seg_loss': total_seg_loss / num_batches,
        }
    
    @torch.no_grad()
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        metric_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """验证一个 epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_seg_loss = 0.0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        all_seg_preds = []
        all_seg_masks = []
        
        for batch in val_loader:
            images = batch['image'].to(self.device)
            targets = {
                'cls_label': batch['label'].to(self.device),
                'seg_mask': batch['mask'].to(self.device),
            }
            
            with autocast(enabled=self.amp_enabled):
                outputs = self.model(images)
                losses = self.criterion(outputs, targets)
            
            total_loss += losses['total'].item()
            if 'cls' in losses:
                total_cls_loss += losses['cls'].item()
            if 'seg' in losses:
                total_seg_loss += losses['seg'].item()
            num_batches += 1
            
            # 收集预测结果
            all_preds.append(outputs['cls_logits'].argmax(dim=1).cpu())
            all_labels.append(targets['cls_label'].cpu())
            all_seg_preds.append(torch.sigmoid(outputs['seg_mask']).cpu())
            all_seg_masks.append(targets['seg_mask'].cpu())
        
        metrics = {
            'loss': total_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'seg_loss': total_seg_loss / num_batches,
        }
        
        # 计算额外指标
        if metric_fn is not None:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            all_seg_preds = torch.cat(all_seg_preds)
            all_seg_masks = torch.cat(all_seg_masks)
            
            extra_metrics = metric_fn(
                all_preds, all_labels,
                all_seg_preds, all_seg_masks
            )
            metrics.update(extra_metrics)
        
        return metrics
    
    def _is_better(self, current_metric: float) -> bool:
        """判断当前指标是否更好"""
        if self.checkpoint_metric_mode == "max":
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric
    
    def _update_history(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
    ):
        """更新训练历史"""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['train_cls_loss'].append(train_metrics.get('cls_loss', 0))
        self.history['train_seg_loss'].append(train_metrics.get('seg_loss', 0))
        self.history['val_cls_loss'].append(val_metrics.get('cls_loss', 0))
        self.history['val_seg_loss'].append(val_metrics.get('seg_loss', 0))
        self.history['learning_rate'].append(learning_rate)
        
        # 添加额外指标
        for key, value in val_metrics.items():
            if key not in ['loss', 'cls_loss', 'seg_loss']:
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
    
    def _log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
        epoch_time: float,
        is_best: bool,
    ):
        """记录 epoch 日志"""
        best_marker = " *" if is_best else ""
        logger.info(
            f"Epoch [{epoch+1}/{total_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"LR: {learning_rate:.6f} "
            f"Time: {epoch_time:.1f}s{best_marker}"
        )
        
        # 记录额外指标
        extra_metrics = {k: v for k, v in val_metrics.items() 
                        if k not in ['loss', 'cls_loss', 'seg_loss']}
        if extra_metrics:
            metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in extra_metrics.items()])
            logger.info(f"  Metrics: {metrics_str}")
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """
        保存检查点
        
        Args:
            path: 保存路径
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'history': self.history,
            'config': {
                'checkpoint_metric': self.checkpoint_metric,
                'checkpoint_metric_mode': self.checkpoint_metric_mode,
                'amp_enabled': self.amp_enabled,
                'grad_accum_steps': self.grad_accum_steps,
            }
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.amp_enabled and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            检查点内容
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint.get('history', self.history)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.amp_enabled and self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
    
    def _save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / "history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
