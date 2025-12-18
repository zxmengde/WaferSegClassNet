#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一训练入口

Usage:
    python train.py --config configs/e0.yaml
    python train.py --config configs/e0.yaml --debug
    python train.py --config configs/e0.yaml --resume results/e0/checkpoints/last.pt
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config_schema import load_config, save_config, ExperimentConfig
from data.dataset import get_dataloaders
from models.multitask import create_model
from models.losses import MultiTaskLoss


def setup_logging(log_dir: str, name: str = "train"):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:
    """获取 git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def create_optimizer(model: nn.Module, config: ExperimentConfig):
    """创建优化器"""
    if config.training.optimizer == "adam":
        return Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")


def create_scheduler(optimizer, config: ExperimentConfig, num_epochs: int):
    """创建学习率调度器"""
    if config.training.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif config.training.scheduler == "step":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        return None


def compute_metrics(outputs, targets, num_classes: int = 38):
    """计算评估指标"""
    # 分类准确率
    cls_preds = outputs['cls_logits'].argmax(dim=1)
    cls_correct = (cls_preds == targets['label']).float().mean().item()
    
    # Dice 系数
    seg_preds = (outputs['seg_mask'] > 0.5).float()
    seg_targets = targets['mask']
    
    intersection = (seg_preds * seg_targets).sum()
    dice = (2. * intersection + 1e-7) / (seg_preds.sum() + seg_targets.sum() + 1e-7)
    
    return {
        'accuracy': cls_correct,
        'dice': dice.item(),
    }


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: MultiTaskLoss,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_accum_steps: int = 1,
):
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_seg_loss = 0.0
    total_accuracy = 0.0
    total_dice = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # 移动数据到设备
        images = batch['image'].to(device)
        targets = {
            'label': batch['label'].to(device),
            'mask': batch['mask'].to(device),
            'label_8': batch['label_8'].to(device),
        }
        
        # 前向传播
        with autocast(enabled=amp_enabled):
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = losses['total'] / grad_accum_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 计算指标
        with torch.no_grad():
            metrics = compute_metrics(outputs, targets)
        
        # 累积统计
        total_loss += losses['total'].item()
        total_cls_loss += losses['cls'].item()
        total_seg_loss += losses['seg'].item()
        total_accuracy += metrics['accuracy']
        total_dice += metrics['dice']
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{total_loss / num_batches:.4f}",
            'acc': f"{total_accuracy / num_batches:.4f}",
            'dice': f"{total_dice / num_batches:.4f}",
        })
    
    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'seg_loss': total_seg_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'dice': total_dice / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: MultiTaskLoss,
    device: torch.device,
    amp_enabled: bool,
):
    """验证"""
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_dice = 0.0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Validating")
    for batch in pbar:
        images = batch['image'].to(device)
        targets = {
            'label': batch['label'].to(device),
            'mask': batch['mask'].to(device),
            'label_8': batch['label_8'].to(device),
        }
        
        with autocast(enabled=amp_enabled):
            outputs = model(images)
            losses = criterion(outputs, targets)
        
        metrics = compute_metrics(outputs, targets)
        
        total_loss += losses['total'].item()
        total_accuracy += metrics['accuracy']
        total_dice += metrics['dice']
        num_batches += 1
        
        # 收集预测结果（用于计算 Macro-F1）
        all_preds.extend(outputs['cls_logits'].argmax(dim=1).cpu().numpy())
        all_labels.extend(targets['label'].cpu().numpy())
        
        pbar.set_postfix({
            'loss': f"{total_loss / num_batches:.4f}",
            'acc': f"{total_accuracy / num_batches:.4f}",
            'dice': f"{total_dice / num_batches:.4f}",
        })
    
    # 计算 Macro-F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'dice': total_dice / num_batches,
        'macro_f1': macro_f1,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    metrics: dict,
    save_path: str,
):
    """保存检查点"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def main():
    parser = argparse.ArgumentParser(description="Train wafer map model")
    parser.add_argument("--config", "-c", type=str, required=True, help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # Debug 模式覆盖
    if args.debug:
        config.debug = True
    
    # 设置实验名称
    exp_name = config.name
    if config.debug:
        exp_name = f"{exp_name}_debug"
    
    # 创建输出目录
    output_dir = Path(config.output.results_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(str(output_dir), "train")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Debug mode: {config.debug}")
    
    # 设置随机种子
    set_seed(config.seed)
    logger.info(f"Random seed: {config.seed}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 确定训练参数
    if config.debug:
        epochs = config.training.debug_epochs
        batch_size = config.training.debug_batch_size
        max_per_class = config.training.debug_max_per_class
        num_workers = 0
    else:
        epochs = config.training.epochs
        batch_size = config.data.batch_size
        max_per_class = None
        num_workers = config.data.num_workers
    
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    # 创建数据加载器
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=config.data.data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        classification_mode=config.data.classification_mode,
        debug=config.debug,
        max_per_class=max_per_class or 5,
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # 创建模型
    logger.info("Creating model...")
    model = create_model(
        encoder=config.model.encoder,
        classification_classes=config.model.classification_classes,
        segmentation_classes=config.model.segmentation_classes,
        separation_enabled=config.model.separation_enabled,
        separation_channels=config.model.separation_channels,
        pretrained_weights=config.model.pretrained_weights,
        output_dir=str(output_dir),
    )
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 创建损失函数
    criterion = MultiTaskLoss(
        classification_loss=config.loss.classification,
        segmentation_loss=config.loss.segmentation,
        separation_loss=config.loss.separation,
        weights=tuple(config.loss.weights),
        focal_gamma=config.loss.focal_gamma,
    )
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, epochs)
    
    # 创建 AMP scaler
    scaler = GradScaler(enabled=config.training.amp_enabled)
    
    # 恢复训练
    start_epoch = 0
    best_metric = 0.0
    
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        start_epoch, metrics = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        best_metric = metrics.get('macro_f1', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best metric: {best_metric:.4f}")
    
    # 保存配置快照
    save_config(config, str(output_dir / "config_snapshot.yaml"))
    
    # 保存元信息
    meta = {
        'git_commit': get_git_commit(),
        'seed': config.seed,
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
    }
    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_dice': [],
        'val_dice': [],
        'val_macro_f1': [],
    }
    
    # 训练循环
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        # 训练
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=config.training.amp_enabled,
            grad_accum_steps=config.training.grad_accum_steps,
        )
        
        # 验证
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=config.training.amp_enabled,
        )
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        
        # 日志
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.4f}, "
            f"Dice: {train_metrics['dice']:.4f}"
        )
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}, "
            f"Dice: {val_metrics['dice']:.4f}, "
            f"Macro-F1: {val_metrics['macro_f1']:.4f}"
        )
        
        # 保存检查点
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 保存最新检查点
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch + 1, val_metrics,
            str(checkpoint_dir / "last.pt")
        )
        
        # 保存最佳检查点
        current_metric = val_metrics.get(config.training.checkpoint_metric, val_metrics['macro_f1'])
        if current_metric > best_metric:
            best_metric = current_metric
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1, val_metrics,
                str(checkpoint_dir / "best.pt")
            )
            logger.info(f"New best model saved! {config.training.checkpoint_metric}: {best_metric:.4f}")
    
    logger.info("\nTraining completed!")
    logger.info(f"Best {config.training.checkpoint_metric}: {best_metric:.4f}")
    
    # 保存训练历史
    with open(output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Debug 模式下自动运行评估
    if config.debug:
        logger.info("\nRunning evaluation in debug mode...")
        os.system(f"python eval.py --config {args.config} --ckpt {checkpoint_dir / 'best.pt'} --debug")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
