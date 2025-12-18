#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SSL 自监督预训练入口

支持:
- --config: 配置文件路径
- --resume: 断点续训检查点路径

Usage:
    python train_ssl.py --config configs/ssl.yaml
    python train_ssl.py --config configs/ssl_debug.yaml
    python train_ssl.py --config configs/ssl.yaml --resume results/ssl/checkpoints/last.pt

Requirements: 4.1 - SSL 预训练
"""

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training.ssl_trainer import (
    SimCLRTrainer,
    create_ssl_model,
    get_ssl_dataloader,
)
from training.utils import set_seed, create_optimizer, create_scheduler


def setup_logging(log_dir: str, name: str = "train_ssl"):
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


def load_ssl_config(config_path: str) -> dict:
    """加载 SSL 配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="SSL pretraining for wafer map")
    parser.add_argument("--config", "-c", type=str, required=True, help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # 加载配置
    config = load_ssl_config(args.config)
    
    # 获取配置参数
    exp_config = config.get('experiment', {})
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    output_config = config.get('output', {})
    
    # 实验名称
    exp_name = exp_config.get('name', 'ssl')
    debug = exp_config.get('debug', False)
    seed = exp_config.get('seed', 42)
    
    # 创建输出目录
    results_dir = output_config.get('results_dir', 'results')
    output_dir = Path(results_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(str(output_dir), "train_ssl")
    logger.info(f"SSL Experiment: {exp_name}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Debug mode: {debug}")
    
    # 设置随机种子
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 数据参数
    data_root = data_config.get('data_root', 'data/processed')
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    image_size = tuple(data_config.get('image_size', [224, 224]))
    max_samples = data_config.get('max_samples', 500)
    
    # Debug 模式调整
    if debug:
        batch_size = data_config.get('debug_batch_size', 8)
        num_workers = 0
        max_samples = data_config.get('debug_max_samples', 100)
    
    logger.info(f"Data root: {data_root}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Image size: {image_size}")
    
    # 检查数据源
    # 优先使用 WM-811K，如果不可用则使用 MixedWM38
    wm811k_path = data_config.get('wm811k_path', None)
    if wm811k_path and Path(wm811k_path).exists():
        logger.info(f"Using WM-811K dataset from {wm811k_path}")
        ssl_data_root = wm811k_path
    else:
        logger.info(f"WM-811K not available, using MixedWM38 as fallback")
        ssl_data_root = data_root
    
    # 创建数据加载器
    logger.info("Creating SSL data loader...")
    train_loader = get_ssl_dataloader(
        data_root=ssl_data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        debug=debug,
        max_samples=max_samples,
        image_size=image_size,
    )
    logger.info(f"SSL samples: {len(train_loader.dataset)}")
    
    if len(train_loader.dataset) == 0:
        logger.error("No samples found for SSL training!")
        return 1
    
    # 模型参数
    encoder_type = model_config.get('encoder', 'custom')
    in_channels = model_config.get('in_channels', 3)
    base_channels = model_config.get('base_channels', 8)
    projection_dim = model_config.get('projection_dim', 128)
    hidden_dim = model_config.get('hidden_dim', 256)
    
    # 创建模型
    logger.info("Creating SimCLR model...")
    model = create_ssl_model(
        encoder_type=encoder_type,
        in_channels=in_channels,
        base_channels=base_channels,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 训练参数
    epochs = training_config.get('epochs', 100)
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.01)
    optimizer_name = training_config.get('optimizer', 'adamw')
    scheduler_name = training_config.get('scheduler', 'cosine')
    amp_enabled = training_config.get('amp_enabled', True)
    temperature = training_config.get('temperature', 0.5)
    save_every = output_config.get('save_every', 10)
    
    # Debug 模式调整
    if debug:
        epochs = training_config.get('debug_epochs', 2)
        save_every = 1
    
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"AMP enabled: {amp_enabled}")
    
    # 创建优化器
    optimizer = create_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    
    # 创建调度器
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        epochs=epochs,
    )
    
    # 创建训练器
    trainer = SimCLRTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        amp_enabled=amp_enabled,
        temperature=temperature,
        output_dir=str(output_dir),
        seed=seed,
        config=config,
    )
    
    # 开始训练
    logger.info("Starting SSL training...")
    history = trainer.train(
        train_loader=train_loader,
        epochs=epochs,
        resume_from=args.resume,
        save_every=save_every,
    )
    
    logger.info("\nSSL training completed!")
    logger.info(f"Final loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Checkpoint saved to: {output_dir / 'checkpoints' / 'last.pt'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
