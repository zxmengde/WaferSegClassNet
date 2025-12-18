#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一评估入口

支持:
- --config: 配置文件路径
- --ckpt: 检查点路径

Usage:
    python eval.py --config configs/e0.yaml --ckpt results/e0/checkpoints/best.pt
    python eval.py --config configs/e0.yaml --ckpt results/e0/checkpoints/best.pt --debug

Requirements: 7.1
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch.cuda.amp import autocast
from tqdm import tqdm

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config_schema import load_config
from data.dataset import get_dataloaders, MixedWM38Dataset
from data.mappings import CLASS_NAME_MAPPING
from models.multitask import create_model
from models.separation import PrototypeSeparator, save_separation_maps


def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "eval.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-7) -> float:
    """计算 Dice 系数"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-7) -> float:
    """计算 IoU"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device: torch.device,
    amp_enabled: bool = True,
):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_dice = []
    all_iou = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device)
        
        with autocast(enabled=amp_enabled):
            outputs = model(images)
        
        # 分类预测
        cls_preds = outputs['cls_logits'].argmax(dim=1)
        all_preds.extend(cls_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 分割指标
        seg_preds = (outputs['seg_mask'] > 0.5).float()
        for i in range(seg_preds.size(0)):
            dice = compute_dice(seg_preds[i], masks[i])
            iou = compute_iou(seg_preds[i], masks[i])
            all_dice.append(dice.item())
            all_iou.append(iou.item())
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    accuracy = (all_preds == all_labels).mean()
    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)
    
    # 每类 F1
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'dice': mean_dice,
        'iou': mean_iou,
        'per_class_f1': per_class_f1,
        'predictions': all_preds,
        'labels': all_labels,
    }


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: dict,
    save_path: str,
):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, predictions)
    
    # 获取实际出现的类别
    unique_labels = np.unique(np.concatenate([labels, predictions]))
    
    # 创建类别名称列表
    names = [class_names.get(i, f"Class_{i}") for i in unique_labels]
    
    # 绘制
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=names,
        yticklabels=names,
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_metrics_csv(metrics: dict, save_path: str, class_names: dict):
    """保存指标到 CSV"""
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 总体指标
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Accuracy', f"{metrics['accuracy']:.4f}"])
        writer.writerow(['Macro-F1', f"{metrics['macro_f1']:.4f}"])
        writer.writerow(['Micro-F1', f"{metrics['micro_f1']:.4f}"])
        writer.writerow(['Weighted-F1', f"{metrics['weighted_f1']:.4f}"])
        writer.writerow(['Dice', f"{metrics['dice']:.4f}"])
        writer.writerow(['IoU', f"{metrics['iou']:.4f}"])
        writer.writerow([])
        
        # 每类 F1
        writer.writerow(['Class', 'Name', 'F1'])
        for i, f1 in enumerate(metrics['per_class_f1']):
            name = class_names.get(i, f"Class_{i}")
            writer.writerow([i, name, f"{f1:.4f}"])


def save_tail_class_analysis(
    metrics: dict,
    class_counts: dict,
    class_names: dict,
    save_path: str,
    tail_threshold: int = 10,
    baseline_metrics: dict = None,
):
    """
    保存尾部类别分析到 CSV
    
    Args:
        metrics: 当前实验的评估指标
        class_counts: 每类样本数
        class_names: 类别名称映射
        save_path: 保存路径
        tail_threshold: 尾部类别阈值
        baseline_metrics: 基线实验的指标（用于计算 delta）
    """
    per_class_f1 = metrics['per_class_f1']
    baseline_f1 = baseline_metrics.get('per_class_f1') if baseline_metrics else None
    
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 表头
        header = ['class_id', 'class_name', 'sample_count', 'is_tail', 'f1']
        if baseline_f1 is not None:
            header.extend(['baseline_f1', 'delta', 'delta_pct'])
        writer.writerow(header)
        
        # 按样本数排序（从少到多）
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
        
        tail_f1_sum = 0.0
        tail_count = 0
        baseline_tail_f1_sum = 0.0
        
        for class_id, count in sorted_classes:
            if class_id >= len(per_class_f1):
                continue
            
            name = class_names.get(class_id, f"Class_{class_id}")
            is_tail = count < tail_threshold and count > 0
            f1 = per_class_f1[class_id]
            
            row = [class_id, name, count, 'Yes' if is_tail else 'No', f"{f1:.4f}"]
            
            if baseline_f1 is not None and class_id < len(baseline_f1):
                base_f1 = baseline_f1[class_id]
                delta = f1 - base_f1
                delta_pct = (delta / max(base_f1, 1e-7)) * 100
                row.extend([f"{base_f1:.4f}", f"{delta:+.4f}", f"{delta_pct:+.1f}%"])
                
                if is_tail:
                    baseline_tail_f1_sum += base_f1
            
            if is_tail:
                tail_f1_sum += f1
                tail_count += 1
            
            writer.writerow(row)
        
        # 添加尾部类别汇总
        writer.writerow([])
        writer.writerow(['Summary'])
        writer.writerow(['Tail class count', tail_count])
        if tail_count > 0:
            tail_macro_f1 = tail_f1_sum / tail_count
            writer.writerow(['Tail Macro-F1', f"{tail_macro_f1:.4f}"])
            
            if baseline_f1 is not None:
                baseline_tail_macro_f1 = baseline_tail_f1_sum / tail_count
                tail_delta = tail_macro_f1 - baseline_tail_macro_f1
                tail_delta_pct = (tail_delta / max(baseline_tail_macro_f1, 1e-7)) * 100
                writer.writerow(['Baseline Tail Macro-F1', f"{baseline_tail_macro_f1:.4f}"])
                writer.writerow(['Tail Delta', f"{tail_delta:+.4f}"])
                writer.writerow(['Tail Delta %', f"{tail_delta_pct:+.1f}%"])


def generate_seg_overlays(
    model,
    dataloader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 10,
    amp_enabled: bool = True,
):
    """生成分割 overlay 可视化"""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for batch in dataloader:
        if count >= num_samples:
            break
        
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        with torch.no_grad(), autocast(enabled=amp_enabled):
            outputs = model(images)
        
        seg_preds = (outputs['seg_mask'] > 0.5).float()
        
        for i in range(images.size(0)):
            if count >= num_samples:
                break
            
            # 转换为 numpy
            img = images[i].cpu().permute(1, 2, 0).numpy()
            mask_true = masks[i, 0].cpu().numpy()
            mask_pred = seg_preds[i, 0].cpu().numpy()
            
            # 创建 overlay
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(img)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(mask_true, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            axes[2].imshow(mask_pred, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"sample_{count:03d}.png", dpi=100)
            plt.close()
            
            count += 1


@torch.no_grad()
def generate_separation_maps(
    model,
    train_dataloader,
    test_dataloader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 10,
    temperature: float = 0.1,
    logger=None,
):
    """
    生成 E3 分离热力图
    
    使用 Prototype 相似度方法：
    1. 从训练集构建 8 类基础缺陷的 prototype
    2. 对测试样本计算与每个 prototype 的相似度
    3. 输出 8 通道热力图
    
    Args:
        model: 多任务模型
        train_dataloader: 训练数据加载器（用于构建 prototype）
        test_dataloader: 测试数据加载器
        device: 计算设备
        output_dir: 输出目录
        num_samples: 最大样本数
        temperature: 相似度温度参数
        logger: 日志记录器
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info("Creating PrototypeSeparator...")
    
    # 创建分离器
    separator = PrototypeSeparator(
        encoder=model.encoder,
        device=str(device),
        num_components=8,
    )
    
    # 从训练数据构建 prototype
    if logger:
        logger.info("Building prototypes from training data...")
    
    stats = separator.build_prototypes(train_dataloader)
    
    if logger:
        logger.info(f"Prototype statistics: {stats}")
    
    # 保存 prototype
    proto_path = output_dir.parent / "prototypes.pt"
    separator.save_prototypes(str(proto_path))
    if logger:
        logger.info(f"Prototypes saved to {proto_path}")
    
    # 收集测试样本
    all_images = []
    all_labels = []
    
    for batch in test_dataloader:
        images = batch['image']
        labels = batch['label_38']
        
        all_images.append(images)
        all_labels.append(labels)
        
        if sum(len(x) for x in all_images) >= num_samples:
            break
    
    all_images = torch.cat(all_images)[:num_samples]
    all_labels = torch.cat(all_labels)[:num_samples]
    
    # 计算分离热力图
    if logger:
        logger.info(f"Computing separation maps for {len(all_images)} samples...")
    
    separation_maps = separator.compute_separation_maps(
        all_images.to(device),
        temperature=temperature,
    )
    
    # 保存分离热力图
    saved_files = save_separation_maps(
        separation_maps=separation_maps,
        images=all_images,
        output_dir=str(output_dir),
        max_samples=num_samples,
    )
    
    if logger:
        logger.info(f"Separation maps saved to {output_dir}")
        logger.info(f"Saved {len(saved_files)} files")
    
    return {
        'prototype_stats': stats,
        'num_samples': len(all_images),
        'saved_files': saved_files,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate wafer map model")
    parser.add_argument("--config", "-c", type=str, required=True, help="Config file path")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    if args.debug:
        config.debug = True
    
    # 设置实验名称
    exp_name = config.name
    if config.debug:
        exp_name = f"{exp_name}_debug"
    
    # 输出目录
    output_dir = Path(config.output.results_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(str(output_dir))
    logger.info(f"Evaluating: {exp_name}")
    logger.info(f"Checkpoint: {args.ckpt}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # 确定参数
    if config.debug:
        batch_size = config.training.debug_batch_size
        num_workers = 0
        max_per_class = config.training.debug_max_per_class
    else:
        batch_size = config.data.batch_size
        num_workers = config.data.num_workers
        max_per_class = None
    
    # 创建数据加载器
    logger.info("Creating data loaders...")
    _, _, test_loader = get_dataloaders(
        data_root=config.data.data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        classification_mode=config.data.classification_mode,
        debug=config.debug,
        max_per_class=max_per_class or 5,
    )
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # 创建模型
    logger.info("Creating model...")
    
    # 对于 E3 prototype 模式，不需要分离头（使用 PrototypeSeparator 代替）
    separation_mode = getattr(config.model, 'separation_mode', 'prototype')
    model_separation_enabled = config.model.separation_enabled and separation_mode != 'prototype'
    
    model = create_model(
        encoder=config.model.encoder,
        classification_classes=config.model.classification_classes,
        segmentation_classes=config.model.segmentation_classes,
        separation_enabled=model_separation_enabled,
        separation_channels=config.model.separation_channels,
    )
    
    # 加载权重
    logger.info(f"Loading checkpoint from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 评估
    logger.info("Evaluating...")
    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        amp_enabled=config.training.amp_enabled,
    )
    
    # 打印结果
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro-F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Micro-F1: {metrics['micro_f1']:.4f}")
    logger.info(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    logger.info(f"Dice: {metrics['dice']:.4f}")
    logger.info(f"IoU: {metrics['iou']:.4f}")
    logger.info("=" * 50)
    
    # 保存指标
    save_metrics_csv(
        metrics=metrics,
        save_path=str(output_dir / "metrics.csv"),
        class_names=CLASS_NAME_MAPPING,
    )
    logger.info(f"Metrics saved to {output_dir / 'metrics.csv'}")
    
    # 生成尾部类别分析（如果使用了加权采样或 focal loss）
    sampler_mode = getattr(config.data, 'sampler', 'uniform')
    loss_type = config.loss.classification
    
    if sampler_mode not in ['uniform', 'none', None] or loss_type in ['focal', 'class_balanced']:
        logger.info("Generating tail class analysis...")
        
        # 获取类别统计
        test_dataset = test_loader.dataset
        class_counts = test_dataset.get_class_counts()
        tail_threshold = getattr(config.data, 'tail_class_threshold', 10)
        
        # 尝试加载基线指标（E0）
        baseline_metrics = None
        baseline_path = Path(config.output.results_dir) / "e0" / "metrics.csv"
        if baseline_path.exists():
            try:
                # 解析基线 metrics.csv 获取 per_class_f1
                baseline_per_class_f1 = []
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    in_class_section = False
                    for row in reader:
                        if len(row) >= 3 and row[0] == 'Class':
                            in_class_section = True
                            continue
                        if in_class_section and len(row) >= 3:
                            try:
                                baseline_per_class_f1.append(float(row[2]))
                            except (ValueError, IndexError):
                                pass
                if baseline_per_class_f1:
                    baseline_metrics = {'per_class_f1': np.array(baseline_per_class_f1)}
                    logger.info(f"Loaded baseline metrics from {baseline_path}")
            except Exception as e:
                logger.warning(f"Failed to load baseline metrics: {e}")
        
        # 保存尾部类别分析
        save_tail_class_analysis(
            metrics=metrics,
            class_counts=class_counts,
            class_names=CLASS_NAME_MAPPING,
            save_path=str(output_dir / "tail_class_analysis.csv"),
            tail_threshold=tail_threshold,
            baseline_metrics=baseline_metrics,
        )
        logger.info(f"Tail class analysis saved to {output_dir / 'tail_class_analysis.csv'}")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        labels=metrics['labels'],
        predictions=metrics['predictions'],
        class_names=CLASS_NAME_MAPPING,
        save_path=str(output_dir / "confusion_matrix.png"),
    )
    logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    
    # 生成分割 overlay
    seg_overlay_dir = output_dir / "seg_overlays"
    generate_seg_overlays(
        model=model,
        dataloader=test_loader,
        device=device,
        output_dir=str(seg_overlay_dir),
        num_samples=10,
        amp_enabled=config.training.amp_enabled,
    )
    logger.info(f"Segmentation overlays saved to {seg_overlay_dir}")
    
    # E3: 生成分离热力图（如果启用）
    if config.model.separation_enabled:
        logger.info("\n" + "=" * 50)
        logger.info("E3 Separation Evaluation")
        logger.info("=" * 50)
        
        # 获取分离模式和温度参数
        separation_mode = getattr(config.model, 'separation_mode', 'prototype')
        separation_temperature = getattr(config.model, 'separation_temperature', 0.1)
        
        logger.info(f"Separation mode: {separation_mode}")
        logger.info(f"Temperature: {separation_temperature}")
        
        if separation_mode == "prototype":
            # 需要训练数据加载器来构建 prototype
            logger.info("Creating train dataloader for prototype building...")
            train_loader, _, _ = get_dataloaders(
                data_root=config.data.data_root,
                batch_size=batch_size,
                num_workers=num_workers,
                classification_mode=config.data.classification_mode,
                debug=config.debug,
                max_per_class=max_per_class or 5,
            )
            
            sep_dir = output_dir / "separation_maps"
            sep_results = generate_separation_maps(
                model=model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                device=device,
                output_dir=str(sep_dir),
                num_samples=10,
                temperature=separation_temperature,
                logger=logger,
            )
            
            logger.info(f"Separation evaluation complete")
            logger.info(f"Prototype stats: {sep_results['prototype_stats']}")
        else:
            logger.warning(f"Separation mode '{separation_mode}' not implemented, skipping")
    
    # 生成训练曲线（如果有历史记录）
    history_path = output_dir / "history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        curves_dir = output_dir / "curves"
        curves_dir.mkdir(exist_ok=True)
        
        # Loss 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(curves_dir / "loss_curve.png", dpi=100)
        plt.close()
        
        # Metric 曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history['val_macro_f1'], label='Macro-F1')
        plt.plot(history['val_dice'], label='Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(curves_dir / "metric_curve.png", dpi=100)
        plt.close()
        
        logger.info(f"Training curves saved to {curves_dir}")
    
    logger.info("\nEvaluation completed!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
