#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一评估入口

Usage:
    python eval.py --config configs/e0.yaml --ckpt results/e0/checkpoints/best.pt
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
from data.dataset import get_dataloaders
from data.mappings import CLASS_NAME_MAPPING
from models.multitask import create_model


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
    model = create_model(
        encoder=config.model.encoder,
        classification_classes=config.model.classification_classes,
        segmentation_classes=config.model.segmentation_classes,
        separation_enabled=config.model.separation_enabled,
        separation_channels=config.model.separation_channels,
    )
    
    # 加载权重
    logger.info(f"Loading checkpoint from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location='cpu')
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
