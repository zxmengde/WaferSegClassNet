# -*- coding: utf-8 -*-
"""
SimCLR 自监督预训练训练器

实现对比学习（InfoNCE损失）用于晶圆图谱的自监督表征学习。
支持 MixedWM38 train 图像作为无标签数据源（保守 fallback）。

Requirements: 4.1 - SSL 预训练
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

from .utils import get_git_commit, save_config_snapshot, save_meta_json

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比学习损失（NT-Xent）
    
    用于 SimCLR 风格的自监督学习。
    对于每个样本，正样本对是同一图像的两个增强视图，
    负样本是 batch 中其他所有样本的增强视图。
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Args:
            temperature: 温度参数，控制分布的锐度
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        计算 InfoNCE 损失
        
        Args:
            z_i: 第一个视图的特征 (B, D)
            z_j: 第二个视图的特征 (B, D)
            
        Returns:
            损失值
        """
        batch_size = z_i.size(0)
        device = z_i.device
        
        # L2 归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 拼接特征 (2B, D)
        z = torch.cat([z_i, z_j], dim=0)
        
        # 计算相似度矩阵 (2B, 2B)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # 创建正样本对的 mask
        # 对于 z_i[k]，正样本是 z_j[k]（位置 batch_size + k）
        # 对于 z_j[k]，正样本是 z_i[k]（位置 k）
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(device)
        
        # 移除对角线（自身相似度）
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class ProjectionHead(nn.Module):
    """
    SimCLR 投影头
    
    将编码器输出映射到对比学习空间。
    使用 MLP 结构：Linear -> ReLU -> Linear
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 256,
        out_features: int = 128,
    ):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 编码器输出 (B, C, H, W)
            
        Returns:
            投影特征 (B, out_features)
        """
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimCLRModel(nn.Module):
    """
    SimCLR 模型
    
    包含编码器和投影头。
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        
        # 获取编码器输出通道数
        encoder_out_channels = getattr(encoder, 'out_channels', 64)
        
        self.projection_head = ProjectionHead(
            in_features=encoder_out_channels,
            hidden_features=hidden_dim,
            out_features=projection_dim,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            投影特征 (B, projection_dim)
        """
        # 编码器前向传播
        features, _ = self.encoder(x)
        
        # 投影头
        z = self.projection_head(features)
        
        return z
    
    def get_encoder(self) -> nn.Module:
        """获取编码器（用于下游任务）"""
        return self.encoder


class SimCLRTrainer:
    """
    SimCLR 自监督预训练训练器
    
    功能:
    - InfoNCE 对比学习损失
    - AMP 混合精度训练
    - 断点续训
    - 训练曲线记录
    """
    
    def __init__(
        self,
        model: SimCLRModel,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        amp_enabled: bool = True,
        temperature: float = 0.5,
        output_dir: str = "results/ssl",
        seed: int = 42,
        config: Optional[Any] = None,
    ):
        """
        Args:
            model: SimCLR 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            amp_enabled: 是否启用 AMP
            temperature: InfoNCE 温度参数
            output_dir: 输出目录
            seed: 随机种子
            config: 配置对象
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp_enabled = amp_enabled and torch.cuda.is_available()
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.config = config
        
        # 损失函数
        self.criterion = InfoNCELoss(temperature=temperature)
        
        # AMP scaler
        self.scaler = GradScaler('cuda') if self.amp_enabled else None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'learning_rate': [],
        }
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "curves").mkdir(exist_ok=True)
        
        # 保存初始元信息
        self._save_initial_meta()
    
    def _save_initial_meta(self):
        """保存初始元信息"""
        if self.config is not None:
            save_config_snapshot(self.config, self.output_dir / "config_snapshot.yaml")
        
        save_meta_json(
            output_dir=self.output_dir,
            seed=self.seed,
            extra_info={
                'amp_enabled': self.amp_enabled,
                'training_type': 'ssl_simclr',
            }
        )
    
    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        resume_from: Optional[str] = None,
        save_every: int = 10,
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器（需要返回两个增强视图）
            epochs: 训练轮数
            resume_from: 断点续训检查点路径
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
        
        logger.info(f"Starting SSL training from epoch {start_epoch} to {total_epochs}")
        logger.info(f"AMP enabled: {self.amp_enabled}")
        
        for epoch in range(start_epoch, total_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个 epoch
            train_loss = self._train_epoch(train_loader)
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['learning_rate'].append(current_lr)
            
            # 保存检查点
            if (epoch + 1) % save_every == 0 or epoch == total_epochs - 1:
                self.save_checkpoint(
                    self.output_dir / "checkpoints" / "last.pt"
                )
            
            # 日志
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch [{epoch+1}/{total_epochs}] "
                f"Loss: {train_loss:.4f} "
                f"LR: {current_lr:.6f} "
                f"Time: {epoch_time:.1f}s"
            )
        
        # 保存训练历史
        self._save_history()
        
        # 生成损失曲线
        self._plot_loss_curve()
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # 获取两个增强视图
            view1 = batch['view1'].to(self.device)
            view2 = batch['view2'].to(self.device)
            
            # 前向传播
            with autocast('cuda', enabled=self.amp_enabled):
                z1 = self.model(view1)
                z2 = self.model(view2)
                loss = self.criterion(z1, z2)
            
            # 反向传播
            self.optimizer.zero_grad()
            if self.amp_enabled:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'encoder_state_dict': self.model.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.amp_enabled and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"SSL checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.history = checkpoint.get('history', self.history)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.amp_enabled and self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"SSL checkpoint loaded from {path}")
        return checkpoint
    
    def _save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / "history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"SSL training history saved to {history_path}")
    
    def _plot_loss_curve(self):
        """绘制损失曲线"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(self.history['train_loss']) + 1)
            ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('SSL Training Loss Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            save_path = self.output_dir / "curves" / "loss_curve.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Loss curve saved to {save_path}")
        except ImportError:
            logger.warning("matplotlib not available, skipping loss curve plot")
        except Exception as e:
            logger.warning(f"Failed to plot loss curve: {e}")


class SSLAugmentation:
    """
    SSL 数据增强
    
    为 SimCLR 生成两个不同的增强视图。
    使用晶圆友好的增强策略。
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        rotation_angles: List[int] = [0, 90, 180, 270],
        flip: bool = True,
        color_jitter: float = 0.2,
        gaussian_blur: bool = True,
    ):
        self.image_size = image_size
        self.rotation_angles = rotation_angles
        self.flip = flip
        self.color_jitter = color_jitter
        self.gaussian_blur = gaussian_blur
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成两个增强视图
        
        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            
        Returns:
            (view1, view2): 两个增强视图
        """
        view1 = self._augment(image)
        view2 = self._augment(image)
        return view1, view2
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """应用随机增强"""
        img = image.copy()
        
        # 确保是 3 通道
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        
        # 随机旋转
        if self.rotation_angles:
            angle = np.random.choice(self.rotation_angles)
            if angle == 90:
                img = np.rot90(img, k=1)
            elif angle == 180:
                img = np.rot90(img, k=2)
            elif angle == 270:
                img = np.rot90(img, k=3)
        
        # 随机翻转
        if self.flip:
            if np.random.random() > 0.5:
                img = np.fliplr(img)
            if np.random.random() > 0.5:
                img = np.flipud(img)
        
        # 颜色抖动（简化版）
        if self.color_jitter > 0:
            # 亮度调整
            brightness = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
            img = np.clip(img * brightness, 0, 1)
            
            # 对比度调整
            contrast = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
            mean = img.mean()
            img = np.clip((img - mean) * contrast + mean, 0, 1)
        
        # 确保数组连续
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        
        return img.astype(np.float32)


class SSLDataset(Dataset):
    """
    SSL 数据集
    
    为 SimCLR 训练提供两个增强视图。
    支持 MixedWM38 作为数据源。
    """
    
    def __init__(
        self,
        data_root: str,
        augmentation: Optional[SSLAugmentation] = None,
        debug: bool = False,
        max_samples: int = 500,
    ):
        """
        Args:
            data_root: 数据根目录
            augmentation: 数据增强
            debug: 是否为 debug 模式
            max_samples: debug 模式下最大样本数
        """
        self.data_root = Path(data_root)
        self.augmentation = augmentation or SSLAugmentation()
        self.debug = debug
        self.max_samples = max_samples
        
        # 加载图像路径
        self.image_paths = self._load_image_paths()
        
        logger.info(f"SSL dataset loaded with {len(self.image_paths)} samples")
    
    def _load_image_paths(self) -> List[str]:
        """加载图像路径列表"""
        images_dir = self.data_root / "Images"
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return []
        
        image_files = sorted([
            str(f) for f in images_dir.iterdir()
            if f.suffix == '.npy'
        ])
        
        # Debug 模式限制样本数
        if self.debug and len(image_files) > self.max_samples:
            image_files = image_files[:self.max_samples]
        
        return image_files
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取样本
        
        Returns:
            {
                'view1': Tensor (3, H, W),
                'view2': Tensor (3, H, W),
            }
        """
        # 加载图像
        image = np.load(self.image_paths[idx]).astype(np.float32) / 255.0
        
        # 生成两个增强视图
        view1, view2 = self.augmentation(image)
        
        # 转换为 PyTorch 张量 (H, W, C) -> (C, H, W)
        view1 = torch.from_numpy(view1).permute(2, 0, 1).float()
        view2 = torch.from_numpy(view2).permute(2, 0, 1).float()
        
        return {
            'view1': view1,
            'view2': view2,
        }


def create_ssl_model(
    encoder_type: str = "custom",
    in_channels: int = 3,
    base_channels: int = 8,
    projection_dim: int = 128,
    hidden_dim: int = 256,
) -> SimCLRModel:
    """
    创建 SimCLR 模型
    
    Args:
        encoder_type: 编码器类型
        in_channels: 输入通道数
        base_channels: 基础通道数
        projection_dim: 投影维度
        hidden_dim: 隐藏层维度
        
    Returns:
        SimCLRModel 实例
    """
    from src.models.encoder import WaferEncoder
    
    encoder = WaferEncoder(in_channels=in_channels, base_channels=base_channels)
    
    model = SimCLRModel(
        encoder=encoder,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
    )
    
    return model


def get_ssl_dataloader(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    debug: bool = False,
    max_samples: int = 500,
    image_size: Tuple[int, int] = (224, 224),
) -> DataLoader:
    """
    创建 SSL 数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        debug: 是否为 debug 模式
        max_samples: debug 模式下最大样本数
        image_size: 图像尺寸
        
    Returns:
        DataLoader 实例
    """
    augmentation = SSLAugmentation(image_size=image_size)
    
    dataset = SSLDataset(
        data_root=data_root,
        augmentation=augmentation,
        debug=debug,
        max_samples=max_samples,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试 SSL 训练器
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    model = create_ssl_model()
    print(f"Model created: {type(model)}")
    
    # 测试前向传播
    x = torch.randn(4, 3, 224, 224)
    z = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {z.shape}")
    
    # 测试 InfoNCE 损失
    criterion = InfoNCELoss(temperature=0.5)
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)
    loss = criterion(z1, z2)
    print(f"InfoNCE loss: {loss.item():.4f}")
