# -*- coding: utf-8 -*-
"""
配置数据类定义

定义实验配置的数据结构，支持 YAML 加载与验证
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import yaml
from pathlib import Path


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    wafer_friendly: bool = True
    rotation: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
    flip: bool = True
    morphological_noise: float = 0.1
    color_jitter: float = 0.0


@dataclass
class DataConfig:
    """数据配置"""
    dataset: str = "MixedWM38"
    data_root: str = "data/processed"
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    classification_mode: str = "single_label"  # single_label | multi_label
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    def __post_init__(self):
        if isinstance(self.augmentation, dict):
            self.augmentation = AugmentationConfig(**self.augmentation)
        if isinstance(self.image_size, list):
            self.image_size = tuple(self.image_size)


@dataclass
class ModelConfig:
    """模型配置"""
    encoder: str = "custom"  # custom | resnet18 | resnet34
    pretrained_weights: Optional[str] = None
    classification_classes: int = 38
    segmentation_classes: int = 1
    separation_enabled: bool = False
    separation_channels: int = 8


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    optimizer: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    amp_enabled: bool = True
    checkpoint_metric: str = "macro_f1"
    grad_accum_steps: int = 1
    
    # Debug 模式参数
    debug_epochs: int = 2
    debug_batch_size: int = 8
    debug_max_per_class: int = 5


@dataclass
class LossConfig:
    """损失函数配置"""
    classification: str = "cross_entropy"  # cross_entropy | focal | class_balanced
    segmentation: str = "bce_dice"
    separation: str = "kl_divergence"
    weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.5])
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    class_balanced_beta: float = 0.9999


@dataclass
class OutputConfig:
    """输出配置"""
    results_dir: str = "results"
    save_every: int = 10


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str = "experiment"
    seed: int = 42
    debug: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.loss, dict):
            self.loss = LossConfig(**self.loss)
        if isinstance(self.output, dict):
            self.output = OutputConfig(**self.output)


def load_config(config_path: str) -> ExperimentConfig:
    """
    从 YAML 文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ExperimentConfig 实例
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 处理嵌套的 experiment 字段
    if 'experiment' in config_dict:
        exp_dict = config_dict.pop('experiment')
        config_dict.update(exp_dict)
    
    return ExperimentConfig(**config_dict)


def save_config(config: ExperimentConfig, save_path: str):
    """
    保存配置到 YAML 文件
    
    Args:
        config: 配置实例
        save_path: 保存路径
    """
    def dataclass_to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [dataclass_to_dict(item) for item in obj]
        else:
            return obj
    
    config_dict = dataclass_to_dict(config)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def validate_config(config: ExperimentConfig) -> List[str]:
    """
    验证配置有效性
    
    Args:
        config: 配置实例
        
    Returns:
        错误信息列表（空列表表示验证通过）
    """
    errors = []
    
    # 验证分类模式
    if config.data.classification_mode not in ['single_label', 'multi_label']:
        errors.append(f"Invalid classification_mode: {config.data.classification_mode}")
    
    # 验证编码器类型
    if config.model.encoder not in ['custom', 'resnet18', 'resnet34']:
        errors.append(f"Invalid encoder: {config.model.encoder}")
    
    # 验证优化器
    if config.training.optimizer not in ['adam', 'adamw', 'sgd']:
        errors.append(f"Invalid optimizer: {config.training.optimizer}")
    
    # 验证损失函数
    if config.loss.classification not in ['cross_entropy', 'focal', 'class_balanced']:
        errors.append(f"Invalid classification loss: {config.loss.classification}")
    
    return errors
