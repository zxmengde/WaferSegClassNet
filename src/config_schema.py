# -*- coding: utf-8 -*-
"""
配置数据类定义

定义实验配置的数据结构，支持 YAML 加载与验证

Requirements: 2.7 - 配置文件YAML格式有效性
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
import yaml
from pathlib import Path


# 必需字段定义
REQUIRED_FIELDS = {
    'experiment': ['name', 'seed'],
    'data': ['dataset', 'data_root', 'image_size', 'batch_size'],
    'model': ['encoder', 'classification_classes', 'segmentation_classes'],
    'training': ['epochs', 'optimizer', 'learning_rate'],
    'loss': ['classification', 'segmentation', 'weights'],
    'output': ['results_dir']
}

# 有效值定义
VALID_VALUES = {
    'classification_mode': ['single_label', 'multi_label'],
    'encoder': ['custom', 'resnet18', 'resnet34'],
    'optimizer': ['adam', 'adamw', 'sgd'],
    'scheduler': ['cosine', 'step', 'plateau', 'none'],
    'classification_loss': ['cross_entropy', 'focal', 'class_balanced'],
    'segmentation_loss': ['bce_dice', 'dice', 'bce'],
    'separation_loss': ['kl_divergence', 'mse']
}


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
    sampler: str = "uniform"  # uniform | inverse | sqrt_inverse | effective_num
    sampler_beta: float = 0.9999  # effective_num 模式的 beta 参数
    tail_class_threshold: int = 100  # 样本数少于此值的类为尾部类
    tail_augmentation_strength: float = 1.0
    
    def __post_init__(self):
        if isinstance(self.augmentation, dict):
            self.augmentation = AugmentationConfig(**self.augmentation)
        if isinstance(self.image_size, list):
            self.image_size = tuple(self.image_size)


@dataclass
class KeyMappingConfig:
    """权重加载 key 映射配置"""
    extract_subtree: Optional[str] = None
    strip_prefix: List[str] = field(default_factory=list)
    use_encoder_state_dict: bool = True  # SSL checkpoint 默认使用 encoder_state_dict


@dataclass
class ModelConfig:
    """模型配置"""
    encoder: str = "custom"  # custom | resnet18 | resnet34
    pretrained_weights: Optional[str] = None
    key_mapping: KeyMappingConfig = field(default_factory=KeyMappingConfig)
    classification_classes: int = 38
    segmentation_classes: int = 1
    separation_enabled: bool = False
    separation_channels: int = 8
    separation_mode: str = "prototype"  # prototype | trained
    separation_temperature: float = 0.1  # cosine similarity 温度参数
    
    def __post_init__(self):
        if isinstance(self.key_mapping, dict):
            self.key_mapping = KeyMappingConfig(**self.key_mapping)


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
    if config.data.classification_mode not in VALID_VALUES['classification_mode']:
        errors.append(f"Invalid classification_mode: {config.data.classification_mode}. "
                      f"Valid values: {VALID_VALUES['classification_mode']}")
    
    # 验证编码器类型
    if config.model.encoder not in VALID_VALUES['encoder']:
        errors.append(f"Invalid encoder: {config.model.encoder}. "
                      f"Valid values: {VALID_VALUES['encoder']}")
    
    # 验证优化器
    if config.training.optimizer not in VALID_VALUES['optimizer']:
        errors.append(f"Invalid optimizer: {config.training.optimizer}. "
                      f"Valid values: {VALID_VALUES['optimizer']}")
    
    # 验证调度器
    if config.training.scheduler not in VALID_VALUES['scheduler']:
        errors.append(f"Invalid scheduler: {config.training.scheduler}. "
                      f"Valid values: {VALID_VALUES['scheduler']}")
    
    # 验证损失函数
    if config.loss.classification not in VALID_VALUES['classification_loss']:
        errors.append(f"Invalid classification loss: {config.loss.classification}. "
                      f"Valid values: {VALID_VALUES['classification_loss']}")
    
    if config.loss.segmentation not in VALID_VALUES['segmentation_loss']:
        errors.append(f"Invalid segmentation loss: {config.loss.segmentation}. "
                      f"Valid values: {VALID_VALUES['segmentation_loss']}")
    
    # 验证数值范围
    if config.training.learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got: {config.training.learning_rate}")
    
    if config.training.epochs <= 0:
        errors.append(f"epochs must be positive, got: {config.training.epochs}")
    
    if config.data.batch_size <= 0:
        errors.append(f"batch_size must be positive, got: {config.data.batch_size}")
    
    if config.model.classification_classes <= 0:
        errors.append(f"classification_classes must be positive, got: {config.model.classification_classes}")
    
    # 验证图像尺寸
    if len(config.data.image_size) != 2:
        errors.append(f"image_size must have 2 elements, got: {len(config.data.image_size)}")
    elif config.data.image_size[0] <= 0 or config.data.image_size[1] <= 0:
        errors.append(f"image_size dimensions must be positive, got: {config.data.image_size}")
    
    # 验证损失权重
    if len(config.loss.weights) < 2:
        errors.append(f"loss weights must have at least 2 elements, got: {len(config.loss.weights)}")
    
    return errors


def validate_yaml_file(yaml_path: str) -> Tuple[bool, List[str]]:
    """
    验证 YAML 配置文件的格式和必需字段
    
    Args:
        yaml_path: YAML 文件路径
        
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    
    # 检查文件是否存在
    if not Path(yaml_path).exists():
        return False, [f"Config file not found: {yaml_path}"]
    
    # 尝试解析 YAML
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"YAML parsing error: {str(e)}"]
    
    if config_dict is None:
        return False, ["Config file is empty"]
    
    if not isinstance(config_dict, dict):
        return False, ["Config file must contain a dictionary at root level"]
    
    # 检查必需的顶级字段
    required_top_level = ['experiment', 'data', 'model', 'training', 'loss', 'output']
    for field in required_top_level:
        if field not in config_dict:
            errors.append(f"Missing required top-level field: {field}")
    
    # 检查各部分的必需字段
    for section, fields in REQUIRED_FIELDS.items():
        if section in config_dict and isinstance(config_dict[section], dict):
            for field in fields:
                if field not in config_dict[section]:
                    errors.append(f"Missing required field in {section}: {field}")
    
    return len(errors) == 0, errors


def load_and_validate_config(config_path: str) -> Tuple[Optional[ExperimentConfig], List[str]]:
    """
    加载并验证配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        (配置实例或None, 错误信息列表)
    """
    # 先验证 YAML 格式
    is_valid, yaml_errors = validate_yaml_file(config_path)
    if not is_valid:
        return None, yaml_errors
    
    # 加载配置
    try:
        config = load_config(config_path)
    except Exception as e:
        return None, [f"Failed to load config: {str(e)}"]
    
    # 验证配置值
    validation_errors = validate_config(config)
    
    return config, validation_errors


def get_effective_config(config: ExperimentConfig) -> ExperimentConfig:
    """
    获取生效的配置（应用 debug 模式覆盖）
    
    Args:
        config: 原始配置
        
    Returns:
        生效的配置（如果是 debug 模式，会覆盖部分参数）
    """
    if not config.debug:
        return config
    
    # Debug 模式下覆盖参数
    import copy
    effective = copy.deepcopy(config)
    effective.training.epochs = config.training.debug_epochs
    effective.data.batch_size = config.training.debug_batch_size
    effective.data.num_workers = 0  # Debug 模式下禁用多进程
    effective.model.separation_enabled = False  # Debug 模式下禁用分离头
    
    return effective
