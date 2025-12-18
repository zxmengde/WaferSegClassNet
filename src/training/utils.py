# -*- coding: utf-8 -*-
"""
训练工具模块

提供训练相关的工具函数：
- set_seed(): 设置随机种子
- get_git_commit(): 获取 git commit hash
- save_config_snapshot(): 保存配置快照
- save_meta_json(): 保存元信息

Requirements: 7.3, 7.4
"""

import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可复现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保 CUDA 操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit() -> str:
    """
    获取当前 git commit hash
    
    Returns:
        commit hash 字符串，如果获取失败返回 "unknown"
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # 返回短 hash
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return "unknown"


def get_git_branch() -> str:
    """
    获取当前 git 分支名
    
    Returns:
        分支名，如果获取失败返回 "unknown"
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return "unknown"


def save_config_snapshot(config: Any, save_path: Union[str, Path]):
    """
    保存配置快照到 YAML 文件
    
    Args:
        config: 配置对象（dataclass 或字典）
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 将 dataclass 转换为字典
    def to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        else:
            return obj
    
    config_dict = to_dict(config)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def save_meta_json(
    output_dir: Union[str, Path],
    seed: int,
    extra_info: Optional[Dict[str, Any]] = None,
):
    """
    保存元信息到 meta.json
    
    Args:
        output_dir: 输出目录
        seed: 随机种子
        extra_info: 额外信息
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    meta = {
        'git_commit': get_git_commit(),
        'git_branch': get_git_branch(),
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        meta['cuda_version'] = torch.version.cuda
        meta['gpu_name'] = torch.cuda.get_device_name(0)
    
    if extra_info:
        meta.update(extra_info)
    
    meta_path = output_dir / "meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_meta_json(output_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    加载 meta.json
    
    Args:
        output_dir: 输出目录
        
    Returns:
        元信息字典
    """
    meta_path = Path(output_dir) / "meta.json"
    if not meta_path.exists():
        return {}
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        optimizer_name: 优化器名称 (adam | adamw | sgd)
        learning_rate: 学习率
        weight_decay: 权重衰减
        
    Returns:
        优化器实例
    """
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 100,
    **kwargs,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_name: 调度器名称 (cosine | step | plateau | none)
        epochs: 训练轮数
        **kwargs: 额外参数
        
    Returns:
        调度器实例或 None
    """
    if scheduler_name.lower() == "none":
        return None
    elif scheduler_name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif scheduler_name.lower() == "step":
        step_size = kwargs.get('step_size', epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_name.lower() == "plateau":
        patience = kwargs.get('patience', 10)
        factor = kwargs.get('factor', 0.1)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=factor
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型参数量
    
    Args:
        model: 模型
        
    Returns:
        参数统计字典
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
    }


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


if __name__ == "__main__":
    # 测试工具函数
    print("Testing training utilities...")
    
    # 测试 set_seed
    set_seed(42)
    print(f"Random seed set to 42")
    
    # 测试 get_git_commit
    commit = get_git_commit()
    print(f"Git commit: {commit}")
    
    # 测试 get_git_branch
    branch = get_git_branch()
    print(f"Git branch: {branch}")
    
    # 测试 save_meta_json
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_meta_json(tmpdir, seed=42, extra_info={'test': True})
        meta = load_meta_json(tmpdir)
        print(f"Meta info: {meta}")
    
    print("\nAll tests passed!")
