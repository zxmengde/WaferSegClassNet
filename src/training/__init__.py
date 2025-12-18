# -*- coding: utf-8 -*-
"""
Training module

提供训练相关的功能：
- Trainer: 统一训练器
- 训练工具函数
"""

from .trainer import Trainer
from .utils import (
    set_seed,
    get_git_commit,
    get_git_branch,
    save_config_snapshot,
    save_meta_json,
    load_meta_json,
    create_optimizer,
    create_scheduler,
    count_parameters,
    format_time,
)

__all__ = [
    'Trainer',
    'set_seed',
    'get_git_commit',
    'get_git_branch',
    'save_config_snapshot',
    'save_meta_json',
    'load_meta_json',
    'create_optimizer',
    'create_scheduler',
    'count_parameters',
    'format_time',
]
