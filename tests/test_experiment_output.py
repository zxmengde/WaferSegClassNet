# -*- coding: utf-8 -*-
"""
实验输出结构完整性属性测试

Property 8: 实验输出结构完整性
Validates: Requirements 7.2, 7.3, 7.4
"""

import pytest
import json
import yaml
import tempfile
import shutil
import os
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# 策略定义
# ============================================================================

# 实验名称策略
experiment_name = st.text(
    alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='_-'),
    min_size=1,
    max_size=30
).filter(lambda x: len(x.strip()) > 0 and x[0].isalpha())

# 随机种子策略
seed_strategy = st.integers(min_value=0, max_value=2**31-1)

# Git commit hash 策略（8字符十六进制或 "unknown"）
git_commit_strategy = st.one_of(
    st.just("unknown"),
    st.text(alphabet='0123456789abcdef', min_size=8, max_size=8)
)


# ============================================================================
# 辅助函数
# ============================================================================

def create_mock_experiment_output(
    output_dir: Path,
    exp_name: str,
    seed: int,
    git_commit: str,
    include_curves: bool = True,
    include_metrics: bool = True,
    include_confusion_matrix: bool = True,
    include_config_snapshot: bool = True,
    include_meta: bool = True,
) -> Path:
    """
    创建模拟的实验输出目录结构
    
    Args:
        output_dir: 输出根目录
        exp_name: 实验名称
        seed: 随机种子
        git_commit: Git commit hash
        include_*: 是否包含各个组件
        
    Returns:
        实验目录路径
    """
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 metrics.csv
    if include_metrics:
        metrics_path = exp_dir / "metrics.csv"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("Metric,Value\n")
            f.write("Accuracy,0.8500\n")
            f.write("Macro-F1,0.7800\n")
            f.write("Dice,0.8200\n")
            f.write("IoU,0.7000\n")
    
    # 创建 confusion_matrix.png（空文件模拟）
    if include_confusion_matrix:
        cm_path = exp_dir / "confusion_matrix.png"
        cm_path.touch()
    
    # 创建 config_snapshot.yaml
    if include_config_snapshot:
        config_snapshot = {
            'name': exp_name,
            'seed': seed,
            'debug': False,
            'data': {
                'dataset': 'MixedWM38',
                'data_root': 'data/processed',
                'batch_size': 32,
            },
            'model': {
                'encoder': 'custom',
                'classification_classes': 38,
            },
            'training': {
                'epochs': 100,
                'learning_rate': 0.001,
            },
        }
        config_path = exp_dir / "config_snapshot.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_snapshot, f, default_flow_style=False)
    
    # 创建 meta.json
    if include_meta:
        meta = {
            'git_commit': git_commit,
            'seed': seed,
            'timestamp': '2024-01-01T00:00:00',
        }
        meta_path = exp_dir / "meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
    
    # 创建 curves/ 目录
    if include_curves:
        curves_dir = exp_dir / "curves"
        curves_dir.mkdir(exist_ok=True)
        (curves_dir / "loss_curve.png").touch()
        (curves_dir / "metric_curve.png").touch()
    
    # 创建 checkpoints/ 目录
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    return exp_dir


def validate_experiment_output(exp_dir: Path) -> tuple:
    """
    验证实验输出目录结构完整性
    
    Args:
        exp_dir: 实验目录路径
        
    Returns:
        (is_valid, errors) 元组
    """
    errors = []
    
    # 必需文件检查
    required_files = [
        "metrics.csv",
        "confusion_matrix.png",
        "config_snapshot.yaml",
        "meta.json",
    ]
    
    for filename in required_files:
        filepath = exp_dir / filename
        if not filepath.exists():
            errors.append(f"Missing required file: {filename}")
    
    # curves/ 目录检查
    curves_dir = exp_dir / "curves"
    if not curves_dir.exists():
        errors.append("Missing curves/ directory")
    else:
        if not (curves_dir / "loss_curve.png").exists():
            errors.append("Missing curves/loss_curve.png")
        if not (curves_dir / "metric_curve.png").exists():
            errors.append("Missing curves/metric_curve.png")
    
    # config_snapshot.yaml 内容验证
    config_path = exp_dir / "config_snapshot.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if config is None:
                errors.append("config_snapshot.yaml is empty")
            elif not isinstance(config, dict):
                errors.append("config_snapshot.yaml must contain a dictionary")
        except yaml.YAMLError as e:
            errors.append(f"config_snapshot.yaml is not valid YAML: {e}")
    
    # meta.json 内容验证
    meta_path = exp_dir / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            # 检查必需字段
            if 'git_commit' not in meta:
                errors.append("meta.json missing 'git_commit' field")
            if 'seed' not in meta:
                errors.append("meta.json missing 'seed' field")
                
        except json.JSONDecodeError as e:
            errors.append(f"meta.json is not valid JSON: {e}")
    
    return len(errors) == 0, errors


# ============================================================================
# Property 8: 实验输出结构完整性
# **Feature: mixed-wm38-recognition, Property 8: 实验输出结构完整性**
# **Validates: Requirements 7.2, 7.3, 7.4**
# ============================================================================

@pytest.mark.property
class TestExperimentOutputStructure:
    """Property 8: 实验输出结构完整性测试"""
    
    @given(
        exp_name=experiment_name,
        seed=seed_strategy,
        git_commit=git_commit_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_complete_output_structure_is_valid(self, exp_name, seed, git_commit):
        """
        **Feature: mixed-wm38-recognition, Property 8: 实验输出结构完整性**
        **Validates: Requirements 7.2, 7.3, 7.4**
        
        对于任意完成的实验，results/<exp_name>/目录应包含：
        - metrics.csv
        - confusion_matrix.png
        - config_snapshot.yaml
        - meta.json
        - curves/目录（包含loss_curve.png, metric_curve.png）
        
        且config_snapshot.yaml内容应与输入配置一致，
        meta.json应包含git_commit和seed字段。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # 创建完整的实验输出
            exp_dir = create_mock_experiment_output(
                output_dir=output_dir,
                exp_name=exp_name,
                seed=seed,
                git_commit=git_commit,
            )
            
            # 验证输出结构
            is_valid, errors = validate_experiment_output(exp_dir)
            
            assert is_valid, f"实验输出结构验证失败: {errors}"
    
    @given(
        exp_name=experiment_name,
        seed=seed_strategy,
        git_commit=git_commit_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_config_snapshot_matches_input(self, exp_name, seed, git_commit):
        """
        **Feature: mixed-wm38-recognition, Property 8: 实验输出结构完整性**
        **Validates: Requirements 7.4**
        
        config_snapshot.yaml内容应与输入配置一致。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            exp_dir = create_mock_experiment_output(
                output_dir=output_dir,
                exp_name=exp_name,
                seed=seed,
                git_commit=git_commit,
            )
            
            # 读取 config_snapshot
            config_path = exp_dir / "config_snapshot.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证关键字段
            assert config['name'] == exp_name, f"实验名称不匹配: {config['name']} != {exp_name}"
            assert config['seed'] == seed, f"随机种子不匹配: {config['seed']} != {seed}"
    
    @given(
        exp_name=experiment_name,
        seed=seed_strategy,
        git_commit=git_commit_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_meta_json_contains_required_fields(self, exp_name, seed, git_commit):
        """
        **Feature: mixed-wm38-recognition, Property 8: 实验输出结构完整性**
        **Validates: Requirements 7.3**
        
        meta.json应包含git_commit和seed字段。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            exp_dir = create_mock_experiment_output(
                output_dir=output_dir,
                exp_name=exp_name,
                seed=seed,
                git_commit=git_commit,
            )
            
            # 读取 meta.json
            meta_path = exp_dir / "meta.json"
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            # 验证必需字段
            assert 'git_commit' in meta, "meta.json 缺少 git_commit 字段"
            assert 'seed' in meta, "meta.json 缺少 seed 字段"
            assert meta['git_commit'] == git_commit, f"git_commit 不匹配: {meta['git_commit']} != {git_commit}"
            assert meta['seed'] == seed, f"seed 不匹配: {meta['seed']} != {seed}"


@pytest.mark.property
class TestIncompleteOutputDetection:
    """测试不完整输出的检测"""
    
    @given(
        exp_name=experiment_name,
        seed=seed_strategy
    )
    @settings(max_examples=50, deadline=None)
    def test_missing_metrics_detected(self, exp_name, seed):
        """
        **Feature: mixed-wm38-recognition, Property 8: 实验输出结构完整性**
        **Validates: Requirements 7.2**
        
        缺少 metrics.csv 应被检测到。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            exp_dir = create_mock_experiment_output(
                output_dir=output_dir,
                exp_name=exp_name,
                seed=seed,
                git_commit="unknown",
                include_metrics=False,  # 不包含 metrics
            )
            
            is_valid, errors = validate_experiment_output(exp_dir)
            
            assert not is_valid, "缺少 metrics.csv 应导致验证失败"
            assert any("metrics.csv" in e for e in errors), "错误信息应提及 metrics.csv"
    
    @given(
        exp_name=experiment_name,
        seed=seed_strategy
    )
    @settings(max_examples=50, deadline=None)
    def test_missing_meta_detected(self, exp_name, seed):
        """
        **Feature: mixed-wm38-recognition, Property 8: 实验输出结构完整性**
        **Validates: Requirements 7.3**
        
        缺少 meta.json 应被检测到。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            exp_dir = create_mock_experiment_output(
                output_dir=output_dir,
                exp_name=exp_name,
                seed=seed,
                git_commit="unknown",
                include_meta=False,  # 不包含 meta
            )
            
            is_valid, errors = validate_experiment_output(exp_dir)
            
            assert not is_valid, "缺少 meta.json 应导致验证失败"
            assert any("meta.json" in e for e in errors), "错误信息应提及 meta.json"
    
    @given(
        exp_name=experiment_name,
        seed=seed_strategy
    )
    @settings(max_examples=50, deadline=None)
    def test_missing_curves_detected(self, exp_name, seed):
        """
        **Feature: mixed-wm38-recognition, Property 8: 实验输出结构完整性**
        **Validates: Requirements 7.2**
        
        缺少 curves/ 目录应被检测到。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            exp_dir = create_mock_experiment_output(
                output_dir=output_dir,
                exp_name=exp_name,
                seed=seed,
                git_commit="unknown",
                include_curves=False,  # 不包含 curves
            )
            
            is_valid, errors = validate_experiment_output(exp_dir)
            
            assert not is_valid, "缺少 curves/ 目录应导致验证失败"
            assert any("curves" in e for e in errors), "错误信息应提及 curves"


# ============================================================================
# 单元测试：验证实际实验输出
# ============================================================================

@pytest.mark.unit
class TestRealExperimentOutput:
    """测试实际实验输出（如果存在）"""
    
    def test_existing_experiment_outputs_are_valid(self):
        """
        **Feature: mixed-wm38-recognition, Property 8: 实验输出结构完整性**
        **Validates: Requirements 7.2, 7.3, 7.4**
        
        results/目录下已完成的实验输出应符合结构要求。
        只检查同时具有 history.json 和 metrics.csv 的实验（表示训练和评估都已完成）。
        """
        results_dir = Path(__file__).parent.parent / 'results'
        if not results_dir.exists():
            pytest.skip("results目录不存在")
        
        exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        if not exp_dirs:
            pytest.skip("results目录下没有实验输出")
        
        completed_experiments = []
        for exp_dir in exp_dirs:
            # 跳过空目录或临时目录
            if not any(exp_dir.iterdir()):
                continue
            
            # 检查是否有 checkpoints 目录（表示是完整的实验）
            if not (exp_dir / "checkpoints").exists():
                continue
            
            # 只检查同时具有 history.json 和 metrics.csv 的实验
            # 这表示训练和评估都已完成
            has_history = (exp_dir / "history.json").exists()
            has_metrics = (exp_dir / "metrics.csv").exists()
            
            if has_history and has_metrics:
                completed_experiments.append(exp_dir)
        
        if not completed_experiments:
            pytest.skip("没有找到已完成的实验（需要同时有 history.json 和 metrics.csv）")
        
        for exp_dir in completed_experiments:
            is_valid, errors = validate_experiment_output(exp_dir)
            assert is_valid, f"{exp_dir.name} 验证失败: {errors}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
