# -*- coding: utf-8 -*-
"""
配置系统属性测试

Property 9: 配置文件YAML格式有效性
Validates: Requirements 2.7
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_schema import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    LossConfig,
    OutputConfig,
    AugmentationConfig,
    load_config,
    save_config,
    validate_config,
    validate_yaml_file,
    load_and_validate_config,
    REQUIRED_FIELDS,
    VALID_VALUES
)


# ============================================================================
# 策略定义：生成有效的配置值
# ============================================================================

valid_classification_modes = st.sampled_from(VALID_VALUES['classification_mode'])
valid_encoders = st.sampled_from(VALID_VALUES['encoder'])
valid_optimizers = st.sampled_from(VALID_VALUES['optimizer'])
valid_schedulers = st.sampled_from(VALID_VALUES['scheduler'])
valid_classification_losses = st.sampled_from(VALID_VALUES['classification_loss'])
valid_segmentation_losses = st.sampled_from(VALID_VALUES['segmentation_loss'])

# 正数策略
positive_int = st.integers(min_value=1, max_value=1000)
positive_float = st.floats(min_value=0.0001, max_value=1.0, allow_nan=False, allow_infinity=False)
image_size = st.tuples(
    st.integers(min_value=32, max_value=512),
    st.integers(min_value=32, max_value=512)
)

# 实验名称策略
experiment_name = st.text(
    alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='_-'),
    min_size=1,
    max_size=50
).filter(lambda x: len(x.strip()) > 0 and x[0].isalpha())


# ============================================================================
# Property 9: 配置文件YAML格式有效性
# **Feature: mixed-wm38-recognition, Property 9: 配置文件YAML格式有效性**
# **Validates: Requirements 2.7**
# ============================================================================

@pytest.mark.property
class TestConfigYAMLValidity:
    """Property 9: 配置文件YAML格式有效性测试"""
    
    @given(
        name=experiment_name,
        seed=st.integers(min_value=0, max_value=2**31-1),
        debug=st.booleans(),
        batch_size=positive_int,
        epochs=positive_int,
        learning_rate=positive_float,
        classification_mode=valid_classification_modes,
        encoder=valid_encoders,
        optimizer=valid_optimizers,
        classification_loss=valid_classification_losses
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_config_roundtrip(
        self,
        name,
        seed,
        debug,
        batch_size,
        epochs,
        learning_rate,
        classification_mode,
        encoder,
        optimizer,
        classification_loss
    ):
        """
        **Feature: mixed-wm38-recognition, Property 9: 配置文件YAML格式有效性**
        **Validates: Requirements 2.7**
        
        对于任意有效的配置参数组合，保存为YAML后应能被正确解析，
        且解析后的配置应与原始配置一致。
        """
        # 构造配置
        config = ExperimentConfig(
            name=name,
            seed=seed,
            debug=debug,
            data=DataConfig(
                batch_size=batch_size,
                classification_mode=classification_mode
            ),
            model=ModelConfig(encoder=encoder),
            training=TrainingConfig(
                epochs=epochs,
                learning_rate=learning_rate,
                optimizer=optimizer
            ),
            loss=LossConfig(classification=classification_loss)
        )
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            temp_path = f.name
        
        try:
            save_config(config, temp_path)
            
            # 验证文件可被YAML解析
            with open(temp_path, 'r', encoding='utf-8') as f:
                parsed = yaml.safe_load(f)
            
            assert parsed is not None, "YAML解析结果不应为空"
            assert isinstance(parsed, dict), "YAML解析结果应为字典"
            
            # 验证关键字段存在
            assert 'name' in parsed, "配置应包含name字段"
            assert 'seed' in parsed, "配置应包含seed字段"
            assert 'data' in parsed, "配置应包含data字段"
            assert 'model' in parsed, "配置应包含model字段"
            assert 'training' in parsed, "配置应包含training字段"
            
            # 验证值一致性
            assert parsed['name'] == name, f"name不一致: {parsed['name']} != {name}"
            assert parsed['seed'] == seed, f"seed不一致: {parsed['seed']} != {seed}"
            assert parsed['data']['batch_size'] == batch_size
            assert parsed['training']['epochs'] == epochs
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @given(
        name=experiment_name,
        seed=st.integers(min_value=0, max_value=2**31-1)
    )
    @settings(max_examples=100, deadline=None)
    def test_config_validation_passes_for_valid_config(self, name, seed):
        """
        **Feature: mixed-wm38-recognition, Property 9: 配置文件YAML格式有效性**
        **Validates: Requirements 2.7**
        
        对于任意使用默认值的有效配置，验证函数应返回空错误列表。
        """
        config = ExperimentConfig(name=name, seed=seed)
        errors = validate_config(config)
        assert errors == [], f"有效配置不应有验证错误: {errors}"


@pytest.mark.property
class TestConfigYAMLFileValidation:
    """测试YAML文件验证功能"""
    
    def test_existing_config_files_are_valid(self):
        """
        **Feature: mixed-wm38-recognition, Property 9: 配置文件YAML格式有效性**
        **Validates: Requirements 2.7**
        
        configs/目录下的所有YAML文件应能被正确解析且包含所有必需字段。
        """
        configs_dir = Path(__file__).parent.parent / 'configs'
        if not configs_dir.exists():
            pytest.skip("configs目录不存在")
        
        yaml_files = list(configs_dir.glob('*.yaml'))
        assert len(yaml_files) > 0, "configs目录应包含至少一个YAML文件"
        
        for yaml_file in yaml_files:
            is_valid, errors = validate_yaml_file(str(yaml_file))
            assert is_valid, f"{yaml_file.name} 验证失败: {errors}"
            
            # 进一步验证可以加载为配置对象
            config, validation_errors = load_and_validate_config(str(yaml_file))
            assert config is not None, f"{yaml_file.name} 加载失败"
            assert validation_errors == [], f"{yaml_file.name} 配置验证失败: {validation_errors}"


# ============================================================================
# 单元测试：验证边界情况
# ============================================================================

@pytest.mark.unit
class TestConfigValidation:
    """配置验证单元测试"""
    
    def test_invalid_classification_mode_rejected(self):
        """无效的分类模式应被拒绝"""
        config = ExperimentConfig(
            name="test",
            data=DataConfig(classification_mode="invalid_mode")
        )
        errors = validate_config(config)
        assert any("classification_mode" in e for e in errors)
    
    def test_invalid_encoder_rejected(self):
        """无效的编码器应被拒绝"""
        config = ExperimentConfig(
            name="test",
            model=ModelConfig(encoder="invalid_encoder")
        )
        errors = validate_config(config)
        assert any("encoder" in e for e in errors)
    
    def test_negative_learning_rate_rejected(self):
        """负学习率应被拒绝"""
        config = ExperimentConfig(
            name="test",
            training=TrainingConfig(learning_rate=-0.001)
        )
        errors = validate_config(config)
        assert any("learning_rate" in e for e in errors)
    
    def test_zero_epochs_rejected(self):
        """零epoch应被拒绝"""
        config = ExperimentConfig(
            name="test",
            training=TrainingConfig(epochs=0)
        )
        errors = validate_config(config)
        assert any("epochs" in e for e in errors)
    
    def test_invalid_yaml_file_rejected(self):
        """无效的YAML文件应被拒绝"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            is_valid, errors = validate_yaml_file(temp_path)
            assert not is_valid
            assert len(errors) > 0
        finally:
            os.unlink(temp_path)
    
    def test_missing_file_rejected(self):
        """不存在的文件应被拒绝"""
        is_valid, errors = validate_yaml_file("/nonexistent/path/config.yaml")
        assert not is_valid
        assert any("not found" in e for e in errors)
    
    def test_empty_config_rejected(self):
        """空配置文件应被拒绝"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            is_valid, errors = validate_yaml_file(temp_path)
            assert not is_valid
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
