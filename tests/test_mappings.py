# -*- coding: utf-8 -*-
"""
标签映射属性测试

Property 1: 38类到8类标签映射正确性
Validates: Requirements 2.2, 2.3
"""

import pytest
import numpy as np
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mappings import (
    LABEL_38_TO_8,
    DEFECT_NAMES_8,
    CLASS_NAME_MAPPING,
    map_38_to_8,
    validate_mapping,
)


# ============================================================================
# 策略定义
# ============================================================================

# 有效的38类标签
valid_label_38 = st.integers(min_value=0, max_value=37)

# 单缺陷类标签 (1-8)
single_defect_label = st.integers(min_value=1, max_value=8)

# 混合缺陷类标签 (9-37)
mixed_defect_label = st.integers(min_value=9, max_value=37)


# ============================================================================
# Property 1: 38类到8类标签映射正确性
# **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
# **Validates: Requirements 2.2, 2.3**
# ============================================================================

@pytest.mark.property
class TestLabelMappingCorrectness:
    """Property 1: 38类到8类标签映射正确性测试"""
    
    @given(label_38=valid_label_38)
    @settings(max_examples=100, deadline=None)
    def test_mapping_output_is_8_dimensional(self, label_38):
        """
        **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
        **Validates: Requirements 2.2, 2.3**
        
        对于任意有效的38类标签，映射结果应为8维向量。
        """
        label_8 = map_38_to_8(label_38)
        assert len(label_8) == 8, f"映射结果应为8维，实际为{len(label_8)}维"
        assert label_8.dtype == np.float32, f"映射结果应为float32类型"
    
    @given(label_38=valid_label_38)
    @settings(max_examples=100, deadline=None)
    def test_mapping_output_is_binary(self, label_38):
        """
        **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
        **Validates: Requirements 2.2, 2.3**
        
        对于任意有效的38类标签，映射结果的每个元素应为0或1。
        """
        label_8 = map_38_to_8(label_38)
        for i, val in enumerate(label_8):
            assert val in [0.0, 1.0], f"位置{i}的值应为0或1，实际为{val}"
    
    @given(label_38=single_defect_label)
    @settings(max_examples=100, deadline=None)
    def test_single_defect_is_one_hot(self, label_38):
        """
        **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
        **Validates: Requirements 2.2, 2.3**
        
        对于单缺陷类（1-8），映射结果应为one-hot向量（恰好一个1）。
        """
        label_8 = map_38_to_8(label_38)
        hot_count = int(np.sum(label_8))
        assert hot_count == 1, f"单缺陷类{label_38}应映射为one-hot，实际有{hot_count}个1"
    
    @given(label_38=single_defect_label)
    @settings(max_examples=100, deadline=None)
    def test_single_defect_hot_position_correct(self, label_38):
        """
        **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
        **Validates: Requirements 2.2, 2.3**
        
        对于单缺陷类（1-8），hot位置应对应正确的缺陷类型索引。
        单缺陷类1-8对应8类基础缺陷的索引0-7。
        """
        label_8 = map_38_to_8(label_38)
        hot_position = int(np.argmax(label_8))
        expected_position = label_38 - 1  # 类1对应索引0，类2对应索引1，...
        assert hot_position == expected_position, \
            f"单缺陷类{label_38}的hot位置应为{expected_position}，实际为{hot_position}"
    
    @given(label_38=mixed_defect_label)
    @settings(max_examples=100, deadline=None)
    def test_mixed_defect_is_multi_hot(self, label_38):
        """
        **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
        **Validates: Requirements 2.2, 2.3**
        
        对于混合缺陷类（9-37），映射结果应为multi-hot向量（至少两个1）。
        """
        label_8 = map_38_to_8(label_38)
        hot_count = int(np.sum(label_8))
        assert hot_count >= 2, f"混合缺陷类{label_38}应映射为multi-hot，实际只有{hot_count}个1"
    
    def test_normal_class_is_all_zeros(self):
        """
        **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
        **Validates: Requirements 2.2, 2.3**
        
        Normal类（0）应映射为全零向量。
        """
        label_8 = map_38_to_8(0)
        assert np.sum(label_8) == 0, "Normal类应映射为全零向量"
    
    @given(label_38=valid_label_38)
    @settings(max_examples=100, deadline=None)
    def test_mapping_is_deterministic(self, label_38):
        """
        **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
        **Validates: Requirements 2.2, 2.3**
        
        对于同一个38类标签，多次映射结果应完全一致。
        """
        result1 = map_38_to_8(label_38)
        result2 = map_38_to_8(label_38)
        assert np.array_equal(result1, result2), "映射应是确定性的"


@pytest.mark.property
class TestMappingCoverage:
    """映射覆盖率测试"""
    
    def test_all_38_classes_covered(self):
        """
        **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
        **Validates: Requirements 2.2, 2.3**
        
        LABEL_38_TO_8字典应覆盖所有38个类别。
        """
        assert len(LABEL_38_TO_8) == 38, f"应覆盖38个类别，实际覆盖{len(LABEL_38_TO_8)}个"
        for i in range(38):
            assert i in LABEL_38_TO_8, f"类别{i}未被覆盖"
    
    def test_validate_mapping_passes(self):
        """
        **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
        **Validates: Requirements 2.2, 2.3**
        
        validate_mapping()应返回通过状态。
        """
        passed, stats = validate_mapping()
        assert passed, f"映射验证失败: {stats}"
        assert stats['covered_classes'] == 38
        assert stats['normal_class'] == 1
        assert stats['single_defect_classes'] == 8
        assert stats['mixed_defect_classes'] == 29
    
    def test_defect_names_match_8_classes(self):
        """
        **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
        **Validates: Requirements 2.2, 2.3**
        
        DEFECT_NAMES_8应包含8个缺陷名称。
        """
        assert len(DEFECT_NAMES_8) == 8, f"应有8个缺陷名称，实际有{len(DEFECT_NAMES_8)}个"


@pytest.mark.property
class TestMappingConsistency:
    """映射一致性测试"""
    
    @given(label_38=mixed_defect_label)
    @settings(max_examples=100, deadline=None)
    def test_mixed_defect_components_in_class_name(self, label_38):
        """
        **Feature: mixed-wm38-recognition, Property 1: 38类到8类标签映射正确性**
        **Validates: Requirements 2.2, 2.3**
        
        对于混合缺陷类，映射结果中的hot位置应与类名中的成分一致。
        例如：C+EL类的映射应在Center和EL位置为1。
        """
        label_8 = map_38_to_8(label_38)
        class_name = CLASS_NAME_MAPPING[label_38]
        
        # 解析类名中的成分
        components = class_name.replace('+', ' ').split()
        
        # 缩写到索引的映射
        abbrev_to_idx = {
            'C': 0, 'Center': 0,
            'D': 1, 'Donut': 1,
            'EL': 2,
            'ER': 3,
            'L': 4, 'LOC': 4,
            'NF': 5,
            'S': 6,
            'Random': 7,
        }
        
        # 验证每个成分对应的位置为1
        for comp in components:
            if comp in abbrev_to_idx:
                idx = abbrev_to_idx[comp]
                assert label_8[idx] == 1.0, \
                    f"类{label_38}({class_name})的成分{comp}对应位置{idx}应为1"


# ============================================================================
# 边界测试
# ============================================================================

@pytest.mark.unit
class TestMappingEdgeCases:
    """映射边界情况测试"""
    
    def test_invalid_label_raises_error(self):
        """无效标签应抛出ValueError"""
        with pytest.raises(ValueError):
            map_38_to_8(-1)
        with pytest.raises(ValueError):
            map_38_to_8(38)
        with pytest.raises(ValueError):
            map_38_to_8(100)
    
    def test_first_and_last_class(self):
        """测试第一个和最后一个类别"""
        # Normal (0)
        label_0 = map_38_to_8(0)
        assert np.sum(label_0) == 0
        
        # D+L+ER+S (37)
        label_37 = map_38_to_8(37)
        assert np.sum(label_37) == 4  # 4个成分


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
