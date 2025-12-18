# -*- coding: utf-8 -*-
"""
编码器测试

测试 WaferEncoder 的权重加载功能
"""

import os
import json
import tempfile
import pytest
import torch

from src.models.encoder import WaferEncoder


class TestWaferEncoder:
    """WaferEncoder 测试类"""
    
    def test_encoder_forward(self):
        """测试编码器前向传播"""
        encoder = WaferEncoder()
        x = torch.randn(2, 3, 224, 224)
        features, skips = encoder(x)
        
        assert features.shape == (2, 64, 14, 14)
        assert len(skips) == 4
        assert skips[0].shape == (2, 8, 224, 224)
        assert skips[1].shape == (2, 16, 112, 112)
        assert skips[2].shape == (2, 16, 56, 56)
        assert skips[3].shape == (2, 32, 28, 28)
    
    def test_encoder_output_channels(self):
        """测试编码器输出通道数"""
        encoder = WaferEncoder(base_channels=8)
        assert encoder.out_channels == 64  # base_channels * 8


class TestWeightLoading:
    """权重加载测试类
    
    验证 load_pretrained() 方法输出包含 matched/missing/unexpected 字段
    _Requirements: 4.2, 4.3_
    """
    
    def test_weight_loading_stats_fields(self):
        """测试权重加载统计字段完整性
        
        验证输出包含 matched/missing/unexpected 字段
        """
        encoder1 = WaferEncoder()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存checkpoint
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            torch.save({'state_dict': encoder1.state_dict()}, ckpt_path)
            
            # 加载权重
            encoder2 = WaferEncoder()
            stats = encoder2.load_pretrained(
                checkpoint_path=ckpt_path,
                output_dir=tmpdir
            )
            
            # 验证必需字段存在
            assert 'matched' in stats, "stats should contain 'matched' field"
            assert 'missing' in stats, "stats should contain 'missing' field"
            assert 'unexpected' in stats, "stats should contain 'unexpected' field"
            assert 'shape_mismatch' in stats, "stats should contain 'shape_mismatch' field"
            assert 'ignored_prefixes' in stats, "stats should contain 'ignored_prefixes' field"
    
    def test_weight_loading_json_output(self):
        """测试 weight_loading.json 文件生成"""
        encoder1 = WaferEncoder()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            torch.save({'state_dict': encoder1.state_dict()}, ckpt_path)
            
            encoder2 = WaferEncoder()
            encoder2.load_pretrained(
                checkpoint_path=ckpt_path,
                output_dir=tmpdir
            )
            
            # 验证 JSON 文件生成
            json_path = os.path.join(tmpdir, "weight_loading.json")
            assert os.path.exists(json_path), "weight_loading.json should be created"
            
            # 验证 JSON 内容
            with open(json_path, 'r') as f:
                saved_stats = json.load(f)
            
            assert 'matched' in saved_stats
            assert 'missing' in saved_stats
            assert 'unexpected' in saved_stats
    
    def test_weight_loading_with_key_mapping(self):
        """测试带 key mapping 的权重加载"""
        encoder1 = WaferEncoder()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存带前缀的 checkpoint
            state_dict = {'encoder.' + k: v for k, v in encoder1.state_dict().items()}
            ckpt_path = os.path.join(tmpdir, "test_ckpt_prefix.pt")
            torch.save({'state_dict': state_dict}, ckpt_path)
            
            # 使用 key_mapping 加载
            encoder2 = WaferEncoder()
            stats = encoder2.load_pretrained(
                checkpoint_path=ckpt_path,
                key_mapping={'strip_prefix': ['encoder.']},
                output_dir=tmpdir
            )
            
            # 验证成功匹配
            assert stats['matched'] > 0, "Should have matched keys after stripping prefix"
            assert stats['missing'] == 0, "Should have no missing keys"
    
    def test_weight_loading_with_extract_subtree(self):
        """测试 extract_subtree 功能"""
        encoder1 = WaferEncoder()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存嵌套的 checkpoint
            state_dict = {'model.encoder.' + k: v for k, v in encoder1.state_dict().items()}
            state_dict['model.other.param'] = torch.randn(10)  # 额外参数
            ckpt_path = os.path.join(tmpdir, "test_ckpt_nested.pt")
            torch.save({'state_dict': state_dict}, ckpt_path)
            
            # 使用 extract_subtree 加载
            encoder2 = WaferEncoder()
            stats = encoder2.load_pretrained(
                checkpoint_path=ckpt_path,
                key_mapping={
                    'extract_subtree': 'model.encoder',
                },
                output_dir=tmpdir
            )
            
            # 验证成功匹配
            assert stats['matched'] > 0, "Should have matched keys after extracting subtree"
    
    def test_weight_loading_shape_mismatch(self):
        """测试形状不匹配的处理"""
        encoder1 = WaferEncoder(base_channels=8)
        encoder2 = WaferEncoder(base_channels=16)  # 不同的通道数
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
            torch.save({'state_dict': encoder1.state_dict()}, ckpt_path)
            
            stats = encoder2.load_pretrained(
                checkpoint_path=ckpt_path,
                output_dir=tmpdir
            )
            
            # 应该有形状不匹配
            assert stats['shape_mismatch'] > 0, "Should detect shape mismatches"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
