#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境和代码验证脚本

验证:
1. Python 环境和依赖
2. CUDA 可用性
3. 模型前向传播
4. 数据加载（如果数据存在）

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_python_version():
    """检查 Python 版本"""
    print("=" * 60)
    print("1. Python 版本检查")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 需要 Python 3.8 或更高版本")
        return False
    
    print("✅ Python 版本符合要求")
    return True


def check_dependencies():
    """检查依赖"""
    print("\n" + "=" * 60)
    print("2. 依赖检查")
    print("=" * 60)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("cv2", "OpenCV"),
    ]
    
    all_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} 未安装")
            all_ok = False
    
    return all_ok


def check_cuda():
    """检查 CUDA"""
    print("\n" + "=" * 60)
    print("3. CUDA 检查")
    print("=" * 60)
    
    import torch
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("✅ CUDA 配置正确")
        return True
    else:
        print("⚠️ CUDA 不可用，将使用 CPU 训练（速度较慢）")
        return True  # 不强制要求 CUDA


def check_model():
    """检查模型"""
    print("\n" + "=" * 60)
    print("4. 模型检查")
    print("=" * 60)
    
    try:
        import torch
        from models.multitask import WaferMultiTaskModel
        
        # 创建模型
        model = WaferMultiTaskModel(
            classification_classes=38,
            segmentation_classes=1,
            separation_enabled=False,
        )
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,}")
        
        # 测试前向传播
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        x = torch.randn(2, 3, 224, 224).to(device)
        outputs = model(x)
        
        print(f"输入形状: {x.shape}")
        print(f"分类输出形状: {outputs['cls_logits'].shape}")
        print(f"分割输出形状: {outputs['seg_mask'].shape}")
        
        print("✅ 模型前向传播正常")
        return True
        
    except Exception as e:
        print(f"❌ 模型检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data():
    """检查数据"""
    print("\n" + "=" * 60)
    print("5. 数据检查")
    print("=" * 60)
    
    data_root = Path("data/processed")
    
    if not data_root.exists():
        print(f"⚠️ 数据目录不存在: {data_root}")
        print("请先运行数据准备脚本:")
        print("  python scripts/prepare_mixedwm38.py --input data/raw/Wafer_Map_Datasets.npz --output data/processed")
        return True  # 不强制要求数据存在
    
    images_dir = data_root / "Images"
    labels_dir = data_root / "Labels"
    masks_dir = data_root / "Masks"
    
    if not images_dir.exists():
        print(f"⚠️ 图像目录不存在: {images_dir}")
        return True
    
    image_files = list(images_dir.glob("*.npy"))
    print(f"图像文件数量: {len(image_files)}")
    
    if len(image_files) == 0:
        print("⚠️ 没有找到图像文件")
        return True
    
    # 尝试加载数据集
    try:
        from data.dataset import MixedWM38Dataset
        
        dataset = MixedWM38Dataset(
            data_root=str(data_root),
            split="train",
            debug=True,
            max_per_class=2,
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本图像形状: {sample['image'].shape}")
            print(f"样本 mask 形状: {sample['mask'].shape}")
            print(f"样本标签 (38类): {sample['label_38'].item()}")
            print("✅ 数据加载正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_config():
    """检查配置"""
    print("\n" + "=" * 60)
    print("6. 配置检查")
    print("=" * 60)
    
    config_path = Path("configs/e0.yaml")
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        from config_schema import load_config, validate_config
        
        config = load_config(str(config_path))
        errors = validate_config(config)
        
        if errors:
            print(f"❌ 配置验证失败:")
            for err in errors:
                print(f"  - {err}")
            return False
        
        print(f"实验名称: {config.name}")
        print(f"批次大小: {config.data.batch_size}")
        print(f"学习率: {config.training.learning_rate}")
        print(f"训练轮数: {config.training.epochs}")
        print("✅ 配置文件有效")
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("MixedWM38 项目环境验证")
    print("=" * 60)
    
    results = []
    
    results.append(("Python 版本", check_python_version()))
    results.append(("依赖", check_dependencies()))
    results.append(("CUDA", check_cuda()))
    results.append(("模型", check_model()))
    results.append(("数据", check_data()))
    results.append(("配置", check_config()))
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "-" * 60)
    if all_passed:
        print("🎉 所有检查通过！环境配置正确。")
        print("\n下一步:")
        print("1. 准备数据: python scripts/prepare_mixedwm38.py --input data/raw/Wafer_Map_Datasets.npz --output data/processed --debug")
        print("2. 运行训练: python train.py --config configs/e0.yaml --debug")
    else:
        print("⚠️ 部分检查未通过，请根据上述提示修复问题。")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
