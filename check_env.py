#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境检查脚本
验证所有依赖是否正确安装，CUDA是否可用
"""

import sys

def check_dependency(name, import_name=None):
    """检查单个依赖是否可导入"""
    if import_name is None:
        import_name = name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    print("=" * 50)
    print("MixedWM38 项目环境检查")
    print("=" * 50)
    
    # 核心依赖列表
    dependencies = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("PyYAML", "yaml"),
        ("tqdm", "tqdm"),
        ("opencv-python", "cv2"),
        ("hypothesis", "hypothesis"),
        ("python-pptx", "pptx"),
    ]
    
    all_ok = True
    print("\n📦 依赖检查:")
    print("-" * 50)
    
    for name, import_name in dependencies:
        ok, info = check_dependency(name, import_name)
        if ok:
            print(f"  ✅ {name}: {info}")
        else:
            print(f"  ❌ {name}: 未安装 ({info})")
            all_ok = False
    
    # PyTorch CUDA 检查
    print("\n🖥️ GPU/CUDA 检查:")
    print("-" * 50)
    
    try:
        import torch
        print(f"  PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 编译版本: {torch.version.cuda}")
        print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
        
        if torch.cuda.is_available():
            print(f"  ✅ CUDA 可用")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    - 显存: {props.total_memory / 1024**3:.1f} GB")
                print(f"    - 计算能力: {props.major}.{props.minor}")
        else:
            print("  ⚠️ CUDA 不可用 - 将使用 CPU 训练")
            all_ok = False
    except Exception as e:
        print(f"  ❌ PyTorch 检查失败: {e}")
        all_ok = False
    
    # 项目结构检查
    print("\n📁 项目结构检查:")
    print("-" * 50)
    
    import os
    required_dirs = [
        "configs",
        "data",
        "data/raw",
        "data/processed",
        "docs",
        "logs",
        "results",
        "scripts",
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/visualization",
        "tests",
        "weights",
    ]
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ⚠️ {dir_path}/ (不存在，将自动创建)")
            os.makedirs(dir_path, exist_ok=True)
    
    # 数据文件检查
    print("\n📊 数据文件检查:")
    print("-" * 50)
    
    data_files = [
        ("data/raw/MixedWM38.npz", "MixedWM38 原始数据"),
        ("data/processed/Images", "处理后的图像目录"),
        ("data/processed/Labels", "处理后的标签目录"),
    ]
    
    for path, desc in data_files:
        if os.path.exists(path):
            print(f"  ✅ {desc}: {path}")
        else:
            print(f"  ⚠️ {desc}: {path} (未找到)")
    
    # 配置文件检查
    print("\n⚙️ 配置文件检查:")
    print("-" * 50)
    
    config_files = [
        "configs/e0.yaml",
        "configs/e0_debug.yaml",
    ]
    
    for cfg in config_files:
        if os.path.exists(cfg):
            print(f"  ✅ {cfg}")
        else:
            print(f"  ⚠️ {cfg} (未找到)")
    
    # 总结
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ 环境检查通过！可以开始训练。")
        print("\n下一步:")
        print("  1. 准备数据: python scripts/prepare_mixedwm38.py")
        print("  2. Debug训练: python train.py --config configs/e0_debug.yaml")
    else:
        print("⚠️ 部分检查未通过，请查看上方详情。")
        print("\n排查建议:")
        print("  1. 确保已激活正确的 conda 环境: conda activate wafer-seg-class")
        print("  2. 重新安装依赖: pip install -r requirements.txt")
        print("  3. 查看文档: docs/SETUP_WINDOWS.md")
    print("=" * 50)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
