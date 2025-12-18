# Windows 环境搭建指南

## 系统要求

- **操作系统**: Windows 11 (24H2) 或 Windows 10 (21H2+)
- **CPU**: Intel Core Ultra 9 285H 或同等性能
- **RAM**: 32GB+
- **GPU**: NVIDIA RTX 4070 SUPER 12GB 或同等性能
- **CUDA**: 12.1+
- **cuDNN**: 8.9+

## 方法一：使用 Conda（推荐）

### 1. 安装 Miniconda

1. 下载 Miniconda：https://docs.conda.io/en/latest/miniconda.html
2. 运行安装程序，选择"Just Me"
3. 勾选"Add Miniconda to PATH"（可选）

### 2. 配置国内镜像源（可选，加速下载）

创建或编辑 `~/.condarc` 文件：

```yaml
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

### 3. 创建环境

```bash
# 方法 A：使用 environment.yml
conda env create -f environment.yml

# 方法 B：手动创建
conda create -n wafer-seg-class python=3.10 -y
conda activate wafer-seg-class
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

### 4. 激活环境

```bash
conda activate wafer-seg-class
```

### 5. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

预期输出：
```
PyTorch: 2.x.x
CUDA available: True
GPU: NVIDIA GeForce RTX 4070 SUPER
```

## 方法二：使用 pip + venv

### 1. 安装 Python 3.10

1. 下载 Python 3.10：https://www.python.org/downloads/
2. 运行安装程序，勾选"Add Python to PATH"

### 2. 创建虚拟环境

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 安装 PyTorch（CUDA 12.1）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. 安装其他依赖

```bash
pip install -r requirements.txt
```

## CUDA 版本检查

### 检查 NVIDIA 驱动版本

```bash
nvidia-smi
```

预期输出应显示：
- Driver Version: 5xx.xx 或更高
- CUDA Version: 12.x

### 检查 CUDA Toolkit

```bash
nvcc --version
```

如果未安装 CUDA Toolkit，可以从 NVIDIA 官网下载：
https://developer.nvidia.com/cuda-downloads

## 常见问题排查

### 问题 1：CUDA not available

**症状**：`torch.cuda.is_available()` 返回 `False`

**排查步骤**：
1. 检查 NVIDIA 驱动是否安装：`nvidia-smi`
2. 检查 PyTorch CUDA 版本是否匹配：
   ```python
   import torch
   print(torch.version.cuda)  # 应该显示 12.1
   ```
3. 重新安装 PyTorch：
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### 问题 2：ModuleNotFoundError

**症状**：`ModuleNotFoundError: No module named 'xxx'`

**解决方案**：
1. 确认虚拟环境已激活
2. 重新安装依赖：`pip install -r requirements.txt`

### 问题 3：CUDA out of memory

**症状**：`RuntimeError: CUDA out of memory`

**解决方案**：
1. 降低 batch_size（在配置文件中修改）
2. 启用 AMP 混合精度训练
3. 使用梯度累积（grad_accum_steps）
4. 降低 image_size

详见 `docs/LEARNER_GUIDE.md` 中的显存不足排查步骤。

### 问题 4：pip 安装超时

**解决方案**：使用国内镜像源

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或配置永久镜像源：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 5：conda 环境创建失败

**解决方案**：
1. 更新 conda：`conda update conda`
2. 清理缓存：`conda clean --all`
3. 手动创建环境（见方法一步骤3的方法B）

## 验证完整安装

运行以下命令验证所有依赖：

```bash
python -c "
import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import yaml
import tqdm
import cv2
import hypothesis
import pptx

print('All dependencies installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

## 下一步

环境配置完成后，请参考：
- `docs/DATASET_LAYOUT.md` - 数据准备
- `docs/LEARNER_GUIDE.md` - 新手学习指南
