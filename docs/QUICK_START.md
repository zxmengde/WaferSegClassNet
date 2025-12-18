# 快速开始指南

## 当前状态

你的环境正在配置中。PyTorch正在后台下载（约2.4GB），这可能需要几个小时。

## 检查下载进度

打开一个新的命令行窗口，运行：

```bash
conda activate wafer-seg-class
python -c "import torch; print('PyTorch已安装！版本:', torch.__version__)"
```

**如果看到版本号**：PyTorch已安装完成，可以继续！  
**如果看到错误**：PyTorch还在下载中，请耐心等待。

## 等待期间可以做什么

1. **准备数据**：
   - 下载 MixedWM38 数据集的 `Wafer_Map_Datasets.npz` 文件
   - 放到 `data/raw/` 目录下

2. **阅读文档**：
   - `docs/DATASET_LAYOUT.md` - 了解数据格式
   - `docs/SETUP_WINDOWS.md` - 环境配置详情
   - `AGENTS.md` - 项目总体规划

3. **了解项目结构**：
   ```
   WaferSegClassNet/
   ├── src/              # 源代码
   │   ├── models/       # 模型定义
   │   ├── data/         # 数据加载
   │   ├── training/     # 训练逻辑
   │   └── evaluation/   # 评估逻辑
   ├── configs/          # 配置文件
   ├── scripts/          # 工具脚本
   └── docs/             # 文档
   ```

## PyTorch安装完成后的第一步

### 1. 验证环境

```bash
conda activate wafer-seg-class
python scripts/verify_setup.py
```

应该看到所有依赖都是 ✅

### 2. 准备数据（Debug模式）

```bash
python scripts/prepare_mixedwm38.py --input data/raw/Wafer_Map_Datasets.npz --output data/processed --debug
```

这会创建一个小数据集（每类5个样本）用于快速测试。

### 3. 运行第一个训练（5分钟内完成）

```bash
python train.py --config configs/e0.yaml --debug
```

这会：
- 训练一个小模型（2个epoch）
- 自动评估
- 生成所有可视化结果
- 输出到 `results/e0_debug/`

### 4. 查看结果

```bash
# 查看指标
type results\e0_debug\metrics.csv

# 查看混淆矩阵
start results\e0_debug\confusion_matrix.png
```

## 遇到问题？

### 问题1：PyTorch下载太慢

**解决方案**：使用国内镜像（需要重新安装）

```bash
conda activate wafer-seg-class
pip uninstall torch torchvision -y
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题2：CUDA不可用

**不用担心**：代码会自动检测并使用CPU训练。虽然慢一些，但debug模式仍然可以在合理时间内完成。

### 问题3：显存不足

**解决方案**：在配置文件中降低batch_size

```yaml
# configs/e0.yaml
training:
  batch_size: 8  # 从16降到8或更小
```

## 下一步学习路径

1. **理解代码结构**：阅读 `src/README.md`
2. **学习配置系统**：查看 `configs/e0.yaml` 的注释
3. **运行完整训练**：去掉 `--debug` 标志
4. **对比实验**：依次运行 E0, E1, E2, E3

## 需要帮助？

查看 `docs/LEARNER_GUIDE.md` 获取：
- 详细的命令说明
- 常见错误排查
- 关键概念解释
- 训练日志解读

---

**提示**：不要关闭正在下载PyTorch的命令行窗口！让它在后台继续运行。
