#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 MixedWM38 数据集的像素值范围
"""

import numpy as np

# 加载数据
print("加载数据...")
data = np.load("data/raw/MixedWM38.npz")
images = data["arr_0"]
labels = data["arr_1"]

print(f"数据集大小: {len(images)}")
print(f"图像形状: {images[0].shape}")
print(f"标签形状: {labels[0].shape}")

# 检查像素值范围
print("\n检查像素值范围...")
unique_values = np.unique(images)
print(f"唯一像素值: {unique_values}")
print(f"最小值: {images.min()}")
print(f"最大值: {images.max()}")

# 统计每个像素值的出现次数
print("\n像素值分布:")
for val in unique_values:
    count = np.sum(images == val)
    percentage = count / images.size * 100
    print(f"  值 {val}: {count:,} ({percentage:.2f}%)")

# 检查几个样本
print("\n前5个样本的像素值范围:")
for i in range(min(5, len(images))):
    print(f"  样本 {i}: min={images[i].min()}, max={images[i].max()}, unique={np.unique(images[i])}")
