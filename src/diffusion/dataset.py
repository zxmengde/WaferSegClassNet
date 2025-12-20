# -*- coding: utf-8 -*-
"""
DDPM 训练用数据集（索引图）
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.mappings import CLASS_MAPPING
from .wafer_utils import rgb_to_index_map


class WaferIndexDataset(Dataset):
    """
    读取 RGB 晶圆图并转换为 0/1/2 索引图
    """

    def __init__(
        self,
        data_root: str,
        image_size: int = 64,
        class_filter: Optional[List[int]] = None,
        class_id_map: Optional[Dict[int, int]] = None,
        max_samples: Optional[int] = None,
        augment: bool = True,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.class_filter = set(class_filter) if class_filter else None
        self.class_id_map = class_id_map or {}
        self.max_samples = max_samples
        self.augment = augment
        self.seed = seed

        self.images_dir = self.data_root / "Images"
        self.labels_dir = self.data_root / "Labels"

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        image_files = sorted([p for p in self.images_dir.iterdir() if p.suffix == ".npy"])
        samples: List[Dict] = []

        for img_path in image_files:
            label_path = self.labels_dir / img_path.name
            if not label_path.exists():
                continue

            label_raw = np.load(label_path)
            label_str = str(label_raw)
            if label_str not in CLASS_MAPPING:
                continue

            label_38 = CLASS_MAPPING[label_str]
            if self.class_filter is not None and label_38 not in self.class_filter:
                continue

            label_mapped = self.class_id_map.get(label_38, label_38)

            samples.append(
                {
                    "image_path": str(img_path),
                    "label_38": label_38,
                    "label": label_mapped,
                }
            )

        if self.max_samples is not None and len(samples) > self.max_samples:
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(len(samples), size=self.max_samples, replace=False)
            samples = [samples[i] for i in indices]

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augment(self, index_map: np.ndarray) -> np.ndarray:
        if not self.augment:
            return index_map

        # 旋转（0/90/180/270）+ 翻转
        k = np.random.randint(0, 4)
        index_map = np.rot90(index_map, k)
        if np.random.rand() < 0.5:
            index_map = np.fliplr(index_map)
        if np.random.rand() < 0.5:
            index_map = np.flipud(index_map)
        return index_map

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        rgb = np.load(sample["image_path"])
        index_map = rgb_to_index_map(rgb)

        if index_map.shape[0] != self.image_size:
            index_map = cv2.resize(
                index_map.astype(np.uint8),
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

        index_map = self._apply_augment(index_map)

        # 映射到 [-1, 1]
        x = index_map.astype(np.float32) - 1.0
        x = torch.from_numpy(x).unsqueeze(0)  # (1, H, W)

        label = torch.tensor(sample["label"], dtype=torch.long)

        return {"image": x, "label": label}
