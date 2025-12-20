#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WM-811K 数据准备脚本（用于 SSL 预训练）

将 WM811K.pkl 中的 waferMap 转为 224x224 的 RGB .npy 文件。

Usage:
    conda run -n wafer-seg-class python scripts/prepare_wm811k.py --input data/raw/MIR-WM811K/Python/WM811K.pkl --output data/wm811k
    conda run -n wafer-seg-class python scripts/prepare_wm811k.py --input data/raw/MIR-WM811K/Python/WM811K.pkl --output data/wm811k --max-samples 50000
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


RGB_PALETTE = np.array(
    [
        [255, 0, 255],   # background
        [0, 255, 255],   # wafer
        [255, 255, 0],   # defect
    ],
    dtype=np.uint8,
)


def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "prepare_wm811k.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def resolve_metadata_path(output_dir: Path) -> Path:
    """避免覆盖已有 metadata.csv"""
    base = output_dir / "metadata.csv"
    if not base.exists():
        return base
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"metadata_{ts}.csv"


def normalize_wafer_map(wafer_map: np.ndarray, image_size: int) -> np.ndarray:
    """将 waferMap 转为统一尺寸的 RGB 图像"""
    if wafer_map is None:
        return None
    wafer_map = np.asarray(wafer_map)
    if wafer_map.ndim != 2:
        return None
    wafer_map = np.clip(wafer_map, 0, 2).astype(np.uint8)
    if wafer_map.shape != (image_size, image_size):
        wafer_map = cv2.resize(
            wafer_map,
            (image_size, image_size),
            interpolation=cv2.INTER_NEAREST,
        )
    return RGB_PALETTE[wafer_map]


def prepare_wm811k(
    input_file: str,
    output_dir: str,
    image_size: int = 224,
    max_samples: int = None,
    log_dir: str = "logs",
) -> dict:
    """准备 WM-811K 数据集"""
    logger = setup_logging(log_dir)
    input_path = Path(input_file)
    output_path = Path(output_dir)
    images_dir = output_path / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"[Error] 输入文件不存在: {input_path}")
        return {}

    logger.info(f"[Info] Loading WM-811K from: {input_path}")
    df = pd.read_pickle(input_path)
    if not isinstance(df, pd.DataFrame):
        logger.error("[Error] WM811K.pkl 不是 DataFrame")
        return {}
    if "waferMap" not in df.columns:
        logger.error("[Error] 未找到 waferMap 列")
        return {}

    total_rows = len(df)
    logger.info(f"[Info] Total rows: {total_rows}")
    logger.info(f"[Info] Columns: {list(df.columns)}")

    metadata_path = resolve_metadata_path(output_path)
    logger.info(f"[Info] Metadata output: {metadata_path}")

    processed = 0
    skipped_invalid = 0
    skipped_existing = 0

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "lotName",
                "waferIndex",
                "dieSize",
                "failureType",
                "trainTestLabel",
                "orig_height",
                "orig_width",
                "image_path",
            ]
        )

        for row in tqdm(df.itertuples(), total=total_rows, desc="Processing"):
            if max_samples is not None and processed >= max_samples:
                break

            idx = int(getattr(row, "Index"))
            wafer_map = getattr(row, "waferMap", None)

            if wafer_map is None:
                skipped_invalid += 1
                continue

            wafer_map = np.asarray(wafer_map)
            if wafer_map.ndim != 2:
                skipped_invalid += 1
                continue

            out_path = images_dir / f"Image_{idx:06d}.npy"
            if out_path.exists():
                skipped_existing += 1
                continue

            rgb_image = normalize_wafer_map(wafer_map, image_size=image_size)
            if rgb_image is None:
                skipped_invalid += 1
                continue

            np.save(out_path, rgb_image.astype(np.uint8))

            writer.writerow(
                [
                    idx,
                    getattr(row, "lotName", ""),
                    getattr(row, "waferIndex", ""),
                    getattr(row, "dieSize", ""),
                    getattr(row, "failureType", ""),
                    getattr(row, "trainTestLabel", ""),
                    int(wafer_map.shape[0]),
                    int(wafer_map.shape[1]),
                    out_path.as_posix(),
                ]
            )

            processed += 1

            if processed % 5000 == 0:
                f.flush()
                logger.info(f"[Info] Processed {processed} samples...")

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "image_size": image_size,
        "total_rows": total_rows,
        "processed": processed,
        "skipped_invalid": skipped_invalid,
        "skipped_existing": skipped_existing,
        "metadata_csv": str(metadata_path),
    }

    stats_path = output_path / "wm811k_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        import json
        json.dump(stats, f, indent=2)

    logger.info(f"[Info] Done. Processed={processed}, skipped_invalid={skipped_invalid}, skipped_existing={skipped_existing}")
    logger.info(f"[Info] Images saved to: {images_dir}")
    logger.info(f"[Info] Stats saved to: {stats_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare WM-811K dataset for SSL")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/raw/MIR-WM811K/Python/WM811K.pkl",
        help="Input WM811K.pkl path",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/wm811k",
        help="Output directory",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Output image size (square)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for quick debug)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory",
    )

    args = parser.parse_args()

    prepare_wm811k(
        input_file=args.input,
        output_dir=args.output,
        image_size=args.image_size,
        max_samples=args.max_samples,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
