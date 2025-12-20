#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPM 采样脚本（生成尾部类合成样本）

Usage:
    conda run -n wafer-seg-class python scripts/sample_ddpm.py --config configs/ddpm.yaml --ckpt results/ddpm_tail/checkpoints/best.pt
    conda run -n wafer-seg-class python scripts/sample_ddpm.py --config configs/ddpm.yaml --ckpt results/ddpm_tail/checkpoints/last.pt --debug
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diffusion.ddpm import SimpleUNet, GaussianDiffusion
from diffusion.wafer_utils import (
    load_label_counts,
    get_tail_classes,
    index_map_to_rgb,
    index_map_to_mask,
    get_label_vector,
)


def setup_logging(output_root: Path) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=True)
    log_file = output_root / "sample.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_existing(images_dir: Path, class_id: int) -> int:
    if not images_dir.exists():
        return 0
    prefix = f"Synth_{class_id}_"
    return len([p for p in images_dir.iterdir() if p.suffix == ".npy" and p.name.startswith(prefix)])


def ensure_output_dirs(output_root: Path, overwrite: bool) -> Dict[str, Path]:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    images_dir = output_root / "Images"
    labels_dir = output_root / "Labels"
    masks_dir = output_root / "Masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    return {"images": images_dir, "labels": labels_dir, "masks": masks_dir}


def sample_for_class(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    class_id: int,
    class_index: int,
    image_size: int,
    output_image_size: int,
    batch_size: int,
    device: torch.device,
    images_dir: Path,
    labels_dir: Path,
    masks_dir: Path,
    start_index: int,
    needed: int,
    min_defect_ratio: float,
    max_defect_ratio: float,
) -> Dict[str, int]:
    generated = 0
    attempts = 0
    max_attempts = max(needed * 5, batch_size)

    while generated < needed and attempts < max_attempts:
        current_batch = min(batch_size, needed - generated)
        labels = torch.full((current_batch,), class_index, device=device, dtype=torch.long)
        samples = diffusion.sample(
            model=model,
            image_size=image_size,
            batch_size=current_batch,
            device=device,
            labels=labels,
        )
        samples = samples.detach().cpu().numpy()

        for i in range(current_batch):
            if generated >= needed:
                break

            x = samples[i, 0]
            index_map = np.rint(np.clip(x + 1.0, 0, 2)).astype(np.uint8)

            if output_image_size != image_size:
                index_map = cv2.resize(
                    index_map,
                    (output_image_size, output_image_size),
                    interpolation=cv2.INTER_NEAREST,
                )

            defect_ratio = float(np.mean(index_map == 2))
            if defect_ratio < min_defect_ratio or defect_ratio > max_defect_ratio:
                continue

            filename = f"Synth_{class_id}_{start_index + generated:05d}.npy"
            rgb = index_map_to_rgb(index_map)
            mask = index_map_to_mask(index_map)
            label_vec = get_label_vector(class_id)

            np.save(images_dir / filename, rgb)
            np.save(labels_dir / filename, label_vec)
            np.save(masks_dir / filename, mask)

            generated += 1
            attempts += 1

        attempts += 1

    return {"generated": generated, "attempts": attempts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample DDPM to generate synthetic tail samples")
    parser.add_argument("--config", "-c", required=True, help="DDPM config yaml")
    parser.add_argument("--ckpt", required=True, help="DDPM checkpoint path")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    diff_cfg = config.get("diffusion", {})
    sample_cfg = config.get("sampling", {})

    debug_mode = args.debug or exp_cfg.get("debug", False)
    data_root = data_cfg.get("data_root", "data/processed")
    image_size = int(data_cfg.get("image_size", 64))
    tail_threshold = int(data_cfg.get("tail_threshold", 100))
    tail_only = bool(data_cfg.get("tail_only", True))
    class_conditional = bool(model_cfg.get("class_conditional", True))

    output_root = Path(sample_cfg.get("output_root", "data/synthetic/ddpm"))
    output_image_size = int(sample_cfg.get("output_image_size", 224))
    target_count = sample_cfg.get("target_count", None)
    samples_per_class = sample_cfg.get("samples_per_class", None)
    max_samples_per_class = sample_cfg.get("max_samples_per_class", None)
    min_defect_ratio = float(sample_cfg.get("min_defect_ratio", 0.001))
    max_defect_ratio = float(sample_cfg.get("max_defect_ratio", 0.4))
    batch_size = int(sample_cfg.get("batch_size", 64))
    seed = int(sample_cfg.get("seed", exp_cfg.get("seed", 42)))
    overwrite = bool(sample_cfg.get("overwrite", False))

    if debug_mode:
        target_count = 10
        batch_size = min(batch_size, 8)

    if target_count is None and samples_per_class is None:
        raise RuntimeError("sampling.target_count 或 sampling.samples_per_class 必须至少设置一个")

    if not class_conditional:
        raise RuntimeError("当前采样脚本要求 class_conditional=true")

    logger = setup_logging(output_root)
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Debug mode: {debug_mode}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    label_counts = load_label_counts(Path(data_root) / "Labels")
    tail_classes = get_tail_classes(label_counts, tail_threshold)
    if tail_only:
        target_classes = sorted(tail_classes)
    else:
        target_classes = sorted(label_counts.keys())
    if not target_classes:
        raise RuntimeError("未找到目标类别，无法进行采样")

    class_id_map = {cid: idx for idx, cid in enumerate(target_classes)}

    output_dirs = ensure_output_dirs(output_root, overwrite)
    images_dir = output_dirs["images"]
    labels_dir = output_dirs["labels"]
    masks_dir = output_dirs["masks"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = SimpleUNet(
        in_channels=1,
        base_channels=int(model_cfg.get("base_channels", 64)),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4])),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        num_classes=len(class_id_map),
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=int(diff_cfg.get("timesteps", 200)),
        beta_start=float(diff_cfg.get("beta_start", 1e-4)),
        beta_end=float(diff_cfg.get("beta_end", 0.02)),
    ).to(device)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generation_stats = {
        "generated_per_class": {},
        "attempts_per_class": {},
        "existing_per_class": {},
        "real_counts": {},
    }

    for class_id in target_classes:
        real_count = int(label_counts.get(class_id, 0))
        existing_count = count_existing(images_dir, class_id)
        generation_stats["existing_per_class"][str(class_id)] = existing_count
        generation_stats["real_counts"][str(class_id)] = real_count

        if samples_per_class is not None:
            needed = int(samples_per_class)
        else:
            needed = max(0, int(target_count) - real_count - existing_count)

        if max_samples_per_class is not None:
            needed = min(needed, int(max_samples_per_class))

        if needed <= 0:
            logger.info(f"Class {class_id}: skip (real={real_count}, existing={existing_count})")
            generation_stats["generated_per_class"][str(class_id)] = 0
            generation_stats["attempts_per_class"][str(class_id)] = 0
            continue

        logger.info(f"Class {class_id}: need {needed} samples")
        stats = sample_for_class(
            model=model,
            diffusion=diffusion,
            class_id=class_id,
            class_index=class_id_map[class_id],
            image_size=image_size,
            output_image_size=output_image_size,
            batch_size=batch_size,
            device=device,
            images_dir=images_dir,
            labels_dir=labels_dir,
            masks_dir=masks_dir,
            start_index=existing_count,
            needed=needed,
            min_defect_ratio=min_defect_ratio,
            max_defect_ratio=max_defect_ratio,
        )
        generation_stats["generated_per_class"][str(class_id)] = stats["generated"]
        generation_stats["attempts_per_class"][str(class_id)] = stats["attempts"]

    generated_this_run = sum(generation_stats["generated_per_class"].values())
    total_existing = sum(generation_stats["existing_per_class"].values())
    total_generated = total_existing + generated_this_run
    logger.info(f"Generated this run: {generated_this_run}")
    logger.info(f"Total synthetic after run: {total_generated}")

    stats_payload = {
        "config": args.config,
        "checkpoint": args.ckpt,
        "data_root": data_root,
        "output_root": str(output_root),
        "image_size": image_size,
        "output_image_size": output_image_size,
        "tail_threshold": tail_threshold,
        "tail_only": tail_only,
        "target_classes": target_classes,
        "target_count": target_count,
        "samples_per_class": samples_per_class,
        "max_samples_per_class": max_samples_per_class,
        "min_defect_ratio": min_defect_ratio,
        "max_defect_ratio": max_defect_ratio,
        "generated_this_run": generated_this_run,
        "total_generated": total_generated,
        "generated_per_class": generation_stats["generated_per_class"],
        "attempts_per_class": generation_stats["attempts_per_class"],
        "existing_per_class": generation_stats["existing_per_class"],
        "real_counts": generation_stats["real_counts"],
    }

    with open(output_root / "synthetic_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    with open(output_root / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    logger.info(f"Sampling completed. Stats saved to {output_root / 'synthetic_stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
