#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPM 训练脚本（用于尾部类生成式扩增）

Usage:
    conda run -n wafer-seg-class python scripts/train_ddpm.py --config configs/ddpm.yaml
    conda run -n wafer-seg-class python scripts/train_ddpm.py --config configs/ddpm.yaml --debug
    conda run -n wafer-seg-class python scripts/train_ddpm.py --config configs/ddpm.yaml --resume results/ddpm_tail/checkpoints/last.pt
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import yaml
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diffusion.ddpm import SimpleUNet, GaussianDiffusion
from diffusion.dataset import WaferIndexDataset
from diffusion.wafer_utils import load_label_counts, get_tail_classes
from data.sampler import WeightedClassSampler
from training.utils import set_seed, save_config_snapshot, save_meta_json


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
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


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    loss: float,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("loss", None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DDPM for tail augmentation")
    parser.add_argument("--config", "-c", required=True, help="DDPM config yaml")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    diff_cfg = config.get("diffusion", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    debug_mode = args.debug or exp_cfg.get("debug", False)
    exp_name = exp_cfg.get("name", "ddpm")
    if debug_mode:
        exp_name = f"{exp_name}_debug"

    results_dir = Path(output_cfg.get("results_dir", "results"))
    output_dir = results_dir / exp_name
    logger = setup_logging(output_dir)

    seed = exp_cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Debug mode: {debug_mode}")

    data_root = data_cfg.get("data_root", "data/processed")
    image_size = int(data_cfg.get("image_size", 64))
    tail_threshold = int(data_cfg.get("tail_threshold", 100))
    tail_only = bool(data_cfg.get("tail_only", True))
    balanced_sampling = bool(data_cfg.get("balanced_sampling", True))
    sampler_mode = data_cfg.get("sampler_mode", "sqrt_inverse")

    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 4))
    max_samples = data_cfg.get("max_samples", None)
    epochs = int(train_cfg.get("epochs", 100))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))

    if debug_mode:
        batch_size = int(data_cfg.get("debug_batch_size", 8))
        max_samples = data_cfg.get("debug_max_samples", 200)
        epochs = int(train_cfg.get("debug_epochs", 2))
        num_workers = 0

    label_counts = load_label_counts(Path(data_root) / "Labels")
    tail_classes = get_tail_classes(label_counts, tail_threshold)
    if tail_only and not tail_classes:
        raise RuntimeError("未找到尾部类别，无法进行 DDPM 训练")

    if tail_only:
        target_classes = sorted(tail_classes)
    else:
        target_classes = sorted(label_counts.keys())

    class_id_map = {cid: idx for idx, cid in enumerate(target_classes)}
    num_classes = len(class_id_map)

    dataset = WaferIndexDataset(
        data_root=data_root,
        image_size=image_size,
        class_filter=target_classes if tail_only else None,
        class_id_map=class_id_map,
        max_samples=max_samples,
        augment=True,
        seed=seed,
    )

    logger.info(f"DDPM dataset size: {len(dataset)}")
    logger.info(f"Tail-only: {tail_only}, classes: {num_classes}")
    if len(dataset) == 0:
        raise RuntimeError("DDPM 数据集为空，请检查 data_root 与 tail_threshold 设置")

    sampler = None
    if balanced_sampling:
        labels = [dataset[i]["label"].item() for i in range(len(dataset))]
        sampler = WeightedClassSampler(labels=labels, mode=sampler_mode)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = SimpleUNet(
        in_channels=1,
        base_channels=int(model_cfg.get("base_channels", 64)),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4])),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        num_classes=num_classes if model_cfg.get("class_conditional", True) else None,
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=int(diff_cfg.get("timesteps", 200)),
        beta_start=float(diff_cfg.get("beta_start", 1e-4)),
        beta_end=float(diff_cfg.get("beta_end", 0.02)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scaler = GradScaler(enabled=bool(train_cfg.get("amp_enabled", True)))

    start_epoch = 0
    best_loss = None
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        start_epoch, best_loss = load_checkpoint(Path(args.resume), model, optimizer, scaler)
        logger.info(f"Resumed at epoch {start_epoch}, best loss: {best_loss}")

    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    save_meta_json(
        output_dir,
        seed=seed,
        extra_info={
            "experiment": exp_name,
            "data_root": data_root,
            "image_size": image_size,
            "tail_only": tail_only,
            "tail_threshold": tail_threshold,
            "tail_classes": target_classes,
            "class_id_map": class_id_map,
            "num_classes": num_classes,
            "dataset_size": len(dataset),
        },
    )

    history = {"train_loss": []}
    save_every = int(train_cfg.get("save_every", 10))

    logger.info("Start training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            x = batch["image"].to(device)
            labels = batch["label"].to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device).long()

            with autocast(enabled=bool(train_cfg.get("amp_enabled", True))):
                loss = diffusion.p_losses(model, x, t, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        history["train_loss"].append(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f}")

        checkpoint_dir = output_dir / "checkpoints"
        save_checkpoint(model, optimizer, scaler, epoch + 1, avg_loss, checkpoint_dir / "last.pt")
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, scaler, epoch + 1, avg_loss, checkpoint_dir / f"epoch_{epoch+1}.pt")

        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, scaler, epoch + 1, avg_loss, checkpoint_dir / "best.pt")

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training completed. Best loss: {best_loss}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
