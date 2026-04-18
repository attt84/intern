from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import ExperimentConfig, load_experiment_config, save_experiment_config
from src.dataset import NYUv2SegmentationDataset
from src.engine import evaluate, train_one_epoch
from src.model import RGBDUNet
from src.transforms import build_transforms
from src.utils import count_parameters, ensure_dir, resolve_device, save_checkpoint, seed_everything, write_history_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NYUv2 semantic segmentation model.")
    parser.add_argument("--config", required=True, help="Path to the experiment config JSON.")
    parser.add_argument("--data-dir", help="Optional override for the dataset root.")
    parser.add_argument("--output-dir", help="Optional override for the output root.")
    parser.add_argument("--device", help="Device override. Example: cuda, cpu, mps.")
    parser.add_argument("--epochs", type=int, help="Optional epoch override.")
    parser.add_argument("--batch-size", type=int, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, help="Optional DataLoader worker override.")
    parser.add_argument("--resume", help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.data_dir:
        config.dataset.data_dir = Path(args.data_dir).resolve()
    if args.output_dir:
        config.output.root_dir = Path(args.output_dir).resolve()
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    return config


def make_loader(
    dataset: NYUv2SegmentationDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_experiment_config(args.config), args)
    device = resolve_device(args.device)
    seed_everything(config.training.seed)

    train_transform = build_transforms(
        config.dataset.image_size,
        train=True,
        rgb_mean=config.dataset.rgb_mean,
        rgb_std=config.dataset.rgb_std,
        depth_mean=config.dataset.depth_mean,
        depth_std=config.dataset.depth_std,
    )
    eval_transform = build_transforms(
        config.dataset.image_size,
        train=False,
        rgb_mean=config.dataset.rgb_mean,
        rgb_std=config.dataset.rgb_std,
        depth_mean=config.dataset.depth_mean,
        depth_std=config.dataset.depth_std,
    )

    train_dataset = NYUv2SegmentationDataset(
        config.dataset.data_dir,
        config.dataset.train_split,
        transform=train_transform,
        require_depth=config.model.use_depth,
        require_labels=True,
    )
    valid_dataset = NYUv2SegmentationDataset(
        config.dataset.data_dir,
        config.dataset.valid_split,
        transform=eval_transform,
        require_depth=config.model.use_depth,
        require_labels=True,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise RuntimeError("Train and valid splits must contain at least one sample.")

    train_loader = make_loader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        device=device,
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        device=device,
    )

    model = RGBDUNet(
        num_classes=config.dataset.num_classes,
        base_channels=config.model.base_channels,
        use_depth=config.model.use_depth,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=config.dataset.ignore_index)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if config.training.amp and device.type == "cuda" else None

    experiment_dir = ensure_dir(config.experiment_dir())
    checkpoint_dir = ensure_dir(config.checkpoint_dir())
    report_dir = ensure_dir(config.report_dir())
    ensure_dir(config.prediction_dir())
    save_experiment_config(config, experiment_dir / "config_snapshot.json")

    start_epoch = 0
    best_miou = float("-inf")
    history: list[dict[str, float]] = []

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state.get("epoch", 0)) + 1
        previous_metrics = state.get("metrics", {})
        best_miou = float(previous_metrics.get("valid_miou", best_miou))

    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    for epoch in range(start_epoch, config.training.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_classes=config.dataset.num_classes,
            ignore_index=config.dataset.ignore_index,
            amp=config.training.amp,
            scaler=scaler,
            max_grad_norm=config.training.max_grad_norm,
            epoch_index=epoch,
            epochs=config.training.epochs,
        )
        valid_metrics = evaluate(
            model,
            valid_loader,
            criterion,
            device,
            num_classes=config.dataset.num_classes,
            ignore_index=config.dataset.ignore_index,
            amp=config.training.amp,
            description=f"valid {epoch + 1}/{config.training.epochs}",
        )
        scheduler.step()

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(train_metrics["loss"]),
            "train_miou": float(train_metrics["miou"]),
            "valid_loss": float(valid_metrics["loss"]),
            "valid_miou": float(valid_metrics["miou"]),
            "valid_pixel_accuracy": float(valid_metrics["pixel_accuracy"]),
            "valid_mean_accuracy": float(valid_metrics["mean_accuracy"]),
        }
        history.append(epoch_row)
        write_history_csv(report_dir / "history.csv", history)

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=epoch_row,
            config=config.to_dict(),
        )
        if epoch_row["valid_miou"] >= best_miou:
            best_miou = epoch_row["valid_miou"]
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=epoch_row,
                config=config.to_dict(),
            )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={epoch_row['train_loss']:.4f} "
            f"train_miou={epoch_row['train_miou']:.4f} "
            f"valid_loss={epoch_row['valid_loss']:.4f} "
            f"valid_miou={epoch_row['valid_miou']:.4f}"
        )

    best_epoch = max(history, key=lambda row: row["valid_miou"]) if history else None
    summary = {
        "experiment_name": config.experiment_name,
        "device": str(device),
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "best_epoch": best_epoch,
    }
    write_json(report_dir / "summary.json", summary)
    print(f"Training finished. Outputs saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
