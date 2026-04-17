from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import load_experiment_config
from src.dataset import NYUv2SegmentationDataset
from src.engine import evaluate
from src.model import RGBDUNet
from src.transforms import build_transforms
from src.utils import load_checkpoint, resolve_device, save_prediction_panel, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NYUv2 semantic segmentation model.")
    parser.add_argument("--config", required=True, help="Path to the experiment config JSON.")
    parser.add_argument("--checkpoint", help="Checkpoint path. Defaults to outputs/<experiment>/checkpoints/best.pt.")
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"), help="Split to evaluate.")
    parser.add_argument("--device", help="Device override. Example: cuda, cpu, mps.")
    parser.add_argument("--num-workers", type=int, help="Optional DataLoader worker override.")
    parser.add_argument("--batch-size", type=int, help="Optional batch size override.")
    parser.add_argument("--num-visualizations", type=int, default=4, help="How many prediction panels to save.")
    return parser.parse_args()


def split_name(config, split: str) -> str:
    if split == "train":
        return config.dataset.train_split
    if split == "valid":
        return config.dataset.valid_split
    return config.dataset.test_split


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = resolve_device(args.device)
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    eval_transform = build_transforms(
        config.dataset.image_size,
        train=False,
        rgb_mean=config.dataset.rgb_mean,
        rgb_std=config.dataset.rgb_std,
        depth_mean=config.dataset.depth_mean,
        depth_std=config.dataset.depth_std,
    )
    dataset = NYUv2SegmentationDataset(
        config.dataset.data_dir,
        split_name(config, args.split),
        transform=eval_transform,
        require_depth=config.model.use_depth,
        require_labels=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = RGBDUNet(
        num_classes=config.dataset.num_classes,
        base_channels=config.model.base_channels,
        use_depth=config.model.use_depth,
        dropout=config.model.dropout,
    ).to(device)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else config.checkpoint_dir() / "best.pt"
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)

    criterion = nn.CrossEntropyLoss(ignore_index=config.dataset.ignore_index)
    metrics = evaluate(
        model,
        loader,
        criterion,
        device,
        num_classes=config.dataset.num_classes,
        ignore_index=config.dataset.ignore_index,
        amp=config.training.amp,
        description=f"{args.split} evaluation",
    )

    report_dir = config.report_dir()
    prediction_dir = config.prediction_dir() / args.split
    report_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint_path),
        "loaded_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "metrics": metrics,
    }
    write_json(report_dir / f"{args.split}_metrics.json", payload)

    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            depth = batch["depth"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images, depth)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1)
            for index in range(images.shape[0]):
                sample_id = batch["sample_id"][index]
                save_prediction_panel(
                    prediction_dir / f"{sample_id}.png",
                    image=batch["image"][index],
                    pred_mask=preds[index].cpu(),
                    target_mask=batch["mask"][index],
                    num_classes=config.dataset.num_classes,
                    rgb_mean=config.dataset.rgb_mean,
                    rgb_std=config.dataset.rgb_std,
                )
                saved += 1
                if saved >= args.num_visualizations:
                    break
            if saved >= args.num_visualizations:
                break

    print(f"Evaluation complete. Metrics saved to: {report_dir / f'{args.split}_metrics.json'}")


if __name__ == "__main__":
    main()
