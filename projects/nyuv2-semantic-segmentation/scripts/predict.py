from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import load_experiment_config
from src.dataset import load_depth_map, load_rgb_image
from src.model import RGBDUNet
from src.transforms import build_transforms
from src.utils import load_checkpoint, resolve_device, save_prediction_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image inference for NYUv2 segmentation.")
    parser.add_argument("--config", required=True, help="Path to the experiment config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--image", required=True, help="Path to the RGB image.")
    parser.add_argument("--depth", help="Path to the depth map. Required when the config uses depth.")
    parser.add_argument("--output", required=True, help="Output path for the prediction panel.")
    parser.add_argument("--device", help="Device override. Example: cuda, cpu, mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = resolve_device(args.device)

    if config.model.use_depth and not args.depth:
        raise ValueError("This config expects a depth file. Pass --depth or switch to rgb_only_unet.json.")

    model = RGBDUNet(
        num_classes=config.dataset.num_classes,
        base_channels=config.model.base_channels,
        use_depth=config.model.use_depth,
        dropout=config.model.dropout,
    ).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    transform = build_transforms(
        config.dataset.image_size,
        train=False,
        rgb_mean=config.dataset.rgb_mean,
        rgb_std=config.dataset.rgb_std,
        depth_mean=config.dataset.depth_mean,
        depth_std=config.dataset.depth_std,
    )
    sample = transform(
        {
            "sample_id": Path(args.image).stem,
            "image": load_rgb_image(args.image),
            "depth": load_depth_map(args.depth) if args.depth else None,
            "mask": None,
        }
    )

    with torch.no_grad():
        image = sample["image"].unsqueeze(0).to(device)
        depth = sample["depth"].unsqueeze(0).to(device)
        logits = model(image, depth)
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu()

    save_prediction_panel(
        args.output,
        image=sample["image"],
        pred_mask=pred_mask,
        target_mask=None,
        num_classes=config.dataset.num_classes,
        rgb_mean=config.dataset.rgb_mean,
        rgb_std=config.dataset.rgb_std,
    )
    print(f"Inference panel saved to: {args.output}")


if __name__ == "__main__":
    main()
