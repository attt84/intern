from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name is not None:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_history_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def denormalize_image(
    image: torch.Tensor,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    image = image.detach().cpu()
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    image = image * std_tensor + mean_tensor
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def build_palette(num_classes: int) -> np.ndarray:
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for index in range(num_classes):
        palette[index] = np.array(
            [
                (37 * index) % 255,
                (67 * index) % 255,
                (97 * index) % 255,
            ],
            dtype=np.uint8,
        )
    return palette


def colorize_mask(mask: torch.Tensor | np.ndarray, *, num_classes: int) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask_array = mask.detach().cpu().numpy()
    else:
        mask_array = mask
    palette = build_palette(num_classes)
    clipped = np.clip(mask_array.astype(np.int64), 0, num_classes - 1)
    return palette[clipped]


def save_prediction_panel(
    output_path: str | Path,
    *,
    image: torch.Tensor,
    pred_mask: torch.Tensor,
    num_classes: int,
    rgb_mean: tuple[float, float, float],
    rgb_std: tuple[float, float, float],
    target_mask: torch.Tensor | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_np = denormalize_image(image, mean=rgb_mean, std=rgb_std)
    pred_np = colorize_mask(pred_mask, num_classes=num_classes)

    columns = 3 if target_mask is not None else 2
    figure, axes = plt.subplots(1, columns, figsize=(5 * columns, 5))
    if columns == 1:
        axes = np.array([axes])
    else:
        axes = np.asarray(axes)

    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    if target_mask is not None:
        target_np = colorize_mask(target_mask, num_classes=num_classes)
        axes[1].imshow(target_np)
        axes[1].set_title("Target")
        axes[1].axis("off")
        axes[2].imshow(pred_np)
        axes[2].set_title("Prediction")
        axes[2].axis("off")
    else:
        axes[1].imshow(pred_np)
        axes[1].set_title("Prediction")
        axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
