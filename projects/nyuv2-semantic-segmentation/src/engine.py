from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional
from tqdm.auto import tqdm

from .metrics import SegmentationMeter


def _autocast_context(device: torch.device, enabled: bool) -> Any:
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _forward_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    depth = batch["depth"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    logits = model(images, depth)
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    return logits, masks


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    ignore_index: int,
    amp: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
    max_grad_norm: float | None = None,
    epoch_index: int = 0,
    epochs: int = 0,
) -> dict[str, float | list[float | None]]:
    model.train()
    meter = SegmentationMeter(num_classes=num_classes, ignore_index=ignore_index)
    progress = tqdm(loader, desc=f"train {epoch_index + 1}/{epochs}", leave=False)

    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            logits, masks = _forward_step(model, batch, device)
            loss = criterion(logits, masks)

        if scaler is not None and amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = int(batch["image"].shape[0])
        meter.update(preds, masks, loss=float(loss.item()), count=batch_size)

        stats = meter.summary()
        progress.set_postfix(loss=f"{stats['loss']:.4f}", miou=f"{stats['miou']:.4f}")

    return meter.summary()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    ignore_index: int,
    amp: bool,
    description: str = "eval",
) -> dict[str, float | list[float | None]]:
    model.eval()
    meter = SegmentationMeter(num_classes=num_classes, ignore_index=ignore_index)
    progress = tqdm(loader, desc=description, leave=False)

    for batch in progress:
        with _autocast_context(device, amp):
            logits, masks = _forward_step(model, batch, device)
            loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)
        batch_size = int(batch["image"].shape[0])
        meter.update(preds, masks, loss=float(loss.item()), count=batch_size)
        stats = meter.summary()
        progress.set_postfix(loss=f"{stats['loss']:.4f}", miou=f"{stats['miou']:.4f}")

    return meter.summary()
