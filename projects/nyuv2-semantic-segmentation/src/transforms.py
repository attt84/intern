from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as tf
from torchvision.transforms import ColorJitter, InterpolationMode


def _to_image_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div(255.0)


def _to_depth_tensor(depth: np.ndarray | None, height: int, width: int) -> torch.Tensor:
    if depth is None:
        depth_tensor = torch.zeros((1, height, width), dtype=torch.float32)
    else:
        depth_array = np.ascontiguousarray(depth.astype(np.float32))
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)
        max_value = float(depth_tensor.max().item()) if depth_tensor.numel() > 0 else 0.0
        if max_value > 0:
            depth_tensor = depth_tensor / max_value
    return depth_tensor


def _to_mask_tensor(mask: np.ndarray | None) -> torch.Tensor | None:
    if mask is None:
        return None
    return torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64)))


class SegmentationTransform:
    """Joint transform for RGB image, depth map, and semantic mask."""

    def __init__(
        self,
        image_size: tuple[int, int],
        *,
        train: bool,
        rgb_mean: tuple[float, float, float],
        rgb_std: tuple[float, float, float],
        depth_mean: float,
        depth_std: float,
        hflip_prob: float = 0.5,
        scale_range: tuple[float, float] = (1.0, 1.25),
        color_jitter_prob: float = 0.8,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.hflip_prob = hflip_prob
        self.scale_range = scale_range
        self.color_jitter_prob = color_jitter_prob
        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.02,
        )

    def _resize(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor | None,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        image = tf.resize(image, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        depth = tf.resize(depth, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        if mask is not None:
            mask = tf.resize(mask.unsqueeze(0).float(), size, interpolation=InterpolationMode.NEAREST).squeeze(0).long()
        return image, depth, mask

    def _random_crop(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        target_height, target_width = self.image_size
        _, height, width = image.shape
        if height == target_height and width == target_width:
            return image, depth, mask
        max_top = max(height - target_height, 0)
        max_left = max(width - target_width, 0)
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0
        image = tf.crop(image, top, left, target_height, target_width)
        depth = tf.crop(depth, top, left, target_height, target_width)
        if mask is not None:
            mask = tf.crop(mask, top, left, target_height, target_width)
        return image, depth, mask

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = sample["image"]
        depth = sample.get("depth")
        mask = sample.get("mask")

        image_tensor = _to_image_tensor(image)
        depth_tensor = _to_depth_tensor(depth, image.shape[0], image.shape[1])
        mask_tensor = _to_mask_tensor(mask)

        if self.train:
            scale = random.uniform(*self.scale_range)
            scaled_height = max(int(self.image_size[0] * scale), self.image_size[0])
            scaled_width = max(int(self.image_size[1] * scale), self.image_size[1])
            resize_size = (scaled_height, scaled_width)
        else:
            resize_size = self.image_size

        image_tensor, depth_tensor, mask_tensor = self._resize(
            image_tensor,
            depth_tensor,
            mask_tensor,
            resize_size,
        )

        if self.train:
            image_tensor, depth_tensor, mask_tensor = self._random_crop(
                image_tensor,
                depth_tensor,
                mask_tensor,
            )
            if random.random() < self.hflip_prob:
                image_tensor = tf.hflip(image_tensor)
                depth_tensor = tf.hflip(depth_tensor)
                if mask_tensor is not None:
                    mask_tensor = tf.hflip(mask_tensor)
            if random.random() < self.color_jitter_prob:
                image_tensor = self.color_jitter(image_tensor)

        image_tensor = tf.normalize(image_tensor, mean=list(self.rgb_mean), std=list(self.rgb_std))
        depth_tensor = (depth_tensor - self.depth_mean) / max(self.depth_std, 1e-6)

        return {
            "sample_id": sample["sample_id"],
            "image": image_tensor,
            "depth": depth_tensor,
            "mask": mask_tensor,
        }


def build_transforms(
    image_size: tuple[int, int],
    *,
    train: bool,
    rgb_mean: tuple[float, float, float],
    rgb_std: tuple[float, float, float],
    depth_mean: float,
    depth_std: float,
) -> SegmentationTransform:
    return SegmentationTransform(
        image_size=image_size,
        train=train,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
        depth_mean=depth_mean,
        depth_std=depth_std,
    )
