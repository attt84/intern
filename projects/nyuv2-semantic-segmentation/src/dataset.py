from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


@dataclass(frozen=True)
class NYUv2SplitPaths:
    root: Path
    split: str

    @property
    def split_dir(self) -> Path:
        return self.root / self.split

    @property
    def image_dir(self) -> Path:
        return self.split_dir / "image"

    @property
    def depth_dir(self) -> Path:
        return self.split_dir / "depth"

    @property
    def label_dir(self) -> Path:
        return self.split_dir / "label"


@dataclass(frozen=True)
class NYUv2Sample:
    sample_id: str
    image_path: Path
    depth_path: Path | None
    label_path: Path | None


def expected_layout(root: str | Path, splits: tuple[str, ...] = ("train", "valid", "test")) -> dict[str, NYUv2SplitPaths]:
    """Return the directory layout expected by the training scripts."""
    root_path = Path(root)
    return {split: NYUv2SplitPaths(root=root_path, split=split) for split in splits}


def _index_directory(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not directory.exists():
        return index
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            index[path.stem] = path
    return index


def list_samples(root: str | Path, split: str = "train") -> list[str]:
    """List file names from the image directory for a given split."""
    image_dir = NYUv2SplitPaths(root=Path(root), split=split).image_dir
    if not image_dir.exists():
        return []
    return sorted(path.name for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def collect_samples(
    root: str | Path,
    split: str,
    *,
    require_depth: bool = True,
    require_labels: bool = True,
) -> list[NYUv2Sample]:
    """Build the paired sample list from image/depth/label directories."""
    paths = NYUv2SplitPaths(root=Path(root), split=split)
    image_index = _index_directory(paths.image_dir)
    depth_index = _index_directory(paths.depth_dir)
    label_index = _index_directory(paths.label_dir)

    samples: list[NYUv2Sample] = []
    missing: list[str] = []
    for sample_id, image_path in image_index.items():
        depth_path = depth_index.get(sample_id)
        label_path = label_index.get(sample_id)
        if require_depth and depth_path is None:
            missing.append(f"{split}/{sample_id}: depth file is missing")
            continue
        if require_labels and label_path is None:
            missing.append(f"{split}/{sample_id}: label file is missing")
            continue
        samples.append(
            NYUv2Sample(
                sample_id=sample_id,
                image_path=image_path,
                depth_path=depth_path,
                label_path=label_path,
            )
        )

    if missing:
        detail = "\n".join(missing[:10])
        if len(missing) > 10:
            detail = f"{detail}\n... and {len(missing) - 10} more"
        raise FileNotFoundError(f"Sample pairing failed for split '{split}'.\n{detail}")
    return samples


def validate_layout(
    root: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    require_depth: bool = True,
    require_labels: bool = True,
) -> list[str]:
    """Return human-readable warnings for missing directories or pairings."""
    warnings: list[str] = []
    for split, paths in expected_layout(root, splits=splits).items():
        required_dirs = [("image", paths.image_dir)]
        if require_depth:
            required_dirs.append(("depth", paths.depth_dir))
        if require_labels:
            required_dirs.append(("label", paths.label_dir))
        for name, directory in required_dirs:
            if not directory.exists():
                warnings.append(f"Missing directory: {split}/{name}")
        if warnings:
            continue
        try:
            collect_samples(root, split, require_depth=require_depth, require_labels=require_labels)
        except FileNotFoundError as error:
            warnings.append(str(error))
    return warnings


def describe_layout(
    root: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    require_depth: bool = True,
    require_labels: bool = True,
) -> str:
    """Create a short summary used by verification and setup commands."""
    warnings = validate_layout(
        root,
        splits=splits,
        require_depth=require_depth,
        require_labels=require_labels,
    )
    if warnings:
        return "\n".join(warnings)

    counts = []
    for split in splits:
        count = len(collect_samples(root, split, require_depth=require_depth, require_labels=require_labels))
        counts.append(f"{split}: {count}")
    return "NYUv2 layout looks ready. " + ", ".join(counts)


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an RGB image as HWC uint8."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_depth_map(path: str | Path) -> np.ndarray:
    """Load a depth map as HW float32."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = np.asarray(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def load_label_mask(path: str | Path) -> np.ndarray:
    """Load a semantic label mask as HW int64."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        label = np.load(path)
    else:
        label = np.asarray(Image.open(path))
    if label.ndim == 3:
        label = label[..., 0]
    return label.astype(np.int64)


class NYUv2SegmentationDataset(Dataset[dict[str, Any]]):
    """Dataset for the self-contained NYUv2 segmentation project."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        transform: Any | None = None,
        require_depth: bool = True,
        require_labels: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.require_depth = require_depth
        self.require_labels = require_labels
        self.samples = collect_samples(
            self.root,
            split,
            require_depth=require_depth,
            require_labels=require_labels,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        payload: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "image": load_rgb_image(sample.image_path),
            "depth": load_depth_map(sample.depth_path) if sample.depth_path is not None else None,
            "mask": load_label_mask(sample.label_path) if sample.label_path is not None else None,
        }
        if self.transform is not None:
            payload = self.transform(payload)
        return payload
