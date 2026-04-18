from __future__ import annotations

import argparse
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


OFFICIAL_LABELED_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NYUv2-style data for the segmentation project.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("demo", "download-official", "extract-official"),
        help="Dataset preparation mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/nyuv2",
        help="Output directory for the extracted or generated dataset.",
    )
    parser.add_argument(
        "--download-dir",
        default="data/downloads",
        help="Directory used to store downloaded files.",
    )
    parser.add_argument(
        "--mat-path",
        help="Path to nyu_depth_v2_labeled.mat. Optional for download mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to extract from the official labeled dataset.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=12,
        help="Demo train sample count.",
    )
    parser.add_argument(
        "--valid-count",
        type=int,
        default=4,
        help="Demo valid sample count.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=4,
        help="Demo test sample count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for demo data generation and official subset shuffling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sample files with the same names.",
    )
    return parser.parse_args()


def ensure_layout(root: Path) -> None:
    for split in ("train", "valid", "test"):
        for name in ("image", "depth", "label"):
            (root / split / name).mkdir(parents=True, exist_ok=True)


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _write_depth(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def _synthetic_scene(index: int, *, height: int = 240, width: int = 320, num_classes: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_coords, x_coords = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.clip((40 + 180 * x_coords + 20 * math.sin(index)), 0, 255).astype(np.uint8)
    image[..., 1] = np.clip((30 + 160 * y_coords + 10 * math.cos(index * 0.7)), 0, 255).astype(np.uint8)
    image[..., 2] = np.clip((80 + 80 * (1.0 - y_coords) + (index * 13) % 70), 0, 255).astype(np.uint8)

    depth = 0.3 + 1.5 * y_coords + 0.15 * np.sin((index + 1) * x_coords * np.pi)
    depth = depth.astype(np.float32)

    label = np.zeros((height, width), dtype=np.uint8)
    label[y_coords < 0.22] = 1
    label[(y_coords >= 0.22) & (y_coords < 0.74)] = 2
    label[y_coords >= 0.74] = 3

    left = 20 + (index * 17) % 120
    right = min(left + 55, width - 1)
    top = 95
    bottom = min(top + 90, height - 1)
    label[top:bottom, left:right] = 4
    image[top:bottom, left:right, :] = np.array([170, 110, 60], dtype=np.uint8)
    depth[top:bottom, left:right] -= 0.18

    circle_x = 210 + (index * 9) % 50
    circle_y = 150 + (index * 7) % 35
    radius = 22
    circle_mask = (x_coords * width - circle_x) ** 2 + (y_coords * height - circle_y) ** 2 <= radius ** 2
    label[circle_mask] = 5 + (index % max(num_classes - 5, 1))
    image[circle_mask] = np.array([220, 60, 90], dtype=np.uint8)
    depth[circle_mask] -= 0.1

    return image, depth, label


def create_demo_dataset(
    output_dir: Path,
    *,
    train_count: int,
    valid_count: int,
    test_count: int,
    overwrite: bool,
) -> None:
    ensure_layout(output_dir)
    spec = {
        "train": train_count,
        "valid": valid_count,
        "test": test_count,
    }
    offset = 0
    for split, count in spec.items():
        for index in range(count):
            image, depth, label = _synthetic_scene(offset + index)
            sample_id = f"{offset + index:04d}"
            image_path = output_dir / split / "image" / f"{sample_id}.png"
            depth_path = output_dir / split / "depth" / f"{sample_id}.npy"
            label_path = output_dir / split / "label" / f"{sample_id}.png"
            if not overwrite and image_path.exists():
                continue
            _write_png(image_path, image)
            _write_depth(depth_path, depth)
            _write_png(label_path, label)
        offset += count
    print(f"Demo dataset is ready at: {output_dir}")


def download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            print(f"Downloaded {block_count * block_size / 1024 / 1024:.1f} MB", end="\r")
            return
        downloaded = min(block_count * block_size, total_size)
        percent = downloaded / total_size * 100.0
        print(f"Downloading: {percent:5.1f}% ({downloaded / 1024 / 1024:.1f} / {total_size / 1024 / 1024:.1f} MB)", end="\r")

    urllib.request.urlretrieve(url, destination, reporthook=report)
    print()
    print(f"Downloaded official labeled dataset to: {destination}")


def _normalize_images(array: np.ndarray) -> np.ndarray:
    candidate_shapes = []
    arr = np.asarray(array)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D images array, got shape {arr.shape}")
    permutations = [
        (0, 1, 2, 3),
        (0, 1, 3, 2),
        (0, 2, 3, 1),
        (0, 3, 1, 2),
        (1, 0, 2, 3),
        (1, 2, 3, 0),
        (1, 3, 0, 2),
        (2, 0, 1, 3),
        (2, 1, 3, 0),
        (2, 3, 0, 1),
        (3, 0, 1, 2),
        (3, 1, 2, 0),
        (3, 2, 0, 1),
    ]
    for perm in permutations:
        permuted = np.transpose(arr, perm)
        if permuted.ndim == 4 and permuted.shape[-1] == 3 and permuted.shape[1] > 10 and permuted.shape[2] > 10:
            candidate_shapes.append(permuted)
    if not candidate_shapes:
        raise ValueError(f"Could not normalize images array with shape {arr.shape}")
    best = min(candidate_shapes, key=lambda item: item.shape[0])
    return np.asarray(best)


def _normalize_volume(array: np.ndarray, sample_count: int) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")
    for axis in range(3):
        if arr.shape[axis] == sample_count:
            if axis == 0:
                return arr
            return np.moveaxis(arr, axis, 0)
    if arr.shape[-1] == sample_count:
        return np.moveaxis(arr, -1, 0)
    raise ValueError(f"Could not find sample axis in shape {arr.shape} for sample_count={sample_count}")


def _load_official_arrays(mat_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import h5py
    except ImportError:
        h5py = None
    try:
        from scipy import io as scipy_io
    except ImportError:
        scipy_io = None

    if h5py is not None and h5py.is_hdf5(mat_path):
        with h5py.File(mat_path, "r") as handle:
            images = np.array(handle["images"])
            depths = np.array(handle["depths"])
            labels = np.array(handle["labels"])
    elif scipy_io is not None:
        payload = scipy_io.loadmat(mat_path)
        images = np.asarray(payload["images"])
        depths = np.asarray(payload["depths"])
        labels = np.asarray(payload["labels"])
    else:
        raise RuntimeError("Either h5py or scipy is required to read nyu_depth_v2_labeled.mat")

    images = _normalize_images(images)
    sample_count = images.shape[0]
    depths = _normalize_volume(depths, sample_count)
    labels = _normalize_volume(labels, sample_count)
    return images, depths, labels


def extract_official_dataset(
    mat_path: Path,
    output_dir: Path,
    *,
    limit: int | None,
    seed: int,
    overwrite: bool,
) -> None:
    ensure_layout(output_dir)
    images, depths, labels = _load_official_arrays(mat_path)
    total = images.shape[0]
    indices = np.arange(total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if limit is not None:
        indices = indices[:limit]

    split_counts = {
        "train": int(len(indices) * 0.7),
        "valid": int(len(indices) * 0.15),
    }
    split_counts["test"] = len(indices) - split_counts["train"] - split_counts["valid"]

    cursor = 0
    for split in ("train", "valid", "test"):
        count = split_counts[split]
        split_indices = indices[cursor : cursor + count]
        cursor += count
        for local_index, sample_index in enumerate(split_indices):
            sample_id = f"{int(sample_index):04d}"
            image_path = output_dir / split / "image" / f"{sample_id}.png"
            depth_path = output_dir / split / "depth" / f"{sample_id}.npy"
            label_path = output_dir / split / "label" / f"{sample_id}.png"
            if not overwrite and image_path.exists():
                continue

            image = np.asarray(images[sample_index], dtype=np.uint8)
            depth = np.asarray(depths[sample_index], dtype=np.float32)
            label = np.asarray(labels[sample_index], dtype=np.int64)
            if label.min() >= 1:
                label = label - 1
            label = np.clip(label, 0, 255).astype(np.uint8)

            _write_png(image_path, image)
            _write_depth(depth_path, depth)
            _write_png(label_path, label)
        print(f"{split}: wrote {count} samples")
    print(f"Official NYUv2 subset is ready at: {output_dir}")


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    download_dir = (PROJECT_ROOT / args.download_dir).resolve()

    if args.mode == "demo":
        create_demo_dataset(
            output_dir,
            train_count=args.train_count,
            valid_count=args.valid_count,
            test_count=args.test_count,
            overwrite=args.overwrite,
        )
        return

    if args.mode == "download-official":
        mat_path = (download_dir / "nyu_depth_v2_labeled.mat").resolve()
        download_with_progress(OFFICIAL_LABELED_URL, mat_path)
        return

    mat_path = Path(args.mat_path).resolve() if args.mat_path else (download_dir / "nyu_depth_v2_labeled.mat").resolve()
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    extract_official_dataset(
        mat_path,
        output_dir,
        limit=args.limit,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
