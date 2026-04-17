from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NYUv2Paths:
    root: Path
    split: str

    @property
    def image_dir(self) -> Path:
        return self.root / self.split / "image"

    @property
    def depth_dir(self) -> Path:
        return self.root / self.split / "depth"

    @property
    def label_dir(self) -> Path:
        return self.root / self.split / "label"


def expected_layout(root: str | Path) -> dict[str, NYUv2Paths]:
    """Return the directory layout expected by the public portfolio notebook."""
    root_path = Path(root)
    return {split: NYUv2Paths(root=root_path, split=split) for split in ("train", "valid", "test")}


def list_samples(root: str | Path, split: str = "train") -> list[str]:
    """List sample file names from the image directory for a given split."""
    paths = NYUv2Paths(root=Path(root), split=split)
    if not paths.image_dir.exists():
        return []
    return sorted(path.name for path in paths.image_dir.iterdir() if path.is_file())


def validate_layout(root: str | Path) -> list[str]:
    """Return human-readable warnings for missing directories."""
    warnings: list[str] = []
    for split, paths in expected_layout(root).items():
        for name, directory in (
            (f"{split}/image", paths.image_dir),
            (f"{split}/depth", paths.depth_dir),
            (f"{split}/label", paths.label_dir),
        ):
            if not directory.exists():
                warnings.append(f"Missing directory: {name}")
    return warnings


def describe_layout(root: str | Path) -> str:
    """Create a short text summary used by the notebook."""
    warnings = validate_layout(root)
    if warnings:
        return "\n".join(warnings)
    sample_count = len(list_samples(root, "train"))
    return f"NYUv2 layout looks ready. Train image files: {sample_count}"
