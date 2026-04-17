from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected a pair of values, got: {value!r}")


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if is_dataclass(value):
        return _serialize(asdict(value))
    return value


@dataclass
class DatasetConfig:
    data_dir: Path
    image_size: tuple[int, int] = (320, 320)
    train_split: str = "train"
    valid_split: str = "valid"
    test_split: str = "test"
    num_classes: int = 41
    ignore_index: int = 255
    rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    depth_mean: float = 0.5
    depth_std: float = 0.25

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetConfig":
        return cls(
            data_dir=Path(payload["data_dir"]),
            image_size=_pair(payload.get("image_size", (320, 320))),
            train_split=payload.get("train_split", "train"),
            valid_split=payload.get("valid_split", "valid"),
            test_split=payload.get("test_split", "test"),
            num_classes=int(payload.get("num_classes", 41)),
            ignore_index=int(payload.get("ignore_index", 255)),
            rgb_mean=tuple(float(value) for value in payload.get("rgb_mean", (0.485, 0.456, 0.406))),
            rgb_std=tuple(float(value) for value in payload.get("rgb_std", (0.229, 0.224, 0.225))),
            depth_mean=float(payload.get("depth_mean", 0.5)),
            depth_std=float(payload.get("depth_std", 0.25)),
        )

    def resolve_relative_paths(self, base_dir: Path) -> None:
        if not self.data_dir.is_absolute():
            self.data_dir = (base_dir / self.data_dir).resolve()


@dataclass
class ModelConfig:
    base_channels: int = 32
    dropout: float = 0.1
    use_depth: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(
            base_channels=int(payload.get("base_channels", 32)),
            dropout=float(payload.get("dropout", 0.1)),
            use_depth=bool(payload.get("use_depth", True)),
        )


@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    amp: bool = True
    max_grad_norm: float | None = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        return cls(
            batch_size=int(payload.get("batch_size", 4)),
            num_workers=int(payload.get("num_workers", 4)),
            epochs=int(payload.get("epochs", 20)),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            seed=int(payload.get("seed", 42)),
            amp=bool(payload.get("amp", True)),
            max_grad_norm=None if payload.get("max_grad_norm") is None else float(payload["max_grad_norm"]),
        )


@dataclass
class OutputConfig:
    root_dir: Path = Path("outputs")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OutputConfig":
        return cls(root_dir=Path(payload.get("root_dir", "outputs")))

    def resolve_relative_paths(self, base_dir: Path) -> None:
        if not self.root_dir.is_absolute():
            self.root_dir = (base_dir / self.root_dir).resolve()


@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            experiment_name=payload["experiment_name"],
            dataset=DatasetConfig.from_dict(payload["dataset"]),
            model=ModelConfig.from_dict(payload["model"]),
            training=TrainingConfig.from_dict(payload["training"]),
            output=OutputConfig.from_dict(payload.get("output", {})),
        )

    def resolve_relative_paths(self, base_dir: Path) -> "ExperimentConfig":
        self.dataset.resolve_relative_paths(base_dir)
        self.output.resolve_relative_paths(base_dir)
        return self

    def experiment_dir(self) -> Path:
        return self.output.root_dir / self.experiment_name

    def checkpoint_dir(self) -> Path:
        return self.experiment_dir() / "checkpoints"

    def report_dir(self) -> Path:
        return self.experiment_dir() / "reports"

    def prediction_dir(self) -> Path:
        return self.experiment_dir() / "predictions"

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    project_root = config_path.parent.parent
    return ExperimentConfig.from_dict(payload).resolve_relative_paths(project_root)


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, ensure_ascii=False, indent=2)
