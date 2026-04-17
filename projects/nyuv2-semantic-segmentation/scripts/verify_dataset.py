from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import load_experiment_config
from src.dataset import collect_samples, describe_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify NYUv2 folder layout before training.")
    parser.add_argument("--config", required=True, help="Path to the experiment config JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    message = describe_layout(
        config.dataset.data_dir,
        splits=(
            config.dataset.train_split,
            config.dataset.valid_split,
            config.dataset.test_split,
        ),
        require_depth=config.model.use_depth,
        require_labels=True,
    )
    print(message)

    for split in (config.dataset.train_split, config.dataset.valid_split, config.dataset.test_split):
        samples = collect_samples(
            config.dataset.data_dir,
            split,
            require_depth=config.model.use_depth,
            require_labels=True,
        )
        print(f"{split}: {len(samples)} paired samples")


if __name__ == "__main__":
    main()
