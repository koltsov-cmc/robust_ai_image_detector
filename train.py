from __future__ import annotations

import argparse

from aigc_detector.experiments import experiment_names
from aigc_detector.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one built-in EVA02-CLIP-B/16 GAP experiment on one GPU in strict FP32."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=experiment_names(),
        help="Built-in experiment name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_training(arguments.experiment)
