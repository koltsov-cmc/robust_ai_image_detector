from __future__ import annotations

import argparse

from aigc_detector.experiments import experiment_names
from aigc_detector.inference import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one built-in EVA02-CLIP-B/16 GAP experiment on one GPU in strict FP32."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=experiment_names(),
        help="Loads runs/<experiment>/best.pt and writes predictions/<experiment>.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_inference(arguments.experiment)
