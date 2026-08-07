#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
for path in (PROJECT_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_maxim.step2_ce.data import YtTableSource  # noqa: E402
from code_maxim.step2_ce.training import add_training_arguments, run_training  # noqa: E402
from code_maxim.step3_fps.model import sampled_softmax_loss  # noqa: E402

SOLUTION_NAME = "step3_fps"
DEFAULT_TABLE = "//home/bm/users/argus/mla/steps/step3_fps/train_100m"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train sampled softmax with 511 in-batch negatives per query."
        )
    )
    parser.add_argument("--data-table", default=DEFAULT_TABLE)
    parser.add_argument("--proxy", default="kolmogorov")
    add_training_arguments(parser)
    parser.set_defaults(
        artifact_dir=str(Path.home() / "mla/data/artifacts/step3_fps"),
        batch_size=512,
        val_table="//home/bm/users/argus/mla/steps/step3_fps/val_clicks_2k",
        underdeep_run_name="step3-fps-train-100m-1epoch",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.batch_size != 512:
        raise ValueError("step3_fps uses a fixed batch size of 512")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_training(
        source=YtTableSource(args.data_table, args.proxy),
        args=args,
        loss_function=sampled_softmax_loss,
        solution_name=SOLUTION_NAME,
        training_mode="sampled_softmax",
        validation_mode="sampled_softmax",
    )


if __name__ == "__main__":
    main()
