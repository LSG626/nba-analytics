"""Train chronological NBA prediction models from local processed or raw data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ml.training import (
    DEFAULT_DATA_DIRS,
    EVALUATION_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    train_models,
)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        action="append",
        type=Path,
        dest="data_directories",
        help="Data directory, repeatable; defaults to processed, raw, then supplied Raw/.",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=PREDICTION_DIR)
    args = parser.parse_args()
    result = train_models(
        args.data_directories or DEFAULT_DATA_DIRS,
        model_dir=args.model_dir,
        evaluation_dir=args.evaluation_dir,
        prediction_dir=args.prediction_dir,
    )
    print(
        "Training complete:",
        result["winner"]["selected_by_validation_log_loss"],
        "winner model;",
        result["scores"]["selected_models"],
        "score models;",
        result["players"].get("available", False),
        "player models available.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
