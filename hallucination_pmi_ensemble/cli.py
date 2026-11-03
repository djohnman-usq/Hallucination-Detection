"""Command-line runner for the standalone frozen hallucination stacker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import confusion_report, evaluate_predictions
from .model import FrozenStacker
from .reporting import create_report


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = PACKAGE_ROOT / "artifacts" / "pmi_plus_length_control_precision60_stacking_model.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing the required detector feature columns.")
    parser.add_argument("--output-dir", required=True, help="Directory for predictions, metrics, confusion, and visuals.")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT), help="Frozen stacker JSON artifact.")
    parser.add_argument("--prefix", default="standalone_full_output", help="Filename prefix for generated outputs.")
    parser.add_argument("--no-visuals", action="store_true", help="Skip PNG visual generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    visuals_dir = output_dir / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input, keep_default_na=False)
    stacker = FrozenStacker.load(args.artifact)
    predictions = stacker.predict_frame(frame)

    predictions_path = output_dir / f"{args.prefix}_predictions.csv"
    metrics_path = output_dir / f"{args.prefix}_metrics.json"
    confusion_path = output_dir / f"{args.prefix}_confusion.csv"

    predictions.to_csv(predictions_path, index=False)
    metrics = evaluate_predictions(predictions)
    metrics["stacking_artifact"] = stacker.artifact.to_mapping()
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    confusion = confusion_report(predictions)
    if not confusion.empty:
        confusion.to_csv(confusion_path, index=False)
    if not args.no_visuals and metrics.get("labelled"):
        create_report(predictions, metrics, visuals_dir)

    print(json.dumps(
        {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "confusion": str(confusion_path) if confusion_path.exists() else None,
            "visuals": str(visuals_dir) if visuals_dir.exists() else None,
            "ensemble": metrics.get("ensemble"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
