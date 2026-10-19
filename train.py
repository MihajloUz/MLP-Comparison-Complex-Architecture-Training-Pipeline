"""Command line entry point.

Prompts the user for the path to a CSV dataset, then runs the full
MLP comparison pipeline. The dataset is *always* supplied by the user; nothing
is downloaded and no sample data is invented here.
"""

from __future__ import annotations

import argparse
import sys
import warnings

from mlp_pipeline import Config, run

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP comparison training pipeline")
    parser.add_argument("--csv", default=None, help="Path to the CSV dataset (skips the prompt).")
    parser.add_argument("--target", default=None, help="Optional target column name.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for artefacts.")
    parser.add_argument("--epochs", type=int, default=60, help="Max epochs for deep models.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--max-rows", type=int, default=20000, help="Subsample larger CSVs.")
    parser.add_argument("--no-keras", action="store_true", help="Train only the sklearn MLP.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    path = args.csv
    if not path:
        path = input("Enter path to CSV: ").strip()
    if not path:
        print("No CSV path provided. Aborting.", file=sys.stderr)
        return 2

    cfg = Config(
        seed=args.seed,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        target_column=args.target,
    )

    try:
        run(path, cfg, use_keras=not args.no_keras)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError) as exc:
        print(f"Error while processing the dataset: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
