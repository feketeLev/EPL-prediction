from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from betproj.ml_dataset import build_bet_level_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/processed/closing_features_calibrated_test_2013_2015.parquet",
    )
    ap.add_argument(
        "--output",
        default="data/processed/bet_level_ml_dataset_2013_2015.parquet",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    bets = build_bet_level_dataset(df)
    bets.to_parquet(output_path, index=False)

    print("Saved:", output_path)
    print("Shape:", bets.shape)
    print(bets.head())


if __name__ == "__main__":
    main()