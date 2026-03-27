from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from betproj.calibration import calibrate_by_year_split, calibration_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/closing_features.parquet")
    ap.add_argument("--train-start-year", type=int, default=2005)
    ap.add_argument("--train-end-year", type=int, default=2012)
    ap.add_argument("--test-start-year", type=int, default=2013)
    ap.add_argument("--test-end-year", type=int, default=2015)
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    df_train, df_test, _ = calibrate_by_year_split(
        df=df,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
    )

    summary = calibration_summary(df_test)

    out_name = (
        f"closing_features_calibrated_test_"
        f"{args.test_start_year}_{args.test_end_year}.parquet"
    )
    out_path = out_dir / out_name
    summary_path = out_dir / (
        f"calibration_summary_{args.test_start_year}_{args.test_end_year}.csv"
    )

    df_test.to_parquet(out_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nTRAIN SHAPE")
    print(df_train.shape)

    print("\nTEST SHAPE")
    print(df_test.shape)

    print("\nCALIBRATION SUMMARY")
    print(summary.to_string(index=False))

    print(f"\nSaved calibrated test data to: {out_path.resolve()}")
    print(f"Saved calibration summary to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()