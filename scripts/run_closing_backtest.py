from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from betproj.backtest import select_bets, threshold_grid_backtest, evaluation_tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/closing_features.parquet")
    ap.add_argument("--mode", default="best_per_match", choices=["best_per_match", "all"])
    ap.add_argument("--ev-threshold", type=float, default=0.0)
    ap.add_argument("--min-n-odds", type=int, default=1)
    ap.add_argument("--max-odds", type=float, default=None)
    ap.add_argument("--min-odds", type=float, default=None)
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path("results/closing_backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    bets = select_bets(
        df=df,
        ev_threshold=args.ev_threshold,
        mode=args.mode,
        min_n_odds=args.min_n_odds,
        max_odds_allowed=args.max_odds,
        min_odds_allowed=args.min_odds,
)

    tables = evaluation_tables(bets)

    thresholds = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    grid = threshold_grid_backtest(
        df=df,
        thresholds=thresholds,
        mode=args.mode,
        min_n_odds=args.min_n_odds,
        max_odds_allowed=args.max_odds,
        min_odds_allowed=args.min_odds,
    )

    bets.to_parquet(out_dir / "selected_bets.parquet", index=False)
    tables["summary"].to_csv(out_dir / "summary.csv", index=False)
    tables["by_year"].to_csv(out_dir / "by_year.csv", index=False)
    tables["by_league"].to_csv(out_dir / "by_league.csv", index=False)
    tables["by_outcome"].to_csv(out_dir / "by_outcome.csv", index=False)
    tables["by_bookie"].to_csv(out_dir / "by_bookie.csv", index=False)
    tables["by_n_odds_bin"].to_csv(out_dir / "by_n_odds_bin.csv", index=False)
    grid.to_csv(out_dir / "threshold_grid.csv", index=False)

    print("\nSUMMARY")
    print(tables["summary"].to_string(index=False))

    print("\nTHRESHOLD GRID")
    print(grid.to_string(index=False))

    print("\nTOP 15 LEAGUES BY ROI")
    print(tables["by_league"].head(15).to_string(index=False))

    print("\nTOP 15 BOOKIES BY ROI")
    print(tables["by_bookie"].head(15).to_string(index=False))

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()