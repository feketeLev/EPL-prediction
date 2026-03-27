from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from betproj.backtest import summarize_bets, summarize_by_group
from betproj.ml_dataset import default_feature_columns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/processed/bet_level_ml_dataset_2013_2015.parquet",
    )
    ap.add_argument("--train-start-year", type=int, default=2013)
    ap.add_argument("--train-end-year", type=int, default=2014)
    ap.add_argument("--test-start-year", type=int, default=2015)
    ap.add_argument("--test-end-year", type=int, default=2015)
    ap.add_argument("--ev-threshold", type=float, default=0.05)
    ap.add_argument("--min-n-odds", type=int, default=10)
    ap.add_argument("--max-odds", type=float, default=4.0)
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path("results/ml_logreg")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    train_mask = df["year"].between(args.train_start_year, args.train_end_year, inclusive="both")
    test_mask = df["year"].between(args.test_start_year, args.test_end_year, inclusive="both")

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    feature_cols = default_feature_columns()
    numeric_features = ["p_raw", "p_cal", "odds", "avg_odds", "n_odds", "gap", "rel_gap", "month"]
    categorical_features = ["bet_outcome", "top_bookie", "league"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    X_train = train_df[feature_cols]
    y_train = train_df["won"]

    X_test = test_df[feature_cols]

    model.fit(X_train, y_train)

    test_df["p_model"] = model.predict_proba(X_test)[:, 1]
    test_df["ev_model"] = test_df["p_model"] * test_df["odds"] - 1.0
    test_df["ev"] = test_df["ev_model"]

    selected = test_df.loc[test_df["ev_model"] > args.ev_threshold].copy()
    selected = selected.loc[selected["n_odds"] >= args.min_n_odds].copy()
    selected = selected.loc[selected["odds"] <= args.max_odds].copy()

    selected = selected.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    selected["cum_profit"] = selected["profit"].cumsum()
    selected["cum_staked"] = selected["stake"].cumsum()
    selected["cum_roi"] = selected["cum_profit"] / selected["cum_staked"]

    summary = summarize_bets(selected)
    by_year = summarize_by_group(selected, "year")
    by_bookie = summarize_by_group(selected, "top_bookie")
    by_outcome = summarize_by_group(selected, "bet_outcome")

    selected.to_parquet(out_dir / "selected_bets.parquet", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    by_year.to_csv(out_dir / "by_year.csv", index=False)
    by_bookie.to_csv(out_dir / "by_bookie.csv", index=False)
    by_outcome.to_csv(out_dir / "by_outcome.csv", index=False)

    print("\nSUMMARY")
    print(summary.to_string(index=False))

    print("\nBY YEAR")
    print(by_year.to_string(index=False))

    print("\nSaved outputs to:", out_dir.resolve())


if __name__ == "__main__":
    main()