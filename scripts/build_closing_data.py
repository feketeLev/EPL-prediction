from pathlib import Path

from betproj.preprocess_closing import preprocess_closing_odds
from betproj.features_closing import build_closing_features


def main():
    df = preprocess_closing_odds()
    df = build_closing_features(df)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "closing_features.parquet"
    df.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()