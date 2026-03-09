from pathlib import Path
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            return pd.read_csv(path, compression="gzip", encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode file: {path}")


def load_closing_odds() -> pd.DataFrame:
    print("RAW_DIR:", RAW_DIR)
    #return read_csv_flexible(RAW_DIR / "closing_odds.csv.gz")
    


def load_odds_series() -> pd.DataFrame:
    return read_csv_flexible(RAW_DIR / "odds_series.csv.gz")


def load_odds_series_matches() -> pd.DataFrame:
    return read_csv_flexible(RAW_DIR / "odds_series_matches.csv.gz")


def load_odds_series_b() -> pd.DataFrame:
    return read_csv_flexible(RAW_DIR / "odds_series_b.csv.gz")


def load_odds_series_b_matches() -> pd.DataFrame:
    return read_csv_flexible(RAW_DIR / "odds_series_b_matches.csv.gz")


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "closing_odds": load_closing_odds(),
        "odds_series": load_odds_series(),
        "odds_series_matches": load_odds_series_matches(),
        "odds_series_b": load_odds_series_b(),
        "odds_series_b_matches": load_odds_series_b_matches(),
    }