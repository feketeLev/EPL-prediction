from pathlib import Path
import numpy as np
import pandas as pd

from betproj.data_loader import load_closing_odds


def add_match_result(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["result"] = np.where(
        out["home_score"] > out["away_score"],
        "H",
        np.where(out["home_score"] < out["away_score"], "A", "D"),
    )

    return out


def parse_match_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    return out


def preprocess_closing_odds() -> pd.DataFrame:
    df = load_closing_odds()
    df = parse_match_date(df)
    df = add_match_result(df)
    return df