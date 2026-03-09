from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_inv(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.where(x > 0)
    return 1.0 / x


def add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["p_avg_home"] = _safe_inv(out["avg_odds_home_win"])
    out["p_avg_draw"] = _safe_inv(out["avg_odds_draw"])
    out["p_avg_away"] = _safe_inv(out["avg_odds_away_win"])

    out["p_max_home"] = _safe_inv(out["max_odds_home_win"])
    out["p_max_draw"] = _safe_inv(out["max_odds_draw"])
    out["p_max_away"] = _safe_inv(out["max_odds_away_win"])

    return out


def add_overround(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["overround_avg"] = (
        _safe_inv(out["avg_odds_home_win"])
        + _safe_inv(out["avg_odds_draw"])
        + _safe_inv(out["avg_odds_away_win"])
    )

    out["overround_max"] = (
        _safe_inv(out["max_odds_home_win"])
        + _safe_inv(out["max_odds_draw"])
        + _safe_inv(out["max_odds_away_win"])
    )

    return out


def add_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["gap_home"] = out["max_odds_home_win"] - out["avg_odds_home_win"]
    out["gap_draw"] = out["max_odds_draw"] - out["avg_odds_draw"]
    out["gap_away"] = out["max_odds_away_win"] - out["avg_odds_away_win"]

    out["rel_gap_home"] = out["gap_home"] / out["avg_odds_home_win"]
    out["rel_gap_draw"] = out["gap_draw"] / out["avg_odds_draw"]
    out["rel_gap_away"] = out["gap_away"] / out["avg_odds_away_win"]

    return out


def add_expected_values_from_avg_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ev_home"] = _safe_inv(out["avg_odds_home_win"]) * out["max_odds_home_win"] - 1.0
    out["ev_draw"] = _safe_inv(out["avg_odds_draw"]) * out["max_odds_draw"] - 1.0
    out["ev_away"] = _safe_inv(out["avg_odds_away_win"]) * out["max_odds_away_win"] - 1.0

    return out


def add_profit_columns(df: pd.DataFrame, stake: float = 1.0) -> pd.DataFrame:
    out = df.copy()

    out["profit_home"] = np.where(
        out["result"] == "H",
        stake * (out["max_odds_home_win"] - 1.0),
        -stake,
    )

    out["profit_draw"] = np.where(
        out["result"] == "D",
        stake * (out["max_odds_draw"] - 1.0),
        -stake,
    )

    out["profit_away"] = np.where(
        out["result"] == "A",
        stake * (out["max_odds_away_win"] - 1.0),
        -stake,
    )

    return out


def build_closing_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = add_normalized_probabilities(out)
    out = add_value_gaps(out)
    out = add_expected_values(out)
    out = add_profit_columns(out)

    return out

def add_normalized_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    inv_home = 1 / out["avg_odds_home_win"]
    inv_draw = 1 / out["avg_odds_draw"]
    inv_away = 1 / out["avg_odds_away_win"]

    denom = inv_home + inv_draw + inv_away

    out["p_home_norm"] = inv_home / denom
    out["p_draw_norm"] = inv_draw / denom
    out["p_away_norm"] = inv_away / denom

    return out


def add_expected_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ev_home"] = out["p_home_norm"] * out["max_odds_home_win"] - 1
    out["ev_draw"] = out["p_draw_norm"] * out["max_odds_draw"] - 1
    out["ev_away"] = out["p_away_norm"] * out["max_odds_away_win"] - 1

    return out