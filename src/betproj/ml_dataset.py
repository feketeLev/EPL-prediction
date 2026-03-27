from __future__ import annotations

import pandas as pd


OUTCOME_CONFIG = {
    "H": {
        "p_raw": "p_home_norm",
        "p_cal": "p_home_cal",
        "odds": "max_odds_home_win",
        "avg_odds": "avg_odds_home_win",
        "n_odds": "n_odds_home_win",
        "bookie": "top_bookie_home_win",
        "profit": "profit_home",
        "gap": "gap_home",
        "rel_gap": "rel_gap_home",
    },
    "D": {
        "p_raw": "p_draw_norm",
        "p_cal": "p_draw_cal",
        "odds": "max_odds_draw",
        "avg_odds": "avg_odds_draw",
        "n_odds": "n_odds_draw",
        "bookie": "top_bookie_draw",
        "profit": "profit_draw",
        "gap": "gap_draw",
        "rel_gap": "rel_gap_draw",
    },
    "A": {
        "p_raw": "p_away_norm",
        "p_cal": "p_away_cal",
        "odds": "max_odds_away_win",
        "avg_odds": "avg_odds_away_win",
        "n_odds": "n_odds_away_win",
        "bookie": "top_bookie_away_win",
        "profit": "profit_away",
        "gap": "gap_away",
        "rel_gap": "rel_gap_away",
    },
}


def build_bet_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["year"] = df["match_date"].dt.year
    df["month"] = df["match_date"].dt.month

    frames = []

    for outcome, cfg in OUTCOME_CONFIG.items():
        tmp = pd.DataFrame(
            {
                "match_id": df["match_id"],
                "match_date": df["match_date"],
                "year": df["year"],
                "month": df["month"],
                "league": df["league"],
                "home_team": df["home_team"],
                "away_team": df["away_team"],
                "result": df["result"],
                "bet_outcome": outcome,
                "p_raw": df[cfg["p_raw"]],
                "p_cal": df[cfg["p_cal"]],
                "odds": df[cfg["odds"]],
                "avg_odds": df[cfg["avg_odds"]],
                "n_odds": df[cfg["n_odds"]],
                "top_bookie": df[cfg["bookie"]],
                "gap": df[cfg["gap"]],
                "rel_gap": df[cfg["rel_gap"]],
                "profit": df[cfg["profit"]],
            }
        )

        tmp["won"] = (tmp["bet_outcome"] == tmp["result"]).astype(int)
        tmp["stake"] = 1.0
        frames.append(tmp)

    bets = pd.concat(frames, ignore_index=True)
    bets = bets.sort_values(["match_date", "match_id", "bet_outcome"]).reset_index(drop=True)
    return bets


def default_feature_columns() -> list[str]:
    return [
        "p_raw",
        "p_cal",
        "odds",
        "avg_odds",
        "n_odds",
        "gap",
        "rel_gap",
        "bet_outcome",
        "top_bookie",
        "league",
        "month",
    ]