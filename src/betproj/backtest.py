from __future__ import annotations

import numpy as np
import pandas as pd


OUTCOME_COLS = {
    "H": {
        "ev": "ev_home",
        "profit": "profit_home",
        "odds": "max_odds_home_win",
        "avg_odds": "avg_odds_home_win",
        "n_odds": "n_odds_home_win",
        "bookie": "top_bookie_home_win",
    },
    "D": {
        "ev": "ev_draw",
        "profit": "profit_draw",
        "odds": "max_odds_draw",
        "avg_odds": "avg_odds_draw",
        "n_odds": "n_odds_draw",
        "bookie": "top_bookie_draw",
    },
    "A": {
        "ev": "ev_away",
        "profit": "profit_away",
        "odds": "max_odds_away_win",
        "avg_odds": "avg_odds_away_win",
        "n_odds": "n_odds_away_win",
        "bookie": "top_bookie_away_win",
    },
}


def _build_candidate_bets(df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for outcome, cols in OUTCOME_COLS.items():
        tmp = pd.DataFrame(
            {
                "match_id": df["match_id"],
                "league": df["league"],
                "match_date": df["match_date"],
                "home_team": df["home_team"],
                "away_team": df["away_team"],
                "result": df["result"],
                "bet_outcome": outcome,
                "ev": df[cols["ev"]],
                "profit": df[cols["profit"]],
                "odds": df[cols["odds"]],
                "avg_odds": df[cols["avg_odds"]],
                "n_odds": df[cols["n_odds"]],
                "top_bookie": df[cols["bookie"]],
            }
        )
        frames.append(tmp)

    bets = pd.concat(frames, ignore_index=True)
    bets["won"] = (bets["bet_outcome"] == bets["result"]).astype(int)
    bets = bets.sort_values(["match_date", "match_id", "bet_outcome"]).reset_index(drop=True)
    return bets


def select_bets(
    df: pd.DataFrame,
    ev_threshold: float = 0.0,
    mode: str = "best_per_match",
    min_n_odds: int = 1,
    max_odds_allowed: float | None = None,
    min_odds_allowed: float | None = None,
) -> pd.DataFrame:
    bets = _build_candidate_bets(df)

    bets = bets.loc[bets["ev"] > ev_threshold].copy()
    bets = bets.loc[bets["n_odds"] >= min_n_odds].copy()

    if max_odds_allowed is not None:
        bets = bets.loc[bets["odds"] <= max_odds_allowed].copy()

    if min_odds_allowed is not None:
        bets = bets.loc[bets["odds"] >= min_odds_allowed].copy()

    if mode == "all":
        selected = bets.copy()

    elif mode == "best_per_match":
        idx = bets.groupby("match_id")["ev"].idxmax()
        selected = bets.loc[idx].copy()

    else:
        raise ValueError("mode must be 'all' or 'best_per_match'")

    selected = selected.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    selected["stake"] = 1.0
    selected["cum_profit"] = selected["profit"].cumsum()
    selected["cum_staked"] = selected["stake"].cumsum()
    selected["cum_roi"] = selected["cum_profit"] / selected["cum_staked"]

    return selected


def summarize_bets(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame(
            [
                {
                    "n_bets": 0,
                    "hit_rate": np.nan,
                    "total_profit": 0.0,
                    "total_staked": 0.0,
                    "roi": np.nan,
                    "avg_ev": np.nan,
                    "avg_odds": np.nan,
                }
            ]
        )

    total_profit = float(bets["profit"].sum())
    total_staked = float(bets["stake"].sum())

    summary = pd.DataFrame(
        [
            {
                "n_bets": int(len(bets)),
                "hit_rate": float(bets["won"].mean()),
                "total_profit": total_profit,
                "total_staked": total_staked,
                "roi": total_profit / total_staked if total_staked > 0 else np.nan,
                "avg_ev": float(bets["ev"].mean()),
                "avg_odds": float(bets["odds"].mean()),
            }
        ]
    )

    return summary


def summarize_by_group(bets: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame(
            columns=[
                group_col,
                "n_bets",
                "hit_rate",
                "total_profit",
                "total_staked",
                "roi",
                "avg_ev",
                "avg_odds",
            ]
        )

    out = (
        bets.groupby(group_col, dropna=False)
        .agg(
            n_bets=("match_id", "size"),
            hit_rate=("won", "mean"),
            total_profit=("profit", "sum"),
            total_staked=("stake", "sum"),
            avg_ev=("ev", "mean"),
            avg_odds=("odds", "mean"),
        )
        .reset_index()
    )

    out["roi"] = out["total_profit"] / out["total_staked"]
    out = out.sort_values("roi", ascending=False).reset_index(drop=True)
    return out


def add_year_column(bets: pd.DataFrame) -> pd.DataFrame:
    out = bets.copy()
    out["year"] = pd.to_datetime(out["match_date"]).dt.year
    return out


def add_n_odds_bin(bets: pd.DataFrame) -> pd.DataFrame:
    out = bets.copy()
    bins = [0, 5, 10, 15, 20, 25, 32, np.inf]
    labels = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-32", "33+"]
    out["n_odds_bin"] = pd.cut(out["n_odds"], bins=bins, labels=labels, right=True)
    return out


def threshold_grid_backtest(
    df: pd.DataFrame,
    thresholds: list[float],
    mode: str = "best_per_match",
    min_n_odds: int = 1,
    max_odds_allowed: float | None = None,
    min_odds_allowed: float | None = None,
) -> pd.DataFrame:
    rows = []

    for threshold in thresholds:
        bets = select_bets(
            df=df,
            ev_threshold=threshold,
            mode=mode,
            min_n_odds=min_n_odds,
            max_odds_allowed=max_odds_allowed,
            min_odds_allowed=min_odds_allowed,
        )
        s = summarize_bets(bets).iloc[0].to_dict()
        s["ev_threshold"] = threshold
        s["mode"] = mode
        s["min_n_odds"] = min_n_odds
        s["max_odds_allowed"] = max_odds_allowed
        s["min_odds_allowed"] = min_odds_allowed
        rows.append(s)

    out = pd.DataFrame(rows)
    return out.sort_values("ev_threshold").reset_index(drop=True)


def evaluation_tables(bets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}

    bets2 = add_year_column(bets)
    bets2 = add_n_odds_bin(bets2)

    out["summary"] = summarize_bets(bets2)
    out["by_year"] = summarize_by_group(bets2, "year")
    out["by_league"] = summarize_by_group(bets2, "league")
    out["by_outcome"] = summarize_by_group(bets2, "bet_outcome")
    out["by_bookie"] = summarize_by_group(bets2, "top_bookie")
    out["by_n_odds_bin"] = summarize_by_group(bets2, "n_odds_bin")

    return out