from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _validate_required_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _summary_from_bets(grouped) -> pd.DataFrame:
    out = (
        grouped.agg(
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
    return out


def _finalize_plot(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> plt.Axes:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return ax


def odds_band_test(
    bets: pd.DataFrame,
    bins: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    plot: bool = True,
    min_bets_for_plot: int = 1,
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_id", "won", "profit", "stake", "ev", "odds"])

    if bins is None:
        bins = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, np.inf]
    if labels is None:
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 2)] + [f"{bins[-2]}+"]

    tmp = bets.copy()
    tmp["odds_band"] = pd.cut(tmp["odds"], bins=bins, labels=labels, right=False, include_lowest=True)

    table = _summary_from_bets(tmp.groupby("odds_band", observed=False, dropna=False))
    table = table.sort_values("avg_odds", na_position="last").reset_index(drop=True)

    if plot:
        plot_df = table.loc[table["n_bets"] >= min_bets_for_plot].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(plot_df["odds_band"].astype(str), plot_df["roi"], marker="o")
        ax.axhline(0.0, color="black")
        _finalize_plot(ax, "ROI by Odds Band", "Odds band", "ROI")
        plt.xticks(rotation=45)
        plt.tight_layout()

    return table


def ev_band_test(
    bets: pd.DataFrame,
    bins: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    plot: bool = True,
    min_bets_for_plot: int = 1,
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_id", "won", "profit", "stake", "ev", "odds"])

    if bins is None:
        bins = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, np.inf]
    if labels is None:
        labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 2)] + [f"{bins[-2]:.2f}+"]

    tmp = bets.copy()
    tmp["ev_band"] = pd.cut(tmp["ev"], bins=bins, labels=labels, right=False, include_lowest=True)

    table = _summary_from_bets(tmp.groupby("ev_band", observed=False, dropna=False))
    table = table.sort_values("avg_ev", na_position="last").reset_index(drop=True)

    if plot:
        plot_df = table.loc[table["n_bets"] >= min_bets_for_plot].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(plot_df["ev_band"].astype(str), plot_df["roi"], marker="o")
        ax.axhline(0.0, color="black")
        _finalize_plot(ax, "ROI by EV Band", "EV band", "ROI")
        plt.xticks(rotation=45)
        plt.tight_layout()

    return table


def n_odds_band_test(
    bets: pd.DataFrame,
    bins: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    plot: bool = True,
    min_bets_for_plot: int = 1,
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_id", "won", "profit", "stake", "ev", "odds", "n_odds"])

    if bins is None:
        bins = [0, 6, 11, 16, 21, 26, 33, np.inf]
    if labels is None:
        labels = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-32", "33+"]

    tmp = bets.copy()
    tmp["n_odds_band"] = pd.cut(tmp["n_odds"], bins=bins, labels=labels, right=False, include_lowest=True)

    table = _summary_from_bets(tmp.groupby("n_odds_band", observed=False, dropna=False))
    table = table.reset_index(drop=True)

    if plot:
        plot_df = table.loc[table["n_bets"] >= min_bets_for_plot].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(plot_df["n_odds_band"].astype(str), plot_df["roi"], marker="o")
        ax.axhline(0.0, color="black")
        _finalize_plot(ax, "ROI by Number of Available Odds", "n_odds band", "ROI")
        plt.xticks(rotation=45)
        plt.tight_layout()

    return table


def bookmaker_test(
    bets: pd.DataFrame,
    min_bets: int = 30,
    plot: bool = True,
    top_n: int = 20,
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_id", "won", "profit", "stake", "ev", "odds", "top_bookie"])

    table = _summary_from_bets(bets.groupby("top_bookie", dropna=False))
    table = table.sort_values(["n_bets", "roi"], ascending=[False, False]).reset_index(drop=True)

    filtered = table.loc[table["n_bets"] >= min_bets].copy()
    filtered = filtered.sort_values("roi", ascending=False).reset_index(drop=True)

    if plot and not filtered.empty:
        plot_df = filtered.head(top_n).copy()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(plot_df["top_bookie"].astype(str), plot_df["roi"])
        ax.axhline(0.0, color="black")
        _finalize_plot(ax, f"ROI by Bookmaker (n_bets ≥ {min_bets})", "Bookmaker", "ROI")
        plt.xticks(rotation=75, ha="right")
        plt.tight_layout()

    return filtered


def league_test(
    bets: pd.DataFrame,
    min_bets: int = 50,
    plot: bool = True,
    top_n: int = 20,
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_id", "league", "won", "profit", "stake", "ev", "odds"])

    table = _summary_from_bets(bets.groupby("league", dropna=False))
    table = table.loc[table["n_bets"] >= min_bets].copy()
    table = table.sort_values("roi", ascending=False).reset_index(drop=True)

    if plot and not table.empty:
        plot_df = table.head(top_n).copy()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(plot_df["league"].astype(str), plot_df["roi"])
        ax.axhline(0.0, color="black")
        _finalize_plot(ax, f"ROI by League (n_bets ≥ {min_bets})", "League", "ROI")
        plt.xticks(rotation=75, ha="right")
        plt.tight_layout()

    return table


def calibration_table(
    bets: pd.DataFrame,
    by: str = "ev",
    n_bins: int = 10,
    bins: Sequence[float] | None = None,
    plot: bool = True,
    min_bets_per_bin: int = 1,
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_id", "won", "profit", "stake", "ev", "odds"])

    if by not in {"ev", "odds"}:
        raise ValueError("by must be 'ev' or 'odds'")

    tmp = bets.copy()

    if bins is None:
        tmp["calibration_bin"] = pd.qcut(tmp[by], q=n_bins, duplicates="drop")
    else:
        tmp["calibration_bin"] = pd.cut(tmp[by], bins=bins, include_lowest=True)

    table = (
        tmp.groupby("calibration_bin", observed=False, dropna=False)
        .agg(
            n_bets=("match_id", "size"),
            avg_ev=("ev", "mean"),
            avg_odds=("odds", "mean"),
            hit_rate=("won", "mean"),
            total_profit=("profit", "sum"),
            total_staked=("stake", "sum"),
        )
        .reset_index()
    )
    table["roi"] = table["total_profit"] / table["total_staked"]
    table = table.sort_values(f"avg_{by}").reset_index(drop=True)

    if plot:
        plot_df = table.loc[table["n_bets"] >= min_bets_per_bin].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        x = plot_df[f"avg_{by}"]
        y = plot_df["roi"]
        ax.plot(x, y, marker="o", label="Calibration")
        ax.axhline(0.0, color="black", label="ROI = 0")
        if by == "ev":
            if len(x):
                x_line = np.linspace(float(x.min()), float(x.max()), 100)
            else:
                x_line = np.linspace(0.0, 1.0, 100)
            ax.plot(x_line, x_line, linestyle="--", label="Perfect calibration")
            _finalize_plot(ax, "EV Calibration", "Expected Value", "Realized ROI")
        else:
            _finalize_plot(ax, "Odds Calibration", "Average Odds", "Realized ROI")
        ax.legend()
        plt.tight_layout()

    return table


def yearly_test(
    bets: pd.DataFrame,
    plot: bool = True,
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_id", "match_date", "won", "profit", "stake", "ev", "odds"])

    tmp = bets.copy()
    tmp["year"] = pd.to_datetime(tmp["match_date"]).dt.year

    table = _summary_from_bets(tmp.groupby("year", dropna=False))
    table = table.sort_values("year").reset_index(drop=True)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(table["year"], table["roi"], marker="o")
        ax.axhline(0.0, color="black")
        _finalize_plot(ax, "ROI by Year", "Year", "ROI")
        plt.tight_layout()

    return table


def cumulative_profit_plot(
    bets: pd.DataFrame,
    title: str = "Cumulative Profit",
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_date", "profit"])

    tmp = bets.copy()
    tmp["match_date"] = pd.to_datetime(tmp["match_date"])
    tmp = tmp.sort_values(["match_date"]).reset_index(drop=True)
    tmp["bet_number"] = np.arange(1, len(tmp) + 1)
    tmp["cum_profit"] = tmp["profit"].cumsum()
    tmp["cum_staked"] = tmp["bet_number"].astype(float)
    tmp["cum_roi"] = tmp["cum_profit"] / tmp["cum_staked"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tmp["bet_number"], tmp["cum_profit"])
    ax.axhline(0.0, color="black")
    _finalize_plot(ax, title, "Bet number", "Cumulative profit")
    plt.tight_layout()

    return tmp


def cumulative_profit_by_time_plot(
    bets: pd.DataFrame,
    freq: str = "M",
    title: str = "Cumulative Profit by Time",
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_date", "profit"])

    tmp = bets.copy()
    tmp["match_date"] = pd.to_datetime(tmp["match_date"])

    out = (
        tmp.groupby(pd.Grouper(key="match_date", freq=freq))
        .agg(period_profit=("profit", "sum"))
        .reset_index()
    )
    out["cum_profit"] = out["period_profit"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(out["match_date"], out["cum_profit"], marker="o")
    ax.axhline(0.0, color="black")
    _finalize_plot(ax, title, "Date", "Cumulative profit")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return out


def drawdown_analysis(
    bets: pd.DataFrame,
    plot: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    _validate_required_columns(bets, ["match_date", "profit"])

    tmp = bets.copy()
    tmp["match_date"] = pd.to_datetime(tmp["match_date"])
    tmp = tmp.sort_values("match_date").reset_index(drop=True)
    tmp["bet_number"] = np.arange(1, len(tmp) + 1)
    tmp["cum_profit"] = tmp["profit"].cumsum()
    tmp["running_peak"] = tmp["cum_profit"].cummax()
    tmp["drawdown"] = tmp["cum_profit"] - tmp["running_peak"]

    max_drawdown = float(tmp["drawdown"].min())
    end_idx = int(tmp["drawdown"].idxmin()) if len(tmp) else -1
    peak_idx = int(tmp.loc[:end_idx, "cum_profit"].idxmax()) if end_idx >= 0 else -1

    stats = {
        "max_drawdown": max_drawdown,
        "peak_bet_number": float(tmp.loc[peak_idx, "bet_number"]) if peak_idx >= 0 else np.nan,
        "trough_bet_number": float(tmp.loc[end_idx, "bet_number"]) if end_idx >= 0 else np.nan,
    }

    if plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(tmp["bet_number"], tmp["drawdown"])
        ax.axhline(0.0, color="black")
        _finalize_plot(ax, "Drawdown Curve", "Bet number", "Drawdown")
        plt.tight_layout()

    return tmp, stats


def stability_by_chunk_test(
    bets: pd.DataFrame,
    n_chunks: int = 10,
    plot: bool = True,
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_date", "profit", "stake", "won", "ev", "odds", "match_id"])

    tmp = bets.copy()
    tmp["match_date"] = pd.to_datetime(tmp["match_date"])
    tmp = tmp.sort_values("match_date").reset_index(drop=True)
    tmp["chunk"] = pd.qcut(np.arange(len(tmp)), q=n_chunks, labels=False, duplicates="drop")

    table = _summary_from_bets(tmp.groupby("chunk", dropna=False))
    table = table.sort_values("chunk").reset_index(drop=True)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(table["chunk"], table["roi"], marker="o")
        ax.axhline(0.0, color="black")
        _finalize_plot(ax, "ROI by Chronological Chunk", "Chunk", "ROI")
        plt.tight_layout()

    return table


def outcome_mix_test(
    bets: pd.DataFrame,
    plot: bool = True,
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_id", "bet_outcome", "won", "profit", "stake", "ev", "odds"])

    table = _summary_from_bets(bets.groupby("bet_outcome", dropna=False))
    table = table.sort_values("roi", ascending=False).reset_index(drop=True)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(table["bet_outcome"].astype(str), table["roi"])
        ax.axhline(0.0, color="black")
        _finalize_plot(ax, "ROI by Outcome", "Outcome", "ROI")
        plt.tight_layout()

    return table


def threshold_sweep_test(
    df_features: pd.DataFrame,
    select_bets_func,
    thresholds: Iterable[float],
    mode: str = "best_per_match",
    min_n_odds: int = 1,
    max_odds_allowed: float | None = None,
    min_odds_allowed: float | None = None,
    plot: bool = True,
) -> pd.DataFrame:
    rows = []

    for threshold in thresholds:
        bets = select_bets_func(
            df=df_features,
            ev_threshold=threshold,
            mode=mode,
            min_n_odds=min_n_odds,
            max_odds_allowed=max_odds_allowed,
            min_odds_allowed=min_odds_allowed,
        )
        if bets.empty:
            rows.append(
                {
                    "ev_threshold": threshold,
                    "n_bets": 0,
                    "hit_rate": np.nan,
                    "total_profit": 0.0,
                    "total_staked": 0.0,
                    "avg_ev": np.nan,
                    "avg_odds": np.nan,
                    "roi": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "ev_threshold": threshold,
                    "n_bets": len(bets),
                    "hit_rate": bets["won"].mean(),
                    "total_profit": bets["profit"].sum(),
                    "total_staked": bets["stake"].sum(),
                    "avg_ev": bets["ev"].mean(),
                    "avg_odds": bets["odds"].mean(),
                    "roi": bets["profit"].sum() / bets["stake"].sum(),
                }
            )

    table = pd.DataFrame(rows).sort_values("ev_threshold").reset_index(drop=True)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(table["ev_threshold"], table["roi"], marker="o")
        ax.axhline(0.0, color="black")
        _finalize_plot(ax, "ROI by EV Threshold", "EV threshold", "ROI")
        plt.tight_layout()

    return table


def randomized_ev_test(
    bets: pd.DataFrame,
    n_bins: int = 10,
    random_state: int = 42,
    plot: bool = True,
) -> pd.DataFrame:
    _validate_required_columns(bets, ["match_id", "profit", "stake", "won", "ev", "odds"])

    tmp = bets.copy()
    rng = np.random.default_rng(random_state)
    tmp["ev_random"] = rng.permutation(tmp["ev"].to_numpy())
    tmp["random_bin"] = pd.qcut(tmp["ev_random"], q=n_bins, duplicates="drop")

    table = (
        tmp.groupby("random_bin", observed=False, dropna=False)
        .agg(
            n_bets=("match_id", "size"),
            avg_ev_random=("ev_random", "mean"),
            total_profit=("profit", "sum"),
            total_staked=("stake", "sum"),
        )
        .reset_index()
    )
    table["roi"] = table["total_profit"] / table["total_staked"]
    table = table.sort_values("avg_ev_random").reset_index(drop=True)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(table["avg_ev_random"], table["roi"], marker="o")
        ax.axhline(0.0, color="black")
        _finalize_plot(ax, "Randomized EV Baseline", "Randomized EV", "Realized ROI")
        plt.tight_layout()

    return table


def bootstrap_roi_test(
    bets: pd.DataFrame,
    n_bootstrap: int = 2000,
    random_state: int = 42,
    plot: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    _validate_required_columns(bets, ["profit", "stake"])

    profits = bets["profit"].to_numpy()
    n = len(profits)
    rng = np.random.default_rng(random_state)

    boot_rois = []
    for _ in range(n_bootstrap):
        sample = rng.choice(profits, size=n, replace=True)
        boot_rois.append(sample.sum() / n)

    boot_df = pd.DataFrame({"bootstrap_roi": boot_rois})

    observed_roi = float(bets["profit"].sum() / bets["stake"].sum())
    ci_low = float(np.percentile(boot_rois, 2.5))
    ci_high = float(np.percentile(boot_rois, 97.5))
    p_boot_le_zero = float(np.mean(np.array(boot_rois) <= 0.0))

    stats = {
        "observed_roi": observed_roi,
        "bootstrap_mean_roi": float(np.mean(boot_rois)),
        "bootstrap_std_roi": float(np.std(boot_rois, ddof=1)),
        "ci_2_5": ci_low,
        "ci_97_5": ci_high,
        "p_bootstrap_roi_le_zero": p_boot_le_zero,
    }

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(boot_df["bootstrap_roi"], bins=40)
        ax.axvline(observed_roi, color="red", linestyle="--")
        ax.axvline(0.0, color="black")
        _finalize_plot(ax, "Bootstrap ROI Distribution", "Bootstrap ROI", "Count")
        plt.tight_layout()

    return boot_df, stats


def accuracy_diagnostics(
    bets: pd.DataFrame,
) -> dict[str, float]:
    _validate_required_columns(bets, ["won", "profit", "ev", "odds", "stake"])

    roi = float(bets["profit"].sum() / bets["stake"].sum())
    avg_profit_per_bet = float(bets["profit"].mean())
    profit_std = float(bets["profit"].std(ddof=1))
    roi_volatility_proxy = profit_std / float(bets["stake"].mean()) if len(bets) > 1 else np.nan

    return {
        "n_bets": float(len(bets)),
        "hit_rate": float(bets["won"].mean()),
        "avg_ev": float(bets["ev"].mean()),
        "avg_odds": float(bets["odds"].mean()),
        "roi": roi,
        "avg_profit_per_bet": avg_profit_per_bet,
        "profit_std": profit_std,
        "roi_volatility_proxy": roi_volatility_proxy,
        "profit_ev_correlation": float(bets["profit"].corr(bets["ev"])),
        "win_ev_correlation": float(bets["won"].corr(bets["ev"])),
    }


def run_diagnostic_suite(
    bets: pd.DataFrame,
    bookmaker_min_bets: int = 30,
    league_min_bets: int = 50,
    chunk_count: int = 10,
) -> dict[str, object]:
    results = {}

    results["summary_metrics"] = accuracy_diagnostics(bets)
    results["odds_band"] = odds_band_test(bets, plot=True)
    results["ev_band"] = ev_band_test(bets, plot=True)
    results["n_odds_band"] = n_odds_band_test(bets, plot=True)
    results["bookmaker"] = bookmaker_test(bets, min_bets=bookmaker_min_bets, plot=True)
    results["league"] = league_test(bets, min_bets=league_min_bets, plot=True)
    results["ev_calibration"] = calibration_table(bets, by="ev", plot=True)
    results["yearly"] = yearly_test(bets, plot=True)
    results["outcome_mix"] = outcome_mix_test(bets, plot=True)
    results["stability_chunks"] = stability_by_chunk_test(bets, n_chunks=chunk_count, plot=True)
    results["randomized_ev"] = randomized_ev_test(bets, plot=True)

    cumulative_profit_plot(bets)
    results["cum_profit_time"] = cumulative_profit_by_time_plot(bets, freq="M")

    dd_df, dd_stats = drawdown_analysis(bets, plot=True)
    results["drawdown_curve"] = dd_df
    results["drawdown_stats"] = dd_stats

    boot_df, boot_stats = bootstrap_roi_test(bets, plot=True)
    results["bootstrap_roi"] = boot_df
    results["bootstrap_stats"] = boot_stats

    return results