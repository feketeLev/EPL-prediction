from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


PROB_COLS = {
    "H": "p_home_norm",
    "D": "p_draw_norm",
    "A": "p_away_norm",
}

MAX_ODDS_COLS = {
    "H": "max_odds_home_win",
    "D": "max_odds_draw",
    "A": "max_odds_away_win",
}

CAL_PROB_COLS = {
    "H": "p_home_cal",
    "D": "p_draw_cal",
    "A": "p_away_cal",
}

CAL_EV_COLS = {
    "H": "ev_home",
    "D": "ev_draw",
    "A": "ev_away",
}


def add_year_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")
    out["year"] = out["match_date"].dt.year
    return out


def fit_outcome_calibrator(
    df: pd.DataFrame,
    outcome: str,
) -> IsotonicRegression:
    prob_col = PROB_COLS[outcome]

    x = pd.to_numeric(df[prob_col], errors="coerce")
    y = (df["result"] == outcome).astype(int)

    mask = x.notna() & y.notna()
    x = x.loc[mask].to_numpy()
    y = y.loc[mask].to_numpy()

    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(x, y)
    return model


def fit_calibrators(df_train: pd.DataFrame) -> dict[str, IsotonicRegression]:
    return {
        "H": fit_outcome_calibrator(df_train, "H"),
        "D": fit_outcome_calibrator(df_train, "D"),
        "A": fit_outcome_calibrator(df_train, "A"),
    }


def apply_calibrators(
    df: pd.DataFrame,
    calibrators: dict[str, IsotonicRegression],
) -> pd.DataFrame:
    out = df.copy()

    for outcome, model in calibrators.items():
        raw_col = PROB_COLS[outcome]
        cal_col = CAL_PROB_COLS[outcome]

        x = pd.to_numeric(out[raw_col], errors="coerce")
        pred = np.full(len(out), np.nan, dtype=float)

        mask = x.notna()
        if mask.any():
            pred[mask] = model.predict(x.loc[mask].to_numpy())

        out[cal_col] = pred

    cal_sum = (
        out["p_home_cal"].fillna(0.0)
        + out["p_draw_cal"].fillna(0.0)
        + out["p_away_cal"].fillna(0.0)
    )

    valid = cal_sum > 0
    out.loc[valid, "p_home_cal"] = out.loc[valid, "p_home_cal"] / cal_sum.loc[valid]
    out.loc[valid, "p_draw_cal"] = out.loc[valid, "p_draw_cal"] / cal_sum.loc[valid]
    out.loc[valid, "p_away_cal"] = out.loc[valid, "p_away_cal"] / cal_sum.loc[valid]

    return out


def add_calibrated_ev(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ev_home"] = out["p_home_cal"] * out["max_odds_home_win"] - 1.0
    out["ev_draw"] = out["p_draw_cal"] * out["max_odds_draw"] - 1.0
    out["ev_away"] = out["p_away_cal"] * out["max_odds_away_win"] - 1.0

    return out


def calibration_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for outcome in ["H", "D", "A"]:
        raw_col = PROB_COLS[outcome]
        cal_col = CAL_PROB_COLS[outcome]
        y = (df["result"] == outcome).astype(float)

        raw_brier = ((df[raw_col] - y) ** 2).mean()
        cal_brier = ((df[cal_col] - y) ** 2).mean()

        rows.append(
            {
                "outcome": outcome,
                "raw_brier": raw_brier,
                "cal_brier": cal_brier,
            }
        )

    return pd.DataFrame(rows)


def calibrate_by_year_split(
    df: pd.DataFrame,
    train_start_year: int,
    train_end_year: int,
    test_start_year: int,
    test_end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, IsotonicRegression]]:
    df2 = add_year_column(df)

    train_mask = df2["year"].between(train_start_year, train_end_year, inclusive="both")
    test_mask = df2["year"].between(test_start_year, test_end_year, inclusive="both")

    df_train = df2.loc[train_mask].copy()
    df_test = df2.loc[test_mask].copy()

    calibrators = fit_calibrators(df_train)
    df_test = apply_calibrators(df_test, calibrators)
    df_test = add_calibrated_ev(df_test)

    return df_train, df_test, calibrators