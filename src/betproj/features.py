import numpy as np
import pandas as pd

def implied_prob(odds: pd.Series) -> pd.Series:
    return 1.0 / odds

def normalize_probs(p: pd.DataFrame) -> pd.DataFrame:
    return p.div(p.sum(axis=1), axis=0)

def consensus_prob_mean_odds(odds: pd.Series) -> float:
    return 1.0 / float(np.mean(odds))

def expected_value(p: float, odds: float) -> float:
    return p * odds - 1.0
