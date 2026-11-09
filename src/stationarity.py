"""
Phase 4: Stationarity Analysis Module.

Performs Augmented Dickey-Fuller (ADF) tests, rolling statistics analysis,
and ACF/PACF diagnostics on both the original level capacity series and the
first-differenced series.
"""

import warnings
from typing import Dict, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf


def run_adf_test(series: pd.Series | np.ndarray, series_name: str, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Execute the Augmented Dickey-Fuller (ADF) unit-root test and return
    structured results with formal hypothesis interpretation.
    """
    clean_s = pd.Series(series).dropna().astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = adfuller(clean_s, autolag="AIC")

    adf_stat = float(result[0])
    p_value = float(result[1])
    used_lag = int(result[2])
    n_obs = int(result[3])
    crit_vals = {k: float(v) for k, v in result[4].items()}
    icbest = float(result[5])

    is_stationary = bool(p_value < alpha and adf_stat < crit_vals["5%"])
    conclusion = (
        f"Reject H0 at alpha={alpha:.2f}: Series '{series_name}' is STATIONARY (no unit root)."
        if is_stationary
        else f"Fail to reject H0 at alpha={alpha:.2f}: Series '{series_name}' is NON-STATIONARY (unit root present)."
    )

    return {
        "series_name": series_name,
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "critical_values": crit_vals,
        "ic_best": icbest,
        "is_stationary_at_5pct": is_stationary,
        "conclusion": conclusion,
    }


def analyze_stationarity(df: pd.DataFrame, train_end_idx: int = 134) -> Dict[str, Any]:
    """
    Perform complete stationarity diagnostics on the full series and on the
    chronological training series (to prevent test-data leakage when selecting d).
    """
    cap_full = df["discharge_capacity_ah"]
    cap_diff_full = cap_full.diff().dropna()

    cap_train = cap_full.iloc[:train_end_idx]
    cap_diff_train = cap_train.diff().dropna()

    adf_full_level = run_adf_test(cap_full, "Full Series Q_t (Level, d=0)")
    adf_full_diff1 = run_adf_test(cap_diff_full, "Full Series ΔQ_t (First Difference, d=1)")

    adf_train_level = run_adf_test(cap_train, "Training Series Q_t (Level, d=0)")
    adf_train_diff1 = run_adf_test(cap_diff_train, "Training Series ΔQ_t (First Difference, d=1)")

    # Determine justified integration order d from training series only
    if adf_train_level["is_stationary_at_5pct"]:
        recommended_d = 0
    elif adf_train_diff1["is_stationary_at_5pct"]:
        recommended_d = 1
    else:
        recommended_d = 2

    acf_level = acf(cap_train, nlags=20, fft=False).tolist()
    pacf_level = pacf(cap_train, nlags=20, method="ywm").tolist()
    acf_diff1 = acf(cap_diff_train, nlags=20, fft=False).tolist()
    pacf_diff1 = pacf(cap_diff_train, nlags=20, method="ywm").tolist()

    # Rolling window shift check (first third vs middle third vs final third of training series)
    n_t = len(cap_train)
    third = n_t // 3
    seg1 = cap_train.iloc[:third]
    seg2 = cap_train.iloc[third : 2 * third]
    seg3 = cap_train.iloc[2 * third :]

    return {
        "adf_full_level": adf_full_level,
        "adf_full_diff1": adf_full_diff1,
        "adf_train_level": adf_train_level,
        "adf_train_diff1": adf_train_diff1,
        "recommended_d": recommended_d,
        "acf_train_level_lags_0_10": acf_level[:11],
        "pacf_train_level_lags_0_10": pacf_level[:11],
        "acf_train_diff1_lags_0_10": acf_diff1[:11],
        "pacf_train_diff1_lags_0_10": pacf_diff1[:11],
        "segment_stats_train": {
            "segment_1_mean": float(seg1.mean()),
            "segment_1_std": float(seg1.std()),
            "segment_2_mean": float(seg2.mean()),
            "segment_2_std": float(seg2.std()),
            "segment_3_mean": float(seg3.mean()),
            "segment_3_std": float(seg3.std()),
        },
    }
