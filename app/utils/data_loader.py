"""
Cached data loading and causal model execution utilities for the Streamlit dashboard.
Resolves all paths relative to the project root and reuses validated `src.*` functions.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.stattools import acf, pacf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import run_full_data_audit
from src.preprocessing import preprocess_battery_data
from src.stationarity import analyze_stationarity
from src.models import (
    _tune_holt_causal,
    estimate_rul_threshold_crossing,
    select_and_fit_arima,
    select_and_fit_holt,
)
from src.evaluation import calculate_metrics


DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "battery_capacity.csv"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FORECASTS_DIR = RESULTS_DIR / "forecasts"
FIGURES_DIR = RESULTS_DIR / "figures"


@st.cache_data(show_spinner=False)
def load_processed_battery_data() -> pd.DataFrame:
    """
    Load the processed cycle-level NASA B0006 dataset (`168 rows x 14 columns`).
    If the CSV is missing, gracefully regenerate it from `data/raw/`.
    """
    if DATA_PROCESSED_PATH.exists():
        df = pd.read_csv(DATA_PROCESSED_PATH)
        return df.sort_values("cycle_number", ascending=True).reset_index(drop=True)

    if RAW_DATA_DIR.exists():
        raw_data, _ = run_full_data_audit(RAW_DATA_DIR)
        df = preprocess_battery_data(raw_data, output_csv_path=DATA_PROCESSED_PATH)
        return df

    raise FileNotFoundError(
        f"Neither processed dataset ({DATA_PROCESSED_PATH}) nor raw data directory ({RAW_DATA_DIR}) was found."
    )


@st.cache_data(show_spinner=False)
def load_stationarity_results() -> Dict[str, Any]:
    """
    Load formal ADF stationarity results from `results/metrics/stationarity_results.json`,
    or compute them via `src.stationarity.analyze_stationarity` if the file is missing.
    """
    stat_path = METRICS_DIR / "stationarity_results.json"
    if stat_path.exists():
        return json.loads(stat_path.read_text(encoding="utf-8"))

    df = load_processed_battery_data()
    return analyze_stationarity(df, train_end_idx=134)


@st.cache_data(show_spinner=False)
def compute_acf_pacf_arrays(max_lags: int = 35, train_only: bool = True) -> Dict[str, Any]:
    """
    Compute ACF and PACF arrays with 95% Bartlett confidence intervals for interactive Plotly charts.
    Defaults to training-only (Cycles 1..134) to prevent test-set lookahead leakage.
    """
    df = load_processed_battery_data()
    sub_df = df[df["cycle_number"] <= 134].copy() if train_only else df.copy()
    q_level = sub_df["discharge_capacity_ah"].values.astype(float)
    q_diff1 = np.diff(q_level)

    lags_lvl = min(max_lags, len(q_level) // 2 - 1)
    lags_dif = min(max_lags, len(q_diff1) // 2 - 1)

    acf_lvl = acf(q_level, nlags=lags_lvl, fft=False)
    pacf_lvl = pacf(q_level, nlags=lags_lvl, method="ywm")
    acf_dif = acf(q_diff1, nlags=lags_dif, fft=False)
    pacf_dif = pacf(q_diff1, nlags=lags_dif, method="ywm")

    return {
        "n_level": len(q_level),
        "n_diff1": len(q_diff1),
        "ci_level": 1.96 / np.sqrt(len(q_level)),
        "ci_diff1": 1.96 / np.sqrt(len(q_diff1)),
        "acf_level": acf_lvl,
        "pacf_level": pacf_lvl,
        "acf_diff1": acf_dif,
        "pacf_diff1": pacf_dif,
    }


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_evaluation_artifacts() -> Dict[str, Any]:
    """
    Load all pre-computed CSV and JSON evaluation artifacts from `results/metrics/` and `results/forecasts/`.
    Gracefully handles any missing file by returning None for that key.
    """
    exp_summary_path = METRICS_DIR / "experiment_summary.json"
    exp_summary = (
        json.loads(exp_summary_path.read_text(encoding="utf-8"))
        if exp_summary_path.exists()
        else None
    )

    return {
        "model_comparison": _safe_read_csv(METRICS_DIR / "model_comparison_metrics.csv"),
        "train_val_test": _safe_read_csv(METRICS_DIR / "train_val_test_metrics.csv"),
        "online_recalibration": _safe_read_csv(METRICS_DIR / "online_recalibration_metrics.csv"),
        "horizon_comparison": _safe_read_csv(METRICS_DIR / "horizon_comparison.csv"),
        "rolling_origin_eval": _safe_read_csv(METRICS_DIR / "rolling_origin_evaluation.csv"),
        "rolling_origin_summary": _safe_read_csv(METRICS_DIR / "rolling_origin_metrics.csv"),
        "residual_diagnostics": _safe_read_csv(METRICS_DIR / "residual_diagnostics.csv"),
        "arima_order_selection": _safe_read_csv(METRICS_DIR / "arima_order_selection.csv"),
        "holt_parameter_tuning": _safe_read_csv(METRICS_DIR / "holt_parameter_tuning.csv"),
        "test_forecasts": _safe_read_csv(FORECASTS_DIR / "test_forecasts_comparison.csv"),
        "experiment_summary": exp_summary,
    }


@st.cache_data(show_spinner=False)
def run_causal_forecast_at_origin(origin_cycle: int = 134, horizon: int = 34) -> Dict[str, Any]:
    """
    Fit all statistical models strictly on historical observations `Cycles 1..origin_cycle`
    (zero future data leakage) and generate multi-step forecasts for `horizon` cycles ahead.

    - When `origin_cycle == 134`, delegates directly to the validated `select_and_fit_arima`
      and `select_and_fit_holt` functions in `src.models`.
    - For any other user-selected `origin_cycle`, fits `ARIMA(0,1,1)` and `ARIMA(0,1,1)` with Drift
      and causally tunes Holt Linear and Damped Holt via `_tune_holt_causal` strictly on `Cycles 1..origin_cycle`.
    """
    df = load_processed_battery_data()
    q_init = float(df["discharge_capacity_ah"].iloc[0])

    train_sub = df[df["cycle_number"] <= origin_cycle].copy()
    future_actual_df = df[
        (df["cycle_number"] > origin_cycle) & (df["cycle_number"] <= origin_cycle + horizon)
    ].copy()

    y_train = train_sub["discharge_capacity_ah"].values.astype(float)
    future_cycles = np.arange(origin_cycle + 1, origin_cycle + horizon + 1, dtype=int)

    if origin_cycle == 134:
        arima_res = select_and_fit_arima(train_sub["discharge_capacity_ah"], test_horizon=horizon, d=1)
        holt_res = select_and_fit_holt(train_sub["discharge_capacity_ah"], test_horizon=horizon)

        models_out = {
            "ARIMA (0,1,1)": {
                "short_name": "ARIMA (0,1,1)",
                "full_name": "Model 1: ARIMA(0, 1, 1) — Baseline",
                "fitted_values": arima_res["fitted_values"],
                "forecast": arima_res["forecast"],
                "ci_lower": arima_res["conf_int_lower"],
                "ci_upper": arima_res["conf_int_upper"],
                "has_ci": True,
                "params": {
                    "Order (p, d, q)": "(0, 1, 1)",
                    "Drift / Trend": "None (trend='n')",
                    "MA(1) θ₁": f"{arima_res['params'].get('ma.L1', 0.0):.4f}",
                    "Innovation Variance σ²": f"{arima_res['params'].get('sigma2', 0.0):.6f}",
                    "Training AIC": f"{arima_res['aic']:.2f}",
                    "Training BIC": f"{arima_res['bic']:.2f}",
                },
                "explanation": (
                    "**ARIMA(0, 1, 1) Baseline (`trend='n'`):** Models first-differenced capacity increments "
                    "$\\Delta Q_t = \\varepsilon_t + \\theta_1 \\varepsilon_{t-1}$ with zero unconditional drift. "
                    "Because $\\mathbb{E}[\\Delta Q_{t+h}] = 0$ for $h \\ge 2$, multi-step forecasts converge to a "
                    "constant horizontal level ($1.3552\\text{ Ah}$), failing to project continued capacity fade."
                ),
            },
            "ARIMA (0,1,1) with drift": {
                "short_name": "ARIMA (0,1,1) with drift",
                "full_name": "Model 2: ARIMA(0, 1, 1) with Linear Drift",
                "fitted_values": arima_res["arima_with_drift_fitted"],
                "forecast": arima_res["arima_with_drift_forecast"],
                "ci_lower": arima_res["arima_with_drift_ci_lower"],
                "ci_upper": arima_res["arima_with_drift_ci_upper"],
                "has_ci": True,
                "params": {
                    "Order (p, d, q)": "(0, 1, 1)",
                    "Drift Slope c (Ah/cycle)": f"{arima_res['arima_with_drift_params'].get('x1', 0.0):.6f}",
                    "MA(1) θ₁": f"{arima_res['arima_with_drift_params'].get('ma.L1', 0.0):.4f}",
                    "Innovation Variance σ²": f"{arima_res['arima_with_drift_params'].get('sigma2', 0.0):.6f}",
                    "Training AIC": f"{arima_res['arima_with_drift_aic']:.2f}",
                    "Training BIC": f"{arima_res['arima_with_drift_bic']:.2f}",
                },
                "explanation": (
                    "**ARIMA(0, 1, 1) with Linear Drift (`trend='t'`):** Augments first-differenced ARIMA with a "
                    "deterministic per-cycle drift constant $c = -0.005251\\text{ Ah/cycle}$ ($\\Delta Q_t = c + \\varepsilon_t + \\theta_1 \\varepsilon_{t-1}$). "
                    "Multi-step forecasts extrapolate linearly with slope $c$, cutting 34-cycle test RMSE by 86.4% relative to un-drifted ARIMA."
                ),
            },
            "Holt Linear Trend": {
                "short_name": "Holt Linear Trend",
                "full_name": f"Model 3: Holt's Linear Trend (Val-Tuned α={holt_res['smoothing_level_alpha']:.2f}, β={holt_res['smoothing_trend_beta']:.2f})",
                "fitted_values": holt_res["fitted_values"],
                "forecast": holt_res["forecast"],
                "ci_lower": None,
                "ci_upper": None,
                "has_ci": False,
                "params": {
                    "Level Smoothing α": f"{holt_res['smoothing_level_alpha']:.2f}",
                    "Trend Smoothing β": f"{holt_res['smoothing_trend_beta']:.2f}",
                    "Terminal Level L_t (Ah)": f"{holt_res['final_train_level']:.4f}",
                    "Terminal Slope T_t (Ah/cycle)": f"{holt_res['final_train_trend']:.6f}",
                    "Training AIC": f"{holt_res['aic']:.2f}",
                    "Selection Method": "Inner Chronological Validation (Cycles 111–134)",
                },
                "explanation": (
                    "**Holt's Linear Trend (Double Exponential Smoothing — Primary Enhanced Model):** Explicitly tracks "
                    "both smoothed capacity level ($L_t$) and local per-cycle degradation slope ($T_t$) via "
                    "$\\hat{Q}_{t+h|t} = L_t + h T_t$. Tuned causally on the inner validation window ($\\alpha=0.10, \\beta=0.05$), "
                    "it filters transient rest-regeneration spikes and achieves the lowest error across short-to-medium horizons ($h=5, 10, 20$) and 5 rolling origins."
                ),
            },
            "Damped Holt": {
                "short_name": "Damped Holt",
                "full_name": (
                    f"Model 4: Damped Holt's Trend (Val-Tuned α={holt_res['damped_params']['alpha']:.2f}, "
                    f"β={holt_res['damped_params']['beta']:.2f}, φ={holt_res['damped_params']['phi']:.2f})"
                ),
                "fitted_values": holt_res["damped_holt_fitted"],
                "forecast": holt_res["damped_holt_forecast"],
                "ci_lower": None,
                "ci_upper": None,
                "has_ci": False,
                "params": {
                    "Level Smoothing α": f"{holt_res['damped_params']['alpha']:.2f}",
                    "Trend Smoothing β": f"{holt_res['damped_params']['beta']:.2f}",
                    "Damping Parameter φ": f"{holt_res['damped_params']['phi']:.2f}",
                    "Training AIC": f"{holt_res['damped_params']['aic']:.2f}",
                    "Training BIC": f"{holt_res['damped_params']['bic']:.2f}",
                    "Selection Method": "Inner Chronological Validation (Cycles 111–134)",
                },
                "explanation": (
                    "**Damped Holt's Trend:** Introduces an autoregressive damping factor $\\phi \\in (0, 1)$ on the trend state "
                    "($\\hat{Q}_{t+h|t} = L_t + \\sum_{j=1}^h \\phi^j T_t$). While damping prevents runaway extrapolation when degradation plateaus, "
                    "Cell B0006 continues losing capacity steadily across Cycles 135–164, so damping ($\\phi=0.98$) flattens slightly above the actual trajectory."
                ),
            },
        }
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_arima = ARIMA(y_train, order=(0, 1, 1)).fit()
            res_a = m_arima.get_forecast(steps=horizon)
            pred_a = np.asarray(res_a.predicted_mean, dtype=float)
            ci_a = np.asarray(res_a.conf_int(alpha=0.05), dtype=float)

            m_drift = ARIMA(y_train, order=(0, 1, 1), trend="t").fit()
            res_d = m_drift.get_forecast(steps=horizon)
            pred_d = np.asarray(res_d.predicted_mean, dtype=float)
            ci_d = np.asarray(res_d.conf_int(alpha=0.05), dtype=float)

            c_alpha, c_beta, _ = _tune_holt_causal(y_train, val_size=18, damped=False)
            m_holt = Holt(y_train, initialization_method="estimated").fit(
                smoothing_level=c_alpha, smoothing_trend=c_beta, optimized=False
            )
            pred_h = np.asarray(m_holt.forecast(horizon), dtype=float)

            d_alpha, d_beta, d_phi = _tune_holt_causal(y_train, val_size=18, damped=True)
            m_damped = Holt(y_train, damped_trend=True, initialization_method="estimated").fit(
                smoothing_level=d_alpha, smoothing_trend=d_beta, damping_trend=d_phi, optimized=False
            )
            pred_dm = np.asarray(m_damped.forecast(horizon), dtype=float)

        models_out = {
            "ARIMA (0,1,1)": {
                "short_name": "ARIMA (0,1,1)",
                "full_name": f"Model 1: ARIMA(0, 1, 1) — Origin {origin_cycle}",
                "fitted_values": np.asarray(m_arima.fittedvalues, dtype=float),
                "forecast": pred_a,
                "ci_lower": ci_a[:, 0],
                "ci_upper": ci_a[:, 1],
                "has_ci": True,
                "params": {
                    "Order (p, d, q)": "(0, 1, 1)",
                    "Drift / Trend": "None (trend='n')",
                    "MA(1) θ₁": f"{float(m_arima.params[0]):.4f}",
                    "Training AIC": f"{float(m_arima.aic):.2f}",
                    "Training Cycles Used": f"1–{origin_cycle}",
                },
                "explanation": (
                    "**ARIMA(0, 1, 1) Baseline (`trend='n'`):** Fitted causally on Cycles "
                    f"1–{origin_cycle}. Multi-step forecasts plateau horizontally without a drift state."
                ),
            },
            "ARIMA (0,1,1) with drift": {
                "short_name": "ARIMA (0,1,1) with drift",
                "full_name": f"Model 2: ARIMA(0, 1, 1) with Drift — Origin {origin_cycle}",
                "fitted_values": np.asarray(m_drift.fittedvalues, dtype=float),
                "forecast": pred_d,
                "ci_lower": ci_d[:, 0],
                "ci_upper": ci_d[:, 1],
                "has_ci": True,
                "params": {
                    "Order (p, d, q)": "(0, 1, 1)",
                    "Drift Slope c (Ah/cycle)": f"{float(m_drift.params[0]):.6f}",
                    "MA(1) θ₁": f"{float(m_drift.params[1]):.4f}",
                    "Training AIC": f"{float(m_drift.aic):.2f}",
                    "Training Cycles Used": f"1–{origin_cycle}",
                },
                "explanation": (
                    f"**ARIMA(0, 1, 1) with Drift (`trend='t'`):** Fitted causally on Cycles 1–{origin_cycle}, "
                    "extrapolating the average historical per-cycle capacity loss rate across future cycles."
                ),
            },
            "Holt Linear Trend": {
                "short_name": "Holt Linear Trend",
                "full_name": f"Model 3: Holt's Linear Trend (Causal α={c_alpha:.2f}, β={c_beta:.2f}) — Origin {origin_cycle}",
                "fitted_values": np.asarray(m_holt.fittedvalues, dtype=float),
                "forecast": pred_h,
                "ci_lower": None,
                "ci_upper": None,
                "has_ci": False,
                "params": {
                    "Level Smoothing α": f"{c_alpha:.2f}",
                    "Trend Smoothing β": f"{c_beta:.2f}",
                    "Terminal Level L_t (Ah)": f"{float(m_holt.level[-1]):.4f}",
                    "Terminal Slope T_t (Ah/cycle)": f"{float(m_holt.trend[-1]):.6f}",
                    "Training AIC": f"{float(m_holt.aic):.2f}",
                    "Training Cycles Used": f"1–{origin_cycle} (Causal Inner Tuning)",
                },
                "explanation": (
                    f"**Holt's Linear Trend:** Tuned causally on an inner validation split of Cycles 1–{origin_cycle} "
                    f"($\\alpha={c_alpha:.2f}, \\beta={c_beta:.2f}$) with zero future leakage."
                ),
            },
            "Damped Holt": {
                "short_name": "Damped Holt",
                "full_name": f"Model 4: Damped Holt (Causal α={d_alpha:.2f}, β={d_beta:.2f}, φ={d_phi:.2f}) — Origin {origin_cycle}",
                "fitted_values": np.asarray(m_damped.fittedvalues, dtype=float),
                "forecast": pred_dm,
                "ci_lower": None,
                "ci_upper": None,
                "has_ci": False,
                "params": {
                    "Level Smoothing α": f"{d_alpha:.2f}",
                    "Trend Smoothing β": f"{d_beta:.2f}",
                    "Damping Parameter φ": f"{d_phi:.2f}",
                    "Training AIC": f"{float(m_damped.aic):.2f}",
                    "Training Cycles Used": f"1–{origin_cycle} (Causal Inner Tuning)",
                },
                "explanation": (
                    f"**Damped Holt's Trend:** Tuned causally on Cycles 1–{origin_cycle} "
                    f"($\\alpha={d_alpha:.2f}, \\beta={d_beta:.2f}, \\phi={d_phi:.2f}$)."
                ),
            },
        }

    # Attach evaluation metrics on overlapping future cycles where actual observations exist
    n_eval = len(future_actual_df)
    actual_eval = future_actual_df["discharge_capacity_ah"].values.astype(float) if n_eval > 0 else np.array([])
    for key, m_info in models_out.items():
        if n_eval > 0:
            pred_eval = m_info["forecast"][:n_eval]
            m_info["eval_metrics"] = calculate_metrics(actual_eval, pred_eval)
            m_info["eval_cycles_count"] = n_eval
        else:
            m_info["eval_metrics"] = None
            m_info["eval_cycles_count"] = 0

    return {
        "origin_cycle": origin_cycle,
        "horizon": horizon,
        "q_init": q_init,
        "train_df": train_sub,
        "future_actual_df": future_actual_df,
        "future_cycles": future_cycles,
        "models": models_out,
    }


@st.cache_data(show_spinner=False)
def run_interactive_rul_analysis(
    origin_cycle: int = 60,
    eol_threshold_ah: float = 1.40,
    max_horizon: int = 115,
) -> Dict[str, Any]:
    """
    Execute causal RUL threshold-crossing estimation from `origin_cycle` using only `Cycles 1..origin_cycle`.
    Reuses `src.models.estimate_rul_threshold_crossing` and adds Damped Holt for completeness.
    """
    df = load_processed_battery_data()
    base_rul = estimate_rul_threshold_crossing(
        df,
        origin_cycle=origin_cycle,
        max_horizon=max_horizon,
        eol_capacity_ah=eol_threshold_ah,
        arima_order=(0, 1, 1),
    )

    cap = df["discharge_capacity_ah"].values.astype(float)
    train_sub = cap[:origin_cycle]
    future_cycles = base_rul["future_cycles"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d_alpha, d_beta, d_phi = _tune_holt_causal(train_sub, val_size=15, damped=True)
        m_damped = Holt(train_sub, damped_trend=True, initialization_method="estimated").fit(
            smoothing_level=d_alpha, smoothing_trend=d_beta, damping_trend=d_phi, optimized=False
        )
        pred_damped = np.asarray(m_damped.forecast(max_horizon), dtype=float)

    idx_dm = np.where(pred_damped < eol_threshold_ah)[0]
    if len(idx_dm) > 0:
        damped_eol_cycle = int(future_cycles[idx_dm[0]])
        damped_rul = int(damped_eol_cycle - origin_cycle)
        damped_err = (
            int(damped_rul - base_rul["actual_rul_cycles"])
            if base_rul["actual_rul_cycles"] is not None
            else None
        )
    else:
        damped_eol_cycle, damped_rul, damped_err = None, None, None

    # Check whether threshold was already crossed at or before origin_cycle
    already_crossed_in_history = bool((train_sub < eol_threshold_ah).any())
    retrospective_available = bool(
        (base_rul["actual_eol_cycle"] is not None) and (base_rul["actual_eol_cycle"] > origin_cycle)
    )

    base_rul.update(
        {
            "damped_causal_alpha": float(d_alpha),
            "damped_causal_beta": float(d_beta),
            "damped_causal_phi": float(d_phi),
            "damped_predicted_eol_cycle": damped_eol_cycle,
            "damped_predicted_rul_cycles": damped_rul,
            "damped_rul_error_cycles": damped_err,
            "damped_trajectory": pred_damped,
            "already_crossed_in_history": already_crossed_in_history,
            "retrospective_available": retrospective_available,
        }
    )
    return base_rul

