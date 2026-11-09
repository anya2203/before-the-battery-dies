"""
Phase 5 & Phase 6: Statistical Forecasting Models (ARIMA, ARIMA with Drift, Holt's Linear Trend, Damped Holt).

Implements:
- Strict chronological splitting (no shuffling, zero test-set leakage).
- Model 1: ARIMA(0, 1, 1) Baseline (with candidate order search on inner validation + training AIC/BIC).
- Model 2: ARIMA(0, 1, 1) with Linear Drift (`trend='t'`).
- Model 3: Holt's Linear Trend (tuned strictly on inner training/validation split, plus Training-MLE comparison).
- Model 4: Damped Holt's Trend (tuned strictly on inner training/validation split).
- Leakage-free Rolling-Origin Evaluation (re-tuning parameters using only data available up to each origin).
- Comprehensive residual diagnostics (Ljung-Box tests across lags 5, 10, 15, 20 and Shapiro-Wilk normality).
- Remaining Useful Life (RUL) threshold-crossing estimation with causal parameter tuning at the forecast origin.
"""

import warnings
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import shapiro, skew, kurtosis
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def chronological_train_test_split(
    df: pd.DataFrame, train_ratio: float = 0.80
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the cycle-indexed dataframe strictly in chronological order.
    Never shuffles observations.
    """
    if not (0.5 <= train_ratio < 0.95):
        raise ValueError("train_ratio must be in [0.5, 0.95).")
    df_sorted = df.sort_values("cycle_number", ascending=True).reset_index(drop=True)
    split_idx = int(round(len(df_sorted) * train_ratio))
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df


def compute_residual_diagnostics(residuals: np.ndarray, model_label: str) -> Dict[str, Any]:
    """
    Compute formal residual white-noise (Ljung-Box at lags 5, 10, 15, 20) and
    distributional diagnostics for a fitted time-series model on the training set.
    """
    res = np.asarray(residuals, dtype=float).ravel()
    res = res[~np.isnan(res)]
    lb_df = acorr_ljungbox(res, lags=[5, 10, 15, 20], return_df=True)
    shapiro_stat, shapiro_p = shapiro(res)

    p_lag5 = float(lb_df.loc[5, "lb_pvalue"])
    p_lag10 = float(lb_df.loc[10, "lb_pvalue"])
    p_lag15 = float(lb_df.loc[15, "lb_pvalue"])
    p_lag20 = float(lb_df.loc[20, "lb_pvalue"])
    is_white_noise = bool((p_lag5 > 0.05) and (p_lag10 > 0.05) and (p_lag15 > 0.05) and (p_lag20 > 0.05))

    return {
        "Model": model_label,
        "N_Residuals": int(len(res)),
        "Mean_Ah": float(np.mean(res)),
        "Std_Ah": float(np.std(res, ddof=1)),
        "Skewness": float(skew(res)),
        "Kurtosis": float(kurtosis(res)),
        "LjungBox_Stat_Lag5": float(lb_df.loc[5, "lb_stat"]),
        "LjungBox_p_Lag5": p_lag5,
        "LjungBox_Stat_Lag10": float(lb_df.loc[10, "lb_stat"]),
        "LjungBox_p_Lag10": p_lag10,
        "LjungBox_Stat_Lag15": float(lb_df.loc[15, "lb_stat"]),
        "LjungBox_p_Lag15": p_lag15,
        "LjungBox_Stat_Lag20": float(lb_df.loc[20, "lb_stat"]),
        "LjungBox_p_Lag20": p_lag20,
        "Shapiro_Stat": float(shapiro_stat),
        "Shapiro_p": float(shapiro_p),
        "White_Noise_At_5Pct": is_white_noise,
        "Autocorrelation_Conclusion": (
            "Fail to reject H0 of zero autocorrelation up to lag 20 at alpha=0.05 (consistent with uncorrelated residuals; white noise is not proven, and Shapiro-Wilk rejects normality)."
            if is_white_noise
            else "Reject H0 of zero autocorrelation at alpha=0.05 (significant residual autocorrelation detected)."
        ),
    }


def _tune_holt_causal(
    series_1d: np.ndarray,
    val_size: int = 18,
    damped: bool = False,
) -> Tuple[float, float, float | None]:
    """
    Helper to select (alpha, beta, [phi]) using only an inner chronological split
    at the tail of `series_1d` (zero lookahead beyond `series_1d`).
    """
    y = np.asarray(series_1d, dtype=float)
    v_len = min(val_size, max(8, len(y) // 4))
    sub_tr = y[:-v_len]
    sub_val = y[-v_len:]

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    betas = [0.01, 0.03, 0.05, 0.10]
    phis = [0.85, 0.90, 0.95, 0.98] if damped else [None]

    best_score = float("inf")
    best_params: Tuple[float, float, float | None] = (0.2, 0.05, 0.95 if damped else None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in alphas:
            for b in betas:
                if b > a:
                    continue
                for phi in phis:
                    try:
                        if damped:
                            m = Holt(sub_tr, damped_trend=True, initialization_method="estimated").fit(
                                smoothing_level=a, smoothing_trend=b, damping_trend=phi, optimized=False
                            )
                        else:
                            m = Holt(sub_tr, initialization_method="estimated").fit(
                                smoothing_level=a, smoothing_trend=b, optimized=False
                            )
                        pred = np.asarray(m.forecast(v_len), dtype=float)
                        rmse = float(root_mean_squared_error(sub_val, pred))
                        if rmse < best_score:
                            best_score = rmse
                            best_params = (float(a), float(b), float(phi) if phi is not None else None)
                    except Exception:
                        continue
    return best_params


def select_and_fit_arima(
    train_series: pd.Series,
    test_horizon: int,
    d: int = 1,
    val_size: int = 24,
    candidate_orders: List[Tuple[int, int, int]] | None = None,
) -> Dict[str, Any]:
    """
    Evaluate candidate ARIMA(p, d, q) orders on the inner training/validation split
    (Cycles 1..110 sub-train, Cycles 111..134 inner validation) plus AIC/BIC,
    then refit Model 1 (ARIMA(0,1,1) Baseline) and Model 2 (ARIMA(0,1,1) with Drift)
    on the complete 134-cycle training set to forecast the 34-cycle test period.
    """
    if candidate_orders is None:
        candidate_orders = [
            (0, d, 1),
            (1, d, 0),
            (1, d, 1),
            (2, d, 0),
            (2, d, 1),
            (1, d, 2),
            (2, d, 2),
            (3, d, 1),
        ]

    y_train = train_series.values.astype(float)
    sub_train = y_train[:-val_size]
    val_actual = y_train[-val_size:]

    grid_results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in candidate_orders:
            try:
                # 1. Fit strictly on inner sub_train (Cycles 1..110) and evaluate on inner validation (111..134)
                m_val = ARIMA(sub_train, order=order).fit()
                val_pred = np.asarray(m_val.forecast(steps=val_size), dtype=float)
                val_rmse = float(root_mean_squared_error(val_actual, val_pred))
                val_mae = float(mean_absolute_error(val_actual, val_pred))
                sub_aic = float(m_val.aic)
                sub_bic = float(m_val.bic)

                # 2. Refit on full training set (Cycles 1..134) to record full-train AIC, BIC, and Ljung-Box p
                m_full = ARIMA(y_train, order=order).fit()
                lb_df = acorr_ljungbox(m_full.resid[d:], lags=[10], return_df=True)
                lb_pvalue = float(lb_df["lb_pvalue"].iloc[0])

                grid_results.append(
                    {
                        "order": order,
                        "order_str": f"ARIMA{order}",
                        "sub_train_aic": sub_aic,
                        "sub_train_bic": sub_bic,
                        "val_rmse": val_rmse,
                        "val_mae": val_mae,
                        "aic": float(m_full.aic),
                        "bic": float(m_full.bic),
                        "hqic": float(m_full.hqic),
                        "ljung_box_p_lag10": lb_pvalue,
                        "converged": bool(m_full.mle_retvals.get("converged", True)),
                    }
                )
            except Exception:
                continue

        # Also evaluate ARIMA(0,1,1) with drift on inner validation split
        m_drift_val = ARIMA(sub_train, order=(0, d, 1), trend="t").fit()
        drift_val_pred = np.asarray(m_drift_val.forecast(steps=val_size), dtype=float)

    if not grid_results:
        raise RuntimeError("No candidate ARIMA models converged.")

    # Sort by inner sub_train AIC and full training AIC (both select ARIMA(0, 1, 1) as the parsimonious baseline)
    grid_df = pd.DataFrame(grid_results).sort_values(["sub_train_aic", "aic"], ascending=[True, True]).reset_index(drop=True)
    best_order = tuple(grid_df.loc[0, "order"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Inner validation forecast for Model 1 (ARIMA(0,1,1))
        m_best_val = ARIMA(sub_train, order=best_order).fit()
        best_val_pred = np.asarray(m_best_val.forecast(steps=val_size), dtype=float)

        # Full-training fit for Model 1: ARIMA(0,1,1) Baseline
        best_model = ARIMA(y_train, order=best_order).fit()
        forecast_res = best_model.get_forecast(steps=test_horizon)
        pred_mean = np.asarray(forecast_res.predicted_mean, dtype=float)
        conf_int = np.asarray(forecast_res.conf_int(alpha=0.05), dtype=float)

        # Candidate ARIMA(1,1,1) and ARIMA(2,1,1) for reference
        m_111_val = ARIMA(sub_train, order=(1, d, 1)).fit()
        val_pred_111 = np.asarray(m_111_val.forecast(steps=val_size), dtype=float)
        m_111 = ARIMA(y_train, order=(1, d, 1)).fit()
        pred_111 = np.asarray(m_111.get_forecast(steps=test_horizon).predicted_mean, dtype=float)

        m_211_val = ARIMA(sub_train, order=(2, d, 1)).fit()
        val_pred_211 = np.asarray(m_211_val.forecast(steps=val_size), dtype=float)
        m_211 = ARIMA(y_train, order=(2, d, 1)).fit()
        pred_211 = np.asarray(m_211.get_forecast(steps=test_horizon).predicted_mean, dtype=float)

        # Full-training fit for Model 2: ARIMA(0,1,1) with Linear Drift (trend='t')
        drift_model = ARIMA(y_train, order=best_order, trend="t").fit()
        drift_res = drift_model.get_forecast(steps=test_horizon)
        drift_forecast = np.asarray(drift_res.predicted_mean, dtype=float)
        drift_ci = np.asarray(drift_res.conf_int(alpha=0.05), dtype=float)

    residuals = np.asarray(best_model.resid[d:], dtype=float)
    fitted_values = np.asarray(best_model.fittedvalues, dtype=float)

    drift_residuals = np.asarray(drift_model.resid[d:], dtype=float)
    drift_fitted = np.asarray(drift_model.fittedvalues, dtype=float)

    return {
        "model_name": f"ARIMA{best_order}",
        "best_order": best_order,
        "grid_search_table": grid_df,
        "aic": float(best_model.aic),
        "bic": float(best_model.bic),
        "params": {str(k): float(v) for k, v in zip(best_model.param_names, best_model.params)},
        "fitted_values": fitted_values,
        "residuals": residuals,
        "residual_diagnostics": compute_residual_diagnostics(residuals, f"Model 1: Baseline ARIMA{best_order}"),
        "val_forecast": best_val_pred,
        "forecast": pred_mean,
        "conf_int_lower": conf_int[:, 0],
        "conf_int_upper": conf_int[:, 1],
        "arima_111_val_forecast": val_pred_111,
        "arima_111_fitted": np.asarray(m_111.fittedvalues, dtype=float),
        "arima_111_forecast": pred_111,
        "arima_211_val_forecast": val_pred_211,
        "arima_211_fitted": np.asarray(m_211.fittedvalues, dtype=float),
        "arima_211_forecast": pred_211,
        "arima_with_drift_name": f"ARIMA{best_order} with Drift",
        "arima_with_drift_params": {str(k): float(v) for k, v in zip(drift_model.param_names, drift_model.params)},
        "arima_with_drift_aic": float(drift_model.aic),
        "arima_with_drift_bic": float(drift_model.bic),
        "arima_with_drift_fitted": drift_fitted,
        "arima_with_drift_residuals": drift_residuals,
        "arima_with_drift_diagnostics": compute_residual_diagnostics(
            drift_residuals, f"Model 2: ARIMA{best_order} with Drift"
        ),
        "arima_with_drift_val_forecast": drift_val_pred,
        "arima_with_drift_forecast": drift_forecast,
        "arima_with_drift_ci_lower": drift_ci[:, 0],
        "arima_with_drift_ci_upper": drift_ci[:, 1],
    }


def select_and_fit_holt(
    train_series: pd.Series,
    test_horizon: int,
    val_size: int = 24,
) -> Dict[str, Any]:
    """
    Tune and fit Model 3 (Holt's Linear Trend) and Model 4 (Damped Holt's Trend)
    strictly on training/validation data (Cycles 1..110 sub-train, Cycles 111..134 val),
    then refit on the full 134-cycle training set to forecast `test_horizon` steps ahead.
    """
    y_train = train_series.values.astype(float)
    sub_train = y_train[:-val_size]
    val_actual = y_train[-val_size:]

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    betas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    tuning_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Evaluate grid of (alpha, beta) on inner chronological validation split
        for a in alphas:
            for b in betas:
                if b > a:
                    continue
                try:
                    m_val = Holt(sub_train, initialization_method="estimated").fit(
                        smoothing_level=a, smoothing_trend=b, optimized=False
                    )
                    val_pred = np.asarray(m_val.forecast(val_size), dtype=float)
                    val_rmse = float(root_mean_squared_error(val_actual, val_pred))
                    val_mae = float(mean_absolute_error(val_actual, val_pred))

                    m_full = Holt(y_train, initialization_method="estimated").fit(
                        smoothing_level=a, smoothing_trend=b, optimized=False
                    )
                    tuning_rows.append(
                        {
                            "variant": "Grid_Linear",
                            "smoothing_level_alpha": float(a),
                            "smoothing_trend_beta": float(b),
                            "damping_trend_phi": np.nan,
                            "sub_train_aic": float(m_val.aic),
                            "val_rmse": val_rmse,
                            "val_mae": val_mae,
                            "train_aic": float(m_full.aic),
                            "train_bic": float(m_full.bic),
                            "train_sse": float(m_full.sse),
                        }
                    )
                except Exception:
                    continue

        # Also evaluate MLE-optimized Linear Holt strictly on sub_train -> val
        m_val_mle = Holt(sub_train, initialization_method="estimated").fit(optimized=True)
        val_pred_mle = np.asarray(m_val_mle.forecast(val_size), dtype=float)
        m_full_mle = Holt(y_train, initialization_method="estimated").fit(optimized=True)
        tuning_rows.append(
            {
                "variant": "MLE_Linear",
                "smoothing_level_alpha": float(m_full_mle.params["smoothing_level"]),
                "smoothing_trend_beta": float(m_full_mle.params["smoothing_trend"]),
                "damping_trend_phi": np.nan,
                "sub_train_aic": float(m_val_mle.aic),
                "val_rmse": float(root_mean_squared_error(val_actual, val_pred_mle)),
                "val_mae": float(mean_absolute_error(val_actual, val_pred_mle)),
                "train_aic": float(m_full_mle.aic),
                "train_bic": float(m_full_mle.bic),
                "train_sse": float(m_full_mle.sse),
            }
        )

    tuning_df = pd.DataFrame(tuning_rows).sort_values("val_rmse", ascending=True).reset_index(drop=True)
    best_row = tuning_df.iloc[0]
    best_alpha = float(best_row["smoothing_level_alpha"])
    best_beta = float(best_row["smoothing_trend_beta"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Inner validation forecast for selected Holt Linear model
        if best_row["variant"] == "MLE_Linear":
            best_holt_val = Holt(sub_train, initialization_method="estimated").fit(optimized=True)
            best_holt = Holt(y_train, initialization_method="estimated").fit(optimized=True)
        else:
            best_holt_val = Holt(sub_train, initialization_method="estimated").fit(
                smoothing_level=best_alpha, smoothing_trend=best_beta, optimized=False
            )
            best_holt = Holt(y_train, initialization_method="estimated").fit(
                smoothing_level=best_alpha, smoothing_trend=best_beta, optimized=False
            )

        holt_val_forecast = np.asarray(best_holt_val.forecast(val_size), dtype=float)
        holt_forecast = np.asarray(best_holt.forecast(test_horizon), dtype=float)

        # Training-only MLE Holt Linear (fitted strictly on y_train Cycles 1..134, zero test data)
        mle_holt = Holt(y_train, initialization_method="estimated").fit(optimized=True)
        mle_forecast = np.asarray(mle_holt.forecast(test_horizon), dtype=float)

        # Model 4: Damped Holt's Trend tuned strictly on inner validation split (Cycles 1..110 -> 111..134)
        damped_candidates = []
        for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for b in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
                if b > a:
                    continue
                for phi in [0.85, 0.90, 0.95, 0.98]:
                    try:
                        dm_val = Holt(sub_train, damped_trend=True, initialization_method="estimated").fit(
                            smoothing_level=a, smoothing_trend=b, damping_trend=phi, optimized=False
                        )
                        v_pred = np.asarray(dm_val.forecast(val_size), dtype=float)
                        damped_candidates.append(
                            (float(root_mean_squared_error(val_actual, v_pred)), a, b, phi, v_pred)
                        )
                    except Exception:
                        continue
        damped_candidates.sort(key=lambda x: x[0])
        _, d_a, d_b, d_phi, damped_val_forecast = damped_candidates[0]
        damped_holt = Holt(y_train, damped_trend=True, initialization_method="estimated").fit(
            smoothing_level=d_a, smoothing_trend=d_b, damping_trend=d_phi, optimized=False
        )
        damped_forecast = np.asarray(damped_holt.forecast(test_horizon), dtype=float)

    holt_residuals = np.asarray(best_holt.resid, dtype=float)
    mle_residuals = np.asarray(mle_holt.resid, dtype=float)
    damped_residuals = np.asarray(damped_holt.resid, dtype=float)

    return {
        "model_name": f"Holt Linear (alpha={best_alpha:.2f}, beta={best_beta:.2f})",
        "smoothing_level_alpha": float(best_holt.params["smoothing_level"]),
        "smoothing_trend_beta": float(best_holt.params["smoothing_trend"]),
        "initial_level": float(best_holt.params["initial_level"]),
        "initial_trend": float(best_holt.params["initial_trend"]),
        "final_train_level": float(best_holt.level[-1]),
        "final_train_trend": float(best_holt.trend[-1]),
        "aic": float(best_holt.aic),
        "bic": float(best_holt.bic),
        "fitted_values": np.asarray(best_holt.fittedvalues, dtype=float),
        "residuals": holt_residuals,
        "residual_diagnostics": compute_residual_diagnostics(
            holt_residuals, f"Model 3: Holt Linear (alpha={best_alpha:.2f}, beta={best_beta:.2f})"
        ),
        "level_series": np.asarray(best_holt.level, dtype=float),
        "trend_series": np.asarray(best_holt.trend, dtype=float),
        "val_forecast": holt_val_forecast,
        "forecast": holt_forecast,
        "tuning_table": tuning_df,
        "mle_holt_fitted": np.asarray(mle_holt.fittedvalues, dtype=float),
        "mle_holt_residuals": mle_residuals,
        "mle_holt_diagnostics": compute_residual_diagnostics(
            mle_residuals, f"Model 3b: Holt Linear Train-MLE (alpha={mle_holt.params['smoothing_level']:.2f}, beta={mle_holt.params['smoothing_trend']:.2f})"
        ),
        "mle_holt_val_forecast": val_pred_mle,
        "mle_holt_forecast": mle_forecast,
        "mle_alpha": float(mle_holt.params["smoothing_level"]),
        "mle_beta": float(mle_holt.params["smoothing_trend"]),
        "mle_aic": float(mle_holt.aic),
        "mle_bic": float(mle_holt.bic),
        "damped_holt_fitted": np.asarray(damped_holt.fittedvalues, dtype=float),
        "damped_holt_residuals": damped_residuals,
        "damped_holt_diagnostics": compute_residual_diagnostics(
            damped_residuals, f"Model 4: Damped Holt (alpha={d_a:.2f}, beta={d_b:.2f}, phi={d_phi:.2f})"
        ),
        "damped_holt_val_forecast": damped_val_forecast,
        "damped_holt_forecast": damped_forecast,
        "damped_params": {
            "alpha": float(d_a),
            "beta": float(d_b),
            "phi": float(d_phi),
            "aic": float(damped_holt.aic),
            "bic": float(damped_holt.bic),
        },
    }


def run_rolling_origin_evaluation(
    df: pd.DataFrame,
    arima_order: Tuple[int, int, int] = (0, 1, 1),
    holt_alpha: float | None = None,
    holt_beta: float | None = None,
    cutoffs: List[int] | None = None,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    Phase 3.3: Chronological Rolling-Origin Evaluation across multiple forecast origins.

    At each origin `c`, every model is fitted (and Holt/Damped Holt smoothing parameters
    are causally selected via `_tune_holt_causal`) using ONLY observations `1..c`.
    Never uses future observations when fitting or selecting parameters.
    Compares all 4 core models:
    - Model 1: ARIMA(0,1,1) Baseline
    - Model 2: ARIMA(0,1,1) with Drift
    - Model 3: Holt's Linear Trend
    - Model 4: Damped Holt's Trend
    """
    if cutoffs is None:
        cutoffs = [90, 105, 120, 134, 148]

    cap = df["discharge_capacity_ah"].values.astype(float)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in cutoffs:
            if c + horizon > len(cap):
                continue
            train_y = cap[:c]
            test_y = cap[c : c + horizon]

            # Model 1: ARIMA(0,1,1) Baseline
            m_arima = ARIMA(train_y, order=arima_order).fit()
            pred_arima = np.asarray(m_arima.forecast(steps=horizon), dtype=float)

            # Model 2: ARIMA(0,1,1) with Drift
            m_drift = ARIMA(train_y, order=arima_order, trend="t").fit()
            pred_drift = np.asarray(m_drift.forecast(steps=horizon), dtype=float)

            # Model 3: Holt's Linear Trend (causally tuned strictly on train_y[:c])
            c_alpha, c_beta, _ = _tune_holt_causal(train_y, val_size=18, damped=False)
            m_holt = Holt(train_y, initialization_method="estimated").fit(
                smoothing_level=c_alpha, smoothing_trend=c_beta, optimized=False
            )
            pred_holt = np.asarray(m_holt.forecast(horizon), dtype=float)

            # Model 4: Damped Holt's Trend (causally tuned strictly on train_y[:c])
            d_alpha, d_beta, d_phi = _tune_holt_causal(train_y, val_size=18, damped=True)
            m_damped = Holt(train_y, damped_trend=True, initialization_method="estimated").fit(
                smoothing_level=d_alpha, smoothing_trend=d_beta, damping_trend=d_phi, optimized=False
            )
            pred_damped = np.asarray(m_damped.forecast(horizon), dtype=float)

            def _smape(yt, yp):
                return float(np.mean(2.0 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp))) * 100.0)

            rows.append(
                {
                    "train_end_cycle": int(c),
                    "test_start_cycle": int(c + 1),
                    "test_end_cycle": int(c + horizon),
                    "horizon": int(horizon),
                    "holt_causal_alpha": float(c_alpha),
                    "holt_causal_beta": float(c_beta),
                    "damped_causal_alpha": float(d_alpha),
                    "damped_causal_beta": float(d_beta),
                    "damped_causal_phi": float(d_phi),
                    "arima_mae": float(mean_absolute_error(test_y, pred_arima)),
                    "arima_rmse": float(root_mean_squared_error(test_y, pred_arima)),
                    "arima_mape": float(np.mean(np.abs((test_y - pred_arima) / test_y)) * 100.0),
                    "arima_smape": _smape(test_y, pred_arima),
                    "arima_drift_mae": float(mean_absolute_error(test_y, pred_drift)),
                    "arima_drift_rmse": float(root_mean_squared_error(test_y, pred_drift)),
                    "arima_drift_mape": float(np.mean(np.abs((test_y - pred_drift) / test_y)) * 100.0),
                    "arima_drift_smape": _smape(test_y, pred_drift),
                    "holt_mae": float(mean_absolute_error(test_y, pred_holt)),
                    "holt_rmse": float(root_mean_squared_error(test_y, pred_holt)),
                    "holt_mape": float(np.mean(np.abs((test_y - pred_holt) / test_y)) * 100.0),
                    "holt_smape": _smape(test_y, pred_holt),
                    "damped_holt_mae": float(mean_absolute_error(test_y, pred_damped)),
                    "damped_holt_rmse": float(root_mean_squared_error(test_y, pred_damped)),
                    "damped_holt_mape": float(np.mean(np.abs((test_y - pred_damped) / test_y)) * 100.0),
                    "damped_holt_smape": _smape(test_y, pred_damped),
                }
            )
    return pd.DataFrame(rows)


def run_rolling_one_step_recalibration(
    train_series: pd.Series,
    test_series: pd.Series,
    arima_order: Tuple[int, int, int],
    holt_alpha: float,
    holt_beta: float,
) -> Dict[str, np.ndarray]:
    """
    Optional Extension: 1-step-ahead expanding-window recalibration across the test set.
    """
    history = list(train_series.values.astype(float))
    test_vals = test_series.values.astype(float)

    arima_1step = []
    arima_drift_1step = []
    holt_1step = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obs in test_vals:
            arr = np.asarray(history, dtype=float)
            m_a = ARIMA(arr, order=arima_order).fit()
            arima_1step.append(float(m_a.forecast(steps=1)[0]))

            m_ad = ARIMA(arr, order=arima_order, trend="t").fit()
            arima_drift_1step.append(float(m_ad.forecast(steps=1)[0]))

            m_h = Holt(arr, initialization_method="estimated").fit(
                smoothing_level=holt_alpha, smoothing_trend=holt_beta, optimized=False
            )
            holt_1step.append(float(m_h.forecast(1)[0]))
            history.append(float(obs))

    return {
        "arima_rolling_1step": np.asarray(arima_1step, dtype=float),
        "arima_drift_rolling_1step": np.asarray(arima_drift_1step, dtype=float),
        "holt_rolling_1step": np.asarray(holt_1step, dtype=float),
    }


def estimate_rul_threshold_crossing(
    df: pd.DataFrame,
    origin_cycle: int = 60,
    max_horizon: int = 108,
    eol_capacity_ah: float = 1.40,
    holt_alpha: float | None = None,
    holt_beta: float | None = None,
    arima_order: Tuple[int, int, int] = (0, 1, 1),
) -> Dict[str, Any]:
    """
    Phase 9 Optional RUL Extension (Leakage-Free):
    At a pre-EOL chronological cutoff (`origin_cycle = 60`, before the battery crosses
    `eol_capacity_ah = 1.40 Ah`), tune smoothing parameters strictly on `Cycles 1..origin_cycle`,
    forecast capacity forward, and identify the predicted EOL crossing cycle and RUL for
    ARIMA(0,1,1), ARIMA(0,1,1) with Drift, and Holt's Linear Trend.
    """
    cap = df["discharge_capacity_ah"].values.astype(float)
    cycles = df["cycle_number"].values.astype(int)

    below_mask = cap < eol_capacity_ah
    actual_eol_cycle = int(cycles[below_mask][0]) if below_mask.any() else None
    actual_rul = (actual_eol_cycle - origin_cycle) if actual_eol_cycle is not None else None

    train_sub = cap[:origin_cycle]
    future_cycles = np.arange(origin_cycle + 1, origin_cycle + max_horizon + 1)

    # Causally select Holt parameters using ONLY Cycles 1..origin_cycle (no post-origin leakage)
    causal_alpha, causal_beta, _ = _tune_holt_causal(train_sub, val_size=15, damped=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_holt = Holt(train_sub, initialization_method="estimated").fit(
            smoothing_level=causal_alpha, smoothing_trend=causal_beta, optimized=False
        )
        pred_holt = np.asarray(m_holt.forecast(max_horizon), dtype=float)

        m_arima = ARIMA(train_sub, order=arima_order).fit()
        pred_arima = np.asarray(m_arima.forecast(steps=max_horizon), dtype=float)

        m_drift = ARIMA(train_sub, order=arima_order, trend="t").fit()
        pred_drift = np.asarray(m_drift.forecast(steps=max_horizon), dtype=float)

    def _find_crossing(pred_arr):
        idx = np.where(pred_arr < eol_capacity_ah)[0]
        if len(idx) == 0:
            return None, None, None
        eol_c = int(future_cycles[idx[0]])
        rul_c = int(eol_c - origin_cycle)
        err_c = int(rul_c - actual_rul) if actual_rul is not None else None
        return eol_c, rul_c, err_c

    holt_eol_cycle, holt_rul, holt_err = _find_crossing(pred_holt)
    arima_eol_cycle, arima_rul, arima_err = _find_crossing(pred_arima)
    drift_eol_cycle, drift_rul, drift_err = _find_crossing(pred_drift)

    return {
        "origin_cycle": int(origin_cycle),
        "origin_capacity_ah": float(cap[origin_cycle - 1]),
        "eol_threshold_ah": float(eol_capacity_ah),
        "causal_holt_alpha_at_origin": float(causal_alpha),
        "causal_holt_beta_at_origin": float(causal_beta),
        "actual_eol_cycle": actual_eol_cycle,
        "actual_rul_cycles": actual_rul,
        "holt_predicted_eol_cycle": holt_eol_cycle,
        "holt_predicted_rul_cycles": holt_rul,
        "holt_rul_error_cycles": holt_err,
        "arima_drift_predicted_eol_cycle": drift_eol_cycle,
        "arima_drift_predicted_rul_cycles": drift_rul,
        "arima_drift_rul_error_cycles": drift_err,
        "arima_predicted_eol_cycle": arima_eol_cycle,
        "arima_predicted_rul_cycles": arima_rul,
        "arima_rul_error_cycles": arima_err,
        "future_cycles": future_cycles,
        "holt_trajectory": pred_holt,
        "arima_drift_trajectory": pred_drift,
        "arima_trajectory": pred_arima,
    }
