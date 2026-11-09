"""
Phase 7: Fair Model Evaluation Module (Strengthened for Phase 2 & Phase 3).

Implements:
- Standard forecasting metrics: MAE, RMSE, MAPE, and sMAPE on identical held-out observations.
- Strict separation of Training (Cycles 1-134), Inner Validation (Cycles 111-134),
  Held-Out Multi-Step Test (Cycles 135-168, h=34), and Online 1-Step Recalibration (h=1).
- Forecast horizon analysis (h = 5, 10, 20, 34 cycles) saved to `results/metrics/horizon_comparison.csv`.
- Rolling-origin summary statistics (mean and standard deviation of MAE, RMSE, MAPE, sMAPE)
  saved to `results/metrics/rolling_origin_metrics.csv`.
- Formal residual diagnostics table saved to `results/metrics/residual_diagnostics.csv`.
"""

from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def calculate_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> Dict[str, float]:
    """
    Calculate MAE (Ah), RMSE (Ah), MAPE (%), and sMAPE (%) between actual and predicted values.

    Raises ValueError on mismatched lengths, empty arrays, NaNs, or zero true values.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    if yt.size == 0 or yp.size == 0:
        raise ValueError("Input arrays for metric calculation must not be empty.")
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch between y_true {yt.shape} and y_pred {yp.shape}.")
    if np.isnan(yt).any() or np.isnan(yp).any():
        raise ValueError("NaN values are not allowed in y_true or y_pred.")
    if np.any(yt == 0.0):
        raise ValueError("y_true contains zero values; MAPE is undefined.")

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(root_mean_squared_error(yt, yp))
    mape = float(np.mean(np.abs((yt - yp) / yt)) * 100.0)
    denom = np.abs(yt) + np.abs(yp)
    smape = float(np.mean(np.where(denom > 0, 2.0 * np.abs(yt - yp) / denom, 0.0)) * 100.0)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
    }


def evaluate_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    arima_res: Dict[str, Any],
    holt_res: Dict[str, Any],
    rolling_1step_res: Dict[str, np.ndarray],
    rolling_origin_df: pd.DataFrame,
    rul_res: Dict[str, Any],
    results_dir: str | Path,
    val_size: int = 24,
) -> Dict[str, Any]:
    """
    Compare all models fairly across identical splits and horizons, and export
    clean CSV/JSON tables to `results/metrics/` and `results/forecasts/`.
    """
    res_path = Path(results_dir)
    metrics_dir = res_path / "metrics"
    forecasts_dir = res_path / "forecasts"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    y_train = train_df["discharge_capacity_ah"].values.astype(float)
    y_val = y_train[-val_size:]
    y_test = test_df["discharge_capacity_ah"].values.astype(float)
    q_initial = float(train_df["discharge_capacity_ah"].iloc[0])

    # 1. Assemble test-period forecast table
    forecast_df = pd.DataFrame(
        {
            "battery_id": test_df["battery_id"].values,
            "cycle_number": test_df["cycle_number"].values,
            "operation_cycle_id": test_df["operation_cycle_id"].values,
            "actual_capacity_ah": y_test,
            "actual_soh_percent": test_df["soh_percent"].values,
            "arima_forecast_ah": arima_res["forecast"],
            "arima_ci_lower_ah": arima_res["conf_int_lower"],
            "arima_ci_upper_ah": arima_res["conf_int_upper"],
            "arima_drift_forecast_ah": arima_res["arima_with_drift_forecast"],
            "arima_drift_ci_lower_ah": arima_res["arima_with_drift_ci_lower"],
            "arima_drift_ci_upper_ah": arima_res["arima_with_drift_ci_upper"],
            "holt_linear_forecast_ah": holt_res["forecast"],
            "holt_mle_forecast_ah": holt_res["mle_holt_forecast"],
            "holt_damped_forecast_ah": holt_res["damped_holt_forecast"],
            "arima_rolling_1step_ah": rolling_1step_res["arima_rolling_1step"],
            "arima_drift_rolling_1step_ah": rolling_1step_res["arima_drift_rolling_1step"],
            "holt_rolling_1step_ah": rolling_1step_res["holt_rolling_1step"],
        }
    )

    # Derived SOH forecasts and errors (e_t = actual - predicted)
    forecast_df["arima_forecast_soh_percent"] = (forecast_df["arima_forecast_ah"] / q_initial) * 100.0
    forecast_df["arima_drift_forecast_soh_percent"] = (forecast_df["arima_drift_forecast_ah"] / q_initial) * 100.0
    forecast_df["holt_linear_forecast_soh_percent"] = (forecast_df["holt_linear_forecast_ah"] / q_initial) * 100.0
    forecast_df["holt_damped_forecast_soh_percent"] = (forecast_df["holt_damped_forecast_ah"] / q_initial) * 100.0

    forecast_df["arima_error_ah"] = forecast_df["actual_capacity_ah"] - forecast_df["arima_forecast_ah"]
    forecast_df["arima_drift_error_ah"] = forecast_df["actual_capacity_ah"] - forecast_df["arima_drift_forecast_ah"]
    forecast_df["holt_linear_error_ah"] = forecast_df["actual_capacity_ah"] - forecast_df["holt_linear_forecast_ah"]
    forecast_df["holt_mle_error_ah"] = forecast_df["actual_capacity_ah"] - forecast_df["holt_mle_forecast_ah"]
    forecast_df["holt_damped_error_ah"] = forecast_df["actual_capacity_ah"] - forecast_df["holt_damped_forecast_ah"]

    forecast_csv = forecasts_dir / "test_forecasts_comparison.csv"
    forecast_df.to_csv(forecast_csv, index=False)

    # 2. Primary Held-Out Test Set Multi-Step Comparison Table (`model_comparison_metrics.csv`)
    # Strictly compares multi-step forecasts on the identical 34-cycle held-out test set
    d_phi = holt_res["damped_params"]["phi"]
    d_alpha = holt_res["damped_params"]["alpha"]
    d_beta = holt_res["damped_params"]["beta"]

    core_models_spec = [
        (
            "Model 1: ARIMA(0, 1, 1) — Baseline",
            y_train[1:],
            arima_res["fitted_values"][1:],
            arima_res["val_forecast"],
            arima_res["forecast"],
        ),
        (
            "Model 2: ARIMA(0, 1, 1) with Drift",
            y_train[1:],
            arima_res["arima_with_drift_fitted"][1:],
            arima_res["arima_with_drift_val_forecast"],
            arima_res["arima_with_drift_forecast"],
        ),
        (
            f"Model 3: Holt's Linear Trend (Val-Tuned: alpha={holt_res['smoothing_level_alpha']:.2f}, beta={holt_res['smoothing_trend_beta']:.2f})",
            y_train,
            holt_res["fitted_values"],
            holt_res["val_forecast"],
            holt_res["forecast"],
        ),
        (
            f"Model 3b: Holt's Linear Trend (Train-MLE: alpha={holt_res['mle_alpha']:.2f}, beta={holt_res['mle_beta']:.2f})",
            y_train,
            holt_res["mle_holt_fitted"],
            holt_res["mle_holt_val_forecast"],
            holt_res["mle_holt_forecast"],
        ),
        (
            f"Model 4: Damped Holt's Trend (Val-Tuned: alpha={d_alpha:.2f}, beta={d_beta:.2f}, phi={d_phi:.2f})",
            y_train,
            holt_res["damped_holt_fitted"],
            holt_res["damped_holt_val_forecast"],
            holt_res["damped_holt_forecast"],
        ),
        (
            "Candidate ARIMA(1, 1, 1) (Multi-Step)",
            y_train[1:],
            arima_res["arima_111_fitted"][1:],
            arima_res["arima_111_val_forecast"],
            arima_res["arima_111_forecast"],
        ),
        (
            "Candidate ARIMA(2, 1, 1) (Multi-Step)",
            y_train[1:],
            arima_res["arima_211_fitted"][1:],
            arima_res["arima_211_val_forecast"],
            arima_res["arima_211_forecast"],
        ),
    ]

    comparison_rows = []
    split_breakdown_rows = []

    for m_label, yt_tr, yp_tr, yp_val, yp_te in core_models_spec:
        m_tr = calculate_metrics(yt_tr, yp_tr)
        m_val = calculate_metrics(y_val, yp_val)
        m_te = calculate_metrics(y_test, yp_te)

        comparison_rows.append(
            {
                "Model": m_label,
                "Evaluation_Set": "Held-Out Test (Cycles 135-168)",
                "Forecast_Mode": "Multi-Step (34-Cycle Horizon)",
                "Horizon_Cycles": int(len(y_test)),
                "MAE_Ah": m_te["MAE"],
                "RMSE_Ah": m_te["RMSE"],
                "MAPE_Percent": m_te["MAPE"],
                "sMAPE_Percent": m_te["sMAPE"],
                "Val_RMSE_Ah": m_val["RMSE"],
                "Train_RMSE_Ah": m_tr["RMSE"],
            }
        )

        for split_name, h_len, m_dict in [
            ("Train (Cycles 1-134, 1-Step In-Sample)", 1, m_tr),
            ("Validation (Cycles 111-134, 24-Step Out-of-Sample)", int(val_size), m_val),
            ("Held-Out Test (Cycles 135-168, 34-Step Out-of-Sample)", int(len(y_test)), m_te),
        ]:
            split_breakdown_rows.append(
                {
                    "Model": m_label,
                    "Evaluation_Set": split_name,
                    "Horizon_Cycles": h_len,
                    "MAE_Ah": m_dict["MAE"],
                    "RMSE_Ah": m_dict["RMSE"],
                    "MAPE_Percent": m_dict["MAPE"],
                    "sMAPE_Percent": m_dict["sMAPE"],
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(metrics_dir / "model_comparison_metrics.csv", index=False)

    split_breakdown_df = pd.DataFrame(split_breakdown_rows)
    split_breakdown_df.to_csv(metrics_dir / "train_val_test_metrics.csv", index=False)

    # Separate table for Online 1-Step Recalibration (so h=1 is never mixed with h=34)
    online_rows = []
    for o_label, o_preds in [
        ("Online 1-Step ARIMA(0, 1, 1)", rolling_1step_res["arima_rolling_1step"]),
        ("Online 1-Step ARIMA(0, 1, 1) with Drift", rolling_1step_res["arima_drift_rolling_1step"]),
        ("Online 1-Step Holt's Linear Trend", rolling_1step_res["holt_rolling_1step"]),
    ]:
        m_o = calculate_metrics(y_test, o_preds)
        online_rows.append(
            {
                "Model": o_label,
                "Evaluation_Set": "Test Period Online Updates (Cycles 135-168)",
                "Forecast_Mode": "Expanding-Window 1-Step-Ahead Recalibration",
                "Horizon_Cycles": 1,
                "Num_Test_Steps": int(len(y_test)),
                "MAE_Ah": m_o["MAE"],
                "RMSE_Ah": m_o["RMSE"],
                "MAPE_Percent": m_o["MAPE"],
                "sMAPE_Percent": m_o["sMAPE"],
            }
        )
    online_df = pd.DataFrame(online_rows)
    online_df.to_csv(metrics_dir / "online_recalibration_metrics.csv", index=False)

    # 3. Phase 3.2: Forecast Horizon Analysis (h = 5, 10, 20, 34 cycles)
    horizons = [5, 10, 20, len(y_test)]
    horizon_rows = []
    for h in horizons:
        yt_h = y_test[:h]
        for label, col in [
            ("Model 1: ARIMA(0, 1, 1) Baseline", "arima_forecast_ah"),
            ("Model 2: ARIMA(0, 1, 1) with Drift", "arima_drift_forecast_ah"),
            (f"Model 3: Holt's Linear Trend (Val-Tuned)", "holt_linear_forecast_ah"),
            (f"Model 3b: Holt's Linear Trend (Train-MLE)", "holt_mle_forecast_ah"),
            ("Model 4: Damped Holt's Trend", "holt_damped_forecast_ah"),
        ]:
            m_h = calculate_metrics(yt_h, forecast_df[col].values[:h])
            horizon_rows.append(
                {
                    "Horizon_h": int(h),
                    "Test_Cycle_Window": f"Cycles 135-{134 + h}",
                    "Model": label,
                    "MAE_Ah": m_h["MAE"],
                    "RMSE_Ah": m_h["RMSE"],
                    "MAPE_Percent": m_h["MAPE"],
                    "sMAPE_Percent": m_h["sMAPE"],
                }
            )
    horizon_df = pd.DataFrame(horizon_rows)
    horizon_df.to_csv(metrics_dir / "horizon_comparison.csv", index=False)
    horizon_df.to_csv(metrics_dir / "horizon_sensitivity_metrics.csv", index=False)

    # 4. Phase 3.3: Rolling-Origin Evaluation Summary (`rolling_origin_metrics.csv`)
    rolling_origin_df.to_csv(metrics_dir / "rolling_origin_evaluation.csv", index=False)
    ro_summary_rows = []
    for m_name, prefix in [
        ("Model 1: ARIMA(0, 1, 1) Baseline", "arima"),
        ("Model 2: ARIMA(0, 1, 1) with Drift", "arima_drift"),
        ("Model 3: Holt's Linear Trend (Causally Tuned)", "holt"),
        ("Model 4: Damped Holt's Trend (Causally Tuned)", "damped_holt"),
    ]:
        ro_summary_rows.append(
            {
                "Model": m_name,
                "Num_Origins": int(len(rolling_origin_df)),
                "Forecast_Horizon_Cycles": int(rolling_origin_df["horizon"].iloc[0]),
                "Origins_Evaluated": ",".join(str(x) for x in rolling_origin_df["train_end_cycle"].tolist()),
                "Mean_MAE_Ah": float(rolling_origin_df[f"{prefix}_mae"].mean()),
                "Std_MAE_Ah": float(rolling_origin_df[f"{prefix}_mae"].std(ddof=1)),
                "Mean_RMSE_Ah": float(rolling_origin_df[f"{prefix}_rmse"].mean()),
                "Std_RMSE_Ah": float(rolling_origin_df[f"{prefix}_rmse"].std(ddof=1)),
                "Mean_MAPE_Percent": float(rolling_origin_df[f"{prefix}_mape"].mean()),
                "Std_MAPE_Percent": float(rolling_origin_df[f"{prefix}_mape"].std(ddof=1)),
                "Mean_sMAPE_Percent": float(rolling_origin_df[f"{prefix}_smape"].mean()),
                "Std_sMAPE_Percent": float(rolling_origin_df[f"{prefix}_smape"].std(ddof=1)),
            }
        )
    rolling_origin_summary_df = pd.DataFrame(ro_summary_rows)
    rolling_origin_summary_df.to_csv(metrics_dir / "rolling_origin_metrics.csv", index=False)

    # 5. Phase 3.4: Residual Diagnostics Table (`residual_diagnostics.csv`)
    resid_diag_df = pd.DataFrame(
        [
            arima_res["residual_diagnostics"],
            arima_res["arima_with_drift_diagnostics"],
            holt_res["residual_diagnostics"],
            holt_res["mle_holt_diagnostics"],
            holt_res["damped_holt_diagnostics"],
        ]
    )
    resid_diag_df.to_csv(metrics_dir / "residual_diagnostics.csv", index=False)

    # Save ARIMA candidate grid and Holt tuning grid
    arima_res["grid_search_table"].to_csv(metrics_dir / "arima_order_selection.csv", index=False)
    holt_res["tuning_table"].to_csv(metrics_dir / "holt_parameter_tuning.csv", index=False)

    # 6. Save JSON summary of key metrics
    summary_payload = {
        "train_cycles": [int(train_df["cycle_number"].min()), int(train_df["cycle_number"].max())],
        "validation_cycles": [int(train_df["cycle_number"].iloc[-val_size]), int(train_df["cycle_number"].max())],
        "test_cycles": [int(test_df["cycle_number"].min()), int(test_df["cycle_number"].max())],
        "n_train": int(len(train_df)),
        "n_val": int(val_size),
        "n_test": int(len(test_df)),
        "model_1_arima_baseline": {
            "order": list(arima_res["best_order"]),
            "aic": arima_res["aic"],
            "bic": arima_res["bic"],
            "metrics_val": calculate_metrics(y_val, arima_res["val_forecast"]),
            "metrics_full_test": calculate_metrics(y_test, arima_res["forecast"]),
            "residual_diagnostics": arima_res["residual_diagnostics"],
        },
        "model_2_arima_with_drift": {
            "order": list(arima_res["best_order"]),
            "aic": arima_res["arima_with_drift_aic"],
            "bic": arima_res["arima_with_drift_bic"],
            "params": arima_res["arima_with_drift_params"],
            "metrics_val": calculate_metrics(y_val, arima_res["arima_with_drift_val_forecast"]),
            "metrics_full_test": calculate_metrics(y_test, arima_res["arima_with_drift_forecast"]),
            "residual_diagnostics": arima_res["arima_with_drift_diagnostics"],
        },
        "model_3_holt_linear_val_tuned": {
            "model_name": holt_res["model_name"],
            "alpha": holt_res["smoothing_level_alpha"],
            "beta": holt_res["smoothing_trend_beta"],
            "aic": holt_res["aic"],
            "bic": holt_res["bic"],
            "metrics_val": calculate_metrics(y_val, holt_res["val_forecast"]),
            "metrics_full_test": calculate_metrics(y_test, holt_res["forecast"]),
            "residual_diagnostics": holt_res["residual_diagnostics"],
        },
        "model_3b_holt_linear_train_mle": {
            "alpha": holt_res["mle_alpha"],
            "beta": holt_res["mle_beta"],
            "aic": holt_res["mle_aic"],
            "bic": holt_res["mle_bic"],
            "metrics_val": calculate_metrics(y_val, holt_res["mle_holt_val_forecast"]),
            "metrics_full_test": calculate_metrics(y_test, holt_res["mle_holt_forecast"]),
            "residual_diagnostics": holt_res["mle_holt_diagnostics"],
        },
        "model_4_damped_holt": {
            "params": holt_res["damped_params"],
            "metrics_val": calculate_metrics(y_val, holt_res["damped_holt_val_forecast"]),
            "metrics_full_test": calculate_metrics(y_test, holt_res["damped_holt_forecast"]),
            "residual_diagnostics": holt_res["damped_holt_diagnostics"],
        },
        "rolling_origin_summary": ro_summary_rows,
        "rul_analysis": {
            k: v
            for k, v in rul_res.items()
            if k not in ("future_cycles", "holt_trajectory", "arima_drift_trajectory", "arima_trajectory")
        },
    }
    with open(metrics_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return {
        "forecast_df": forecast_df,
        "comparison_df": comparison_df,
        "split_breakdown_df": split_breakdown_df,
        "online_df": online_df,
        "horizon_df": horizon_df,
        "rolling_origin_summary_df": rolling_origin_summary_df,
        "residual_diagnostics_df": resid_diag_df,
        "summary": summary_payload,
    }
