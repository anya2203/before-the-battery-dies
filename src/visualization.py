"""
Phase 5, 6, 7 & 9 Visualization Module (Strengthened for Phase 3.4 & Phase 3.5).

Generates publication-grade diagnostic and comparative forecast figures in
`results/figures/`:
- fig09_arima_diagnostics_and_forecast.png
- fig10_holt_level_trend_and_forecast.png
- fig11_model_comparison_forecasts.png
- fig12_residual_and_error_comparison.png
- fig13_rolling_origin_and_rul.png
- fig14_comprehensive_residual_diagnostics.png (Phase 3.4: Residual time-series, distribution, ACF & Ljung-Box)
- fig15_actual_vs_predicted_six_panel.png (Phase 3.5: 6-panel actual vs. predicted comparison on consistent scales)
"""

import os
from pathlib import Path
from typing import Dict, Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / ".matplotlib"),
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)


def generate_model_and_evaluation_figures(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    arima_res: Dict[str, Any],
    holt_res: Dict[str, Any],
    eval_res: Dict[str, Any],
    rolling_origin_df: pd.DataFrame,
    rul_res: Dict[str, Any],
    figures_dir: str | Path,
) -> Dict[str, str]:
    """
    Generate figures 9 through 15 and save them to `results/figures/`.
    """
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_cycles = train_df["cycle_number"].values
    test_cycles = test_df["cycle_number"].values
    y_train = train_df["discharge_capacity_ah"].values
    y_test = test_df["discharge_capacity_ah"].values
    f_df = eval_res["forecast_df"]

    # Figure 9: ARIMA Baseline Forecast & Residual Diagnostics (including Residual Time Series)
    fig9_path = fig_dir / "fig09_arima_diagnostics_and_forecast.png"
    fig = plt.figure(figsize=(13, 8.5), dpi=180)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(train_cycles, y_train, color="#1f4e78", lw=2.0, label="Training Capacity (Cycles 1–134)")
    ax1.plot(train_cycles[1:], arima_res["fitted_values"][1:], color="#3498db", lw=1.4, ls="--", alpha=0.85, label=f"In-Sample {arima_res['model_name']} Fit")
    ax1.plot(test_cycles, y_test, color="#2c3e50", lw=2.2, label="Actual Test Capacity (Cycles 135–168)")
    ax1.plot(test_cycles, arima_res["forecast"], color="#c0392b", lw=2.4, label=f"Model 1: Baseline {arima_res['model_name']} Forecast")
    ax1.plot(test_cycles, arima_res["arima_with_drift_forecast"], color="#2980b9", lw=2.2, ls="-.", label=f"Model 2: {arima_res['model_name']} with Drift Forecast")
    ax1.fill_between(
        test_cycles,
        arima_res["conf_int_lower"],
        arima_res["conf_int_upper"],
        color="#c0392b",
        alpha=0.14,
        label="95% Confidence Interval (Baseline ARIMA)",
    )
    ax1.axvline(train_cycles[-1] + 0.5, color="black", ls=":", lw=1.5, label="Chronological Train/Test Split")
    ax1.set_title(f"Figure 9A: {arima_res['model_name']} Baseline & Drift Forecasts vs. Actual Capacity", fontweight="bold")
    ax1.set_xlabel("Discharge Cycle")
    ax1.set_ylabel("Discharge Capacity (Ah)")
    ax1.set_ylim(1.05, 2.15)
    ax1.legend(loc="upper right", ncol=2, fontsize=8.5)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(train_cycles[1:], arima_res["residuals"], color="#1f4e78", lw=1.4)
    ax2.axhline(0.0, color="#c0392b", ls="--", lw=1.2)
    ax2.set_title(f"Figure 9B: {arima_res['model_name']} Residual Time Series", fontweight="bold")
    ax2.set_xlabel("Discharge Cycle")
    ax2.set_ylabel("Residual Error (Ah)")

    ax3 = fig.add_subplot(gs[1, 1])
    sns.histplot(arima_res["residuals"], bins=22, kde=True, color="#2980b9", ax=ax3)
    ax3.axvline(0.0, color="#c0392b", ls="--", lw=1.5)
    ax3.set_title(f"Figure 9C: {arima_res['model_name']} Residual Distribution", fontweight="bold")
    ax3.set_xlabel("Residual Error (Ah)")
    ax3.set_ylabel("Frequency")

    ax4 = fig.add_subplot(gs[1, 2])
    plot_acf(arima_res["residuals"], lags=25, ax=ax4, color="#1f4e78", vlines_kwargs={"colors": "#1f4e78"})
    ax4.set_title(f"Figure 9D: Residual ACF (LB p={arima_res['residual_diagnostics']['LjungBox_p_Lag10']:.3f})", fontweight="bold")
    ax4.set_xlabel("Lag (Cycles)")
    ax4.set_ylabel("Autocorrelation")

    fig.tight_layout()
    fig.savefig(fig9_path)
    plt.close(fig)

    # Figure 10: Holt's Linear Trend Level, Trend & Forecast
    fig10_path = fig_dir / "fig10_holt_level_trend_and_forecast.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=180, sharex=False)
    ax1.plot(train_cycles, y_train, color="#1f4e78", lw=2.0, label="Training Capacity (Cycles 1–134)")
    ax1.plot(test_cycles, y_test, color="#2c3e50", lw=2.2, label="Actual Test Capacity (Cycles 135–168)")
    ax1.plot(test_cycles, holt_res["forecast"], color="#27ae60", lw=2.4, label=f"Model 3: Val-Tuned {holt_res['model_name']}")
    ax1.plot(test_cycles, holt_res["damped_holt_forecast"], color="#8e44ad", lw=2.0, ls="--", label=f"Model 4: Damped Holt (phi={holt_res['damped_params']['phi']:.2f})")
    ax1.plot(test_cycles, holt_res["mle_holt_forecast"], color="#e67e22", lw=1.8, ls=":", label=f"Model 3b: Train-MLE Holt (alpha={holt_res['mle_alpha']:.2f}, beta={holt_res['mle_beta']:.2f})")
    ax1.axvline(train_cycles[-1] + 0.5, color="black", ls=":", lw=1.5)
    ax1.set_title("Figure 10A: Holt's Linear & Damped Trend Forecasts vs. Actual Test Observations", fontweight="bold")
    ax1.set_xlabel("Discharge Cycle")
    ax1.set_ylabel("Discharge Capacity (Ah)")
    ax1.set_ylim(1.05, 2.15)
    ax1.legend(loc="upper right", fontsize=9)

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(train_cycles, holt_res["level_series"], color="#2980b9", lw=2.0, label="Estimated Level L_t (Ah)")
    l2 = ax2_twin.plot(train_cycles, holt_res["trend_series"], color="#d35400", lw=1.8, ls="--", label="Estimated Trend Slope T_t (Ah/cycle)")
    ax2_twin.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax2.set_title("Figure 10B: Evolution of Holt's Level (L_t) and Trend Slope (T_t) Across Training Cycles", fontweight="bold")
    ax2.set_xlabel("Discharge Cycle")
    ax2.set_ylabel("Smoothed Level L_t (Ah)", color="#2980b9")
    ax2_twin.set_ylabel("Smoothed Trend T_t (Ah/cycle)", color="#d35400")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(fig10_path)
    plt.close(fig)

    # Figure 11: Actual vs Predicted Comparison on Test Period (Zoomed + Full View)
    fig11_path = fig_dir / "fig11_model_comparison_forecasts.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=180)

    # Left: Full series context
    ax1.plot(df["cycle_number"], df["discharge_capacity_ah"], color="#34495e", lw=2.0, label="Actual Capacity")
    ax1.plot(test_cycles, f_df["arima_forecast_ah"], color="#c0392b", lw=2.2, label="Model 1: ARIMA(0,1,1) Baseline")
    ax1.plot(test_cycles, f_df["arima_drift_forecast_ah"], color="#2980b9", lw=2.2, ls="-.", label="Model 2: ARIMA(0,1,1) with Drift")
    ax1.plot(test_cycles, f_df["holt_linear_forecast_ah"], color="#27ae60", lw=2.2, label="Model 3: Holt's Linear Trend")
    ax1.plot(test_cycles, f_df["holt_damped_forecast_ah"], color="#8e44ad", lw=2.0, ls="--", label="Model 4: Damped Holt's Trend")
    ax1.axvline(train_cycles[-1] + 0.5, color="black", ls=":", lw=1.4, label="Train/Test Split (Cycle 134)")
    ax1.set_title("Full Degradation Trajectory (Cycles 1–168)", fontweight="bold")
    ax1.set_xlabel("Discharge Cycle")
    ax1.set_ylabel("Discharge Capacity (Ah)")
    ax1.set_ylim(1.05, 2.15)
    ax1.legend(loc="upper right", fontsize=8.2)

    # Right: Zoomed test window (Cycles 135-168)
    ax2.plot(test_cycles, y_test, color="#2c3e50", marker="o", ms=4, lw=2.2, label="Actual Test Capacity")
    ax2.plot(test_cycles, f_df["arima_forecast_ah"], color="#c0392b", lw=2.2, label="Model 1: ARIMA(0,1,1) Baseline")
    ax2.plot(test_cycles, f_df["arima_drift_forecast_ah"], color="#2980b9", lw=2.2, ls="-.", label="Model 2: ARIMA(0,1,1) with Drift")
    ax2.plot(test_cycles, f_df["holt_linear_forecast_ah"], color="#27ae60", lw=2.2, label="Model 3: Holt's Linear Trend (Val-Tuned)")
    ax2.plot(test_cycles, f_df["holt_mle_forecast_ah"], color="#e67e22", lw=1.8, ls=":", label="Model 3b: Holt's Linear Trend (Train-MLE)")
    ax2.plot(test_cycles, f_df["holt_damped_forecast_ah"], color="#8e44ad", lw=2.0, ls="--", label="Model 4: Damped Holt's Trend")
    ax2.set_title("Zoomed Held-Out Test Period (Cycles 135–168)", fontweight="bold")
    ax2.set_xlabel("Discharge Cycle")
    ax2.set_ylabel("Discharge Capacity (Ah)")
    ax2.set_ylim(1.15, 1.40)
    ax2.legend(loc="best", fontsize=8.0)

    fig.suptitle("Figure 11: Comparative Multi-Step Forecasting Performance Across Core Models", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig11_path, bbox_inches="tight")
    plt.close(fig)

    # Figure 12: Out-of-Sample Forecast Errors & Horizon Metric Comparison
    fig12_path = fig_dir / "fig12_residual_and_error_comparison.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=180)

    ax1.plot(test_cycles, f_df["arima_error_ah"], color="#c0392b", marker="s", ms=3.5, lw=1.8, label="Model 1: ARIMA(0,1,1) Error")
    ax1.plot(test_cycles, f_df["arima_drift_error_ah"], color="#2980b9", marker="d", ms=3.5, lw=1.8, ls="-.", label="Model 2: ARIMA(0,1,1) + Drift Error")
    ax1.plot(test_cycles, f_df["holt_linear_error_ah"], color="#27ae60", marker="^", ms=3.5, lw=1.8, label="Model 3: Holt Linear Error")
    ax1.plot(test_cycles, f_df["holt_damped_error_ah"], color="#8e44ad", ls="--", lw=1.8, label="Model 4: Damped Holt Error")
    ax1.axhline(0.0, color="black", lw=1.2, ls=":")
    ax1.set_title("Held-Out Forecast Errors e_t = Q_t - Q_hat_t (Cycles 135–168)", fontweight="bold")
    ax1.set_xlabel("Discharge Cycle")
    ax1.set_ylabel("Forecast Error (Ah)")
    ax1.legend(loc="lower left", fontsize=8.2)

    h_df = eval_res["horizon_df"]
    core_h_df = h_df[h_df["Model"].isin([
        "Model 1: ARIMA(0, 1, 1) Baseline",
        "Model 2: ARIMA(0, 1, 1) with Drift",
        "Model 3: Holt's Linear Trend (Val-Tuned)",
        "Model 4: Damped Holt's Trend",
    ])]
    sns.barplot(
        data=core_h_df,
        x="Horizon_h",
        y="RMSE_Ah",
        hue="Model",
        ax=ax2,
        palette=["#c0392b", "#2980b9", "#27ae60", "#8e44ad"],
    )
    ax2.set_title("Multi-Step RMSE Across Forecast Horizons (h = 5, 10, 20, 34 Cycles)", fontweight="bold")
    ax2.set_xlabel("Forecast Horizon h (Cycles Ahead)")
    ax2.set_ylabel("Root Mean Squared Error (Ah)")
    ax2.legend(loc="upper left", fontsize=8.0)

    fig.suptitle("Figure 12: Held-Out Forecast Error Trajectory and Horizon Sensitivity Analysis", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig12_path, bbox_inches="tight")
    plt.close(fig)

    # Figure 13: Rolling-Origin Evaluation & Optional RUL Estimation
    fig13_path = fig_dir / "fig13_rolling_origin_and_rul.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=180)

    x_pos = np.arange(len(rolling_origin_df))
    width = 0.20
    ax1.bar(x_pos - 1.5 * width, rolling_origin_df["arima_rmse"], width=width, color="#c0392b", label="Model 1: ARIMA(0,1,1)")
    ax1.bar(x_pos - 0.5 * width, rolling_origin_df["arima_drift_rmse"], width=width, color="#2980b9", label="Model 2: ARIMA+Drift")
    ax1.bar(x_pos + 0.5 * width, rolling_origin_df["holt_rmse"], width=width, color="#27ae60", label="Model 3: Holt Linear")
    ax1.bar(x_pos + 1.5 * width, rolling_origin_df["damped_holt_rmse"], width=width, color="#8e44ad", label="Model 4: Damped Holt")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"Origin {c}\n({c+1}–{c+20})" for c in rolling_origin_df["train_end_cycle"]])
    ax1.set_title("Leakage-Free Rolling-Origin Evaluation (20-Cycle Horizon at 5 Origins)", fontweight="bold")
    ax1.set_xlabel("Chronological Forecast Origin (Cycle Cutoff)")
    ax1.set_ylabel("Out-of-Sample RMSE (Ah)")
    ax1.legend(loc="upper right", fontsize=8.0)

    # Right subplot: Phase 9 RUL threshold crossing from origin_cycle
    orig_c = rul_res["origin_cycle"]
    ax2.plot(df["cycle_number"], df["discharge_capacity_ah"], color="#2c3e50", lw=2.0, label="Actual Capacity")
    ax2.plot(rul_res["future_cycles"], rul_res["holt_trajectory"], color="#27ae60", lw=2.2, label="Model 3: Holt Linear Projection")
    ax2.plot(rul_res["future_cycles"], rul_res["arima_drift_trajectory"], color="#2980b9", lw=2.0, ls="-.", label="Model 2: ARIMA+Drift Projection")
    ax2.plot(rul_res["future_cycles"], rul_res["arima_trajectory"], color="#c0392b", lw=2.0, ls="--", label="Model 1: ARIMA(0,1,1) Projection")
    ax2.axhline(rul_res["eol_threshold_ah"], color="#e74c3c", ls=":", lw=1.8, label=f"NASA EOL Threshold ({rul_res['eol_threshold_ah']:.2f} Ah)")
    ax2.axvline(orig_c, color="black", ls=":", lw=1.4, label=f"Forecast Origin (Cycle {orig_c})")
    if rul_res["actual_eol_cycle"] is not None:
        ax2.scatter([rul_res["actual_eol_cycle"]], [rul_res["eol_threshold_ah"]], color="#2c3e50", s=65, zorder=6, label=f"Actual EOL (Cycle {rul_res['actual_eol_cycle']})")
    if rul_res["holt_predicted_eol_cycle"] is not None:
        ax2.scatter([rul_res["holt_predicted_eol_cycle"]], [rul_res["eol_threshold_ah"]], color="#27ae60", marker="D", s=65, zorder=6, label=f"Holt EOL (Cycle {rul_res['holt_predicted_eol_cycle']})")
    if rul_res.get("arima_drift_predicted_eol_cycle") is not None:
        ax2.scatter([rul_res["arima_drift_predicted_eol_cycle"]], [rul_res["eol_threshold_ah"]], color="#2980b9", marker="^", s=65, zorder=6, label=f"ARIMA+Drift EOL (Cycle {rul_res['arima_drift_predicted_eol_cycle']})")
    ax2.set_title(f"Phase 9 Extension: RUL Threshold-Crossing Forecast from Cycle {orig_c}", fontweight="bold")
    ax2.set_xlabel("Discharge Cycle")
    ax2.set_ylabel("Discharge Capacity (Ah)")
    ax2.set_ylim(1.05, 2.15)
    ax2.legend(loc="upper right", fontsize=7.6)

    fig.suptitle("Figure 13: Rolling-Origin Stability and Remaining Useful Life (RUL) Estimation", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig13_path, bbox_inches="tight")
    plt.close(fig)

    # Figure 14 (Phase 3.4): Comprehensive Residual Diagnostics Across All 4 Core Models
    fig14_path = fig_dir / "fig14_comprehensive_residual_diagnostics.png"
    fig, axes = plt.subplots(4, 3, figsize=(14, 13.5), dpi=180)
    diag_models = [
        ("Model 1: ARIMA(0,1,1) Baseline", train_cycles[1:], arima_res["residuals"], arima_res["residual_diagnostics"], "#c0392b"),
        ("Model 2: ARIMA(0,1,1) with Drift", train_cycles[1:], arima_res["arima_with_drift_residuals"], arima_res["arima_with_drift_diagnostics"], "#2980b9"),
        ("Model 3: Holt Linear (Val-Tuned)", train_cycles, holt_res["residuals"], holt_res["residual_diagnostics"], "#27ae60"),
        ("Model 4: Damped Holt (Val-Tuned)", train_cycles, holt_res["damped_holt_residuals"], holt_res["damped_holt_diagnostics"], "#8e44ad"),
    ]

    for row_idx, (m_title, cyc_arr, res_arr, diag_info, col_hex) in enumerate(diag_models):
        # Col 0: Residual Time Series
        ax_ts = axes[row_idx, 0]
        ax_ts.plot(cyc_arr, res_arr, color=col_hex, lw=1.4)
        ax_ts.axhline(0.0, color="black", ls="--", lw=1.0)
        ax_ts.set_title(f"{m_title} — Training Residuals", fontweight="bold", fontsize=9.5)
        ax_ts.set_xlabel("Discharge Cycle")
        ax_ts.set_ylabel("Residual (Ah)")

        # Col 1: Residual Distribution
        ax_hist = axes[row_idx, 1]
        sns.histplot(res_arr, bins=22, kde=True, color=col_hex, ax=ax_hist)
        ax_hist.axvline(0.0, color="black", ls="--", lw=1.1)
        ax_hist.axvline(np.mean(res_arr), color="#e67e22", ls=":", lw=1.3, label=f"Mean={np.mean(res_arr):.4f}")
        ax_hist.set_title(f"{m_title} — Residual Distribution", fontweight="bold", fontsize=9.5)
        ax_hist.set_xlabel("Residual (Ah)")
        ax_hist.set_ylabel("Count")
        ax_hist.legend(loc="upper right", fontsize=8.0)

        # Col 2: Residual ACF & Ljung-Box p-value
        ax_acf = axes[row_idx, 2]
        plot_acf(res_arr, lags=25, ax=ax_acf, color=col_hex, vlines_kwargs={"colors": col_hex})
        wn_tag = "White Noise" if diag_info["White_Noise_At_5Pct"] else "Autocorrelated"
        ax_acf.set_title(
            f"Residual ACF (LB p@10={diag_info['LjungBox_p_Lag10']:.3f} — {wn_tag})",
            fontweight="bold",
            fontsize=9.5,
        )
        ax_acf.set_xlabel("Lag (Cycles)")
        ax_acf.set_ylabel("ACF")

    fig.suptitle(
        "Figure 14: Comprehensive Training-Set Residual Diagnostics Across Core Statistical Models",
        fontweight="bold",
        y=1.01,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig14_path, bbox_inches="tight")
    plt.close(fig)

    # Figure 15 (Phase 3.5): Publication-Quality 6-Panel Actual vs. Predicted Comparison on Consistent Scales
    fig15_path = fig_dir / "fig15_actual_vs_predicted_six_panel.png"
    fig, axes = plt.subplots(3, 2, figsize=(14, 12.5), dpi=180)

    # Panel 1: Training Capacity and Actual Test Capacity
    ax = axes[0, 0]
    ax.plot(train_cycles, y_train, color="#1f4e78", lw=2.2, label="Training Capacity (Cycles 1–134)")
    ax.plot(test_cycles, y_test, color="#2c3e50", lw=2.2, ls="-", marker="o", ms=3, label="Actual Test Capacity (Cycles 135–168)")
    ax.axvline(train_cycles[-1] + 0.5, color="black", ls=":", lw=1.5, label="Chronological Split (Cycle 134)")
    ax.set_title("1. Chronological Train/Test Split Overview", fontweight="bold")
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel("Discharge Capacity (Ah)")
    ax.set_ylim(1.10, 2.10)
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel 2: Model 1 — ARIMA(0,1,1) Baseline Forecast vs. Actual Test Observations
    ax = axes[0, 1]
    ax.plot(test_cycles, y_test, color="#2c3e50", marker="o", ms=3.5, lw=2.0, label="Actual Test Capacity (Ah)")
    ax.plot(test_cycles, f_df["arima_forecast_ah"], color="#c0392b", lw=2.4, label="Model 1: ARIMA(0,1,1) Baseline")
    ax.fill_between(test_cycles, f_df["arima_ci_lower_ah"], f_df["arima_ci_upper_ah"], color="#c0392b", alpha=0.14, label="95% Confidence Interval")
    ax.set_title("2. Model 1: ARIMA(0,1,1) Baseline vs. Actual Test Capacity", fontweight="bold")
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel("Discharge Capacity (Ah)")
    ax.set_ylim(1.05, 1.65)
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel 3: Model 2 — ARIMA(0,1,1) with Drift vs. Actual Test Observations
    ax = axes[1, 0]
    ax.plot(test_cycles, y_test, color="#2c3e50", marker="o", ms=3.5, lw=2.0, label="Actual Test Capacity (Ah)")
    ax.plot(test_cycles, f_df["arima_drift_forecast_ah"], color="#2980b9", lw=2.4, ls="-.", label="Model 2: ARIMA(0,1,1) with Drift")
    ax.fill_between(test_cycles, f_df["arima_drift_ci_lower_ah"], f_df["arima_drift_ci_upper_ah"], color="#2980b9", alpha=0.14, label="95% Confidence Interval")
    ax.set_title("3. Model 2: ARIMA(0,1,1) with Linear Drift vs. Actual Test Capacity", fontweight="bold")
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel("Discharge Capacity (Ah)")
    ax.set_ylim(1.05, 1.65)
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel 4: Model 3 — Holt's Linear Trend Forecast vs. Actual Test Observations
    ax = axes[1, 1]
    ax.plot(test_cycles, y_test, color="#2c3e50", marker="o", ms=3.5, lw=2.0, label="Actual Test Capacity (Ah)")
    ax.plot(test_cycles, f_df["holt_linear_forecast_ah"], color="#27ae60", lw=2.4, label="Model 3: Holt Linear (Val-Tuned: α=0.10, β=0.05)")
    ax.plot(test_cycles, f_df["holt_mle_forecast_ah"], color="#e67e22", lw=2.0, ls=":", label="Model 3b: Holt Linear (Train-MLE: α=0.88, β=0.00)")
    ax.set_title("4. Model 3: Holt's Linear Trend vs. Actual Test Capacity", fontweight="bold")
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel("Discharge Capacity (Ah)")
    ax.set_ylim(1.05, 1.65)
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel 5: Combined Comparison of All Core Models
    ax = axes[2, 0]
    ax.plot(test_cycles, y_test, color="#2c3e50", marker="o", ms=3.5, lw=2.2, label="Actual Test Capacity")
    ax.plot(test_cycles, f_df["arima_forecast_ah"], color="#c0392b", lw=2.0, label="Model 1: ARIMA(0,1,1) Baseline")
    ax.plot(test_cycles, f_df["arima_drift_forecast_ah"], color="#2980b9", lw=2.0, ls="-.", label="Model 2: ARIMA(0,1,1) with Drift")
    ax.plot(test_cycles, f_df["holt_linear_forecast_ah"], color="#27ae60", lw=2.2, label="Model 3: Holt's Linear Trend")
    ax.plot(test_cycles, f_df["holt_damped_forecast_ah"], color="#8e44ad", lw=2.0, ls="--", label="Model 4: Damped Holt's Trend")
    ax.set_title("5. Combined Comparison of All Core Models (Test Horizon h=34)", fontweight="bold")
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel("Discharge Capacity (Ah)")
    ax.set_ylim(1.15, 1.40)
    ax.legend(loc="lower left", fontsize=8.0)

    # Panel 6: Forecast Error Comparison
    ax = axes[2, 1]
    ax.plot(test_cycles, f_df["arima_error_ah"], color="#c0392b", marker="s", ms=3, lw=1.8, label="Model 1: ARIMA(0,1,1) Error")
    ax.plot(test_cycles, f_df["arima_drift_error_ah"], color="#2980b9", marker="d", ms=3, lw=1.8, ls="-.", label="Model 2: ARIMA+Drift Error")
    ax.plot(test_cycles, f_df["holt_linear_error_ah"], color="#27ae60", marker="^", ms=3, lw=1.8, label="Model 3: Holt Linear Error")
    ax.plot(test_cycles, f_df["holt_damped_error_ah"], color="#8e44ad", lw=1.8, ls="--", label="Model 4: Damped Holt Error")
    ax.axhline(0.0, color="black", ls=":", lw=1.2)
    ax.set_title("6. Out-of-Sample Forecast Error Comparison (e_t = Q_t - Q_hat_t)", fontweight="bold")
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel("Forecast Error (Ah)")
    ax.set_ylim(-0.20, 0.06)
    ax.legend(loc="lower left", fontsize=8.0)

    fig.suptitle(
        "Figure 15: Six-Panel Publication Comparison of Actual vs. Forecasted Battery Discharge Capacity",
        fontweight="bold",
        y=1.01,
        fontsize=12.5,
    )
    fig.tight_layout()
    fig.savefig(fig15_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "fig09": str(fig9_path),
        "fig10": str(fig10_path),
        "fig11": str(fig11_path),
        "fig12": str(fig12_path),
        "fig13": str(fig13_path),
        "fig14": str(fig14_path),
        "fig15": str(fig15_path),
    }
