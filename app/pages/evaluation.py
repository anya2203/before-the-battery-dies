"""
PAGE 5 — MODEL EVALUATION
Displays interactive model comparison tables (MAE, RMSE, MAPE, sMAPE), bar charts,
forecast horizon comparison (5, 10, 20, 34 cycles), rolling-origin evaluation,
online 1-step recalibration, and Ljung–Box residual diagnostic summaries.
Clearly separates held-out multi-step forecasts from rolling-origin and online 1-step evaluations.
"""

from __future__ import annotations

import streamlit as st

from app.components.charts import (
    plot_horizon_comparison_chart,
    plot_model_comparison_bars,
    plot_rolling_origin_chart,
)
from app.components.metrics import (
    render_callout,
    render_kpi_card,
    render_page_header,
    render_section_header,
)
from app.utils.data_loader import FIGURES_DIR, load_evaluation_artifacts


def render_evaluation_page() -> None:
    artifacts = load_evaluation_artifacts()
    comp_df = artifacts["model_comparison"]
    split_df = artifacts["train_val_test"]
    online_df = artifacts["online_recalibration"]
    horiz_df = artifacts["horizon_comparison"]
    roll_eval_df = artifacts["rolling_origin_eval"]
    roll_sum_df = artifacts["rolling_origin_summary"]
    resid_df = artifacts["residual_diagnostics"]

    render_page_header(
        eyebrow="Phase 7 — Fair Multi-Step & Rolling Forecast Evaluation",
        title="Quantitative Model Evaluation, Horizon Scaling & Residual Diagnostics",
        subtitle=(
            "Leakage-free empirical comparison of ARIMA(0,1,1), ARIMA(0,1,1) with Drift, Holt's Linear Trend, "
            "and Damped Holt across held-out multi-step testing (Cycles 135–168), multi-horizon windows (h = 5, 10, 20, 34), "
            "five rolling origins, online 1-step recalibration, and Ljung–Box portmanteau tests."
        ),
    )

    if comp_df is None:
        st.error("Evaluation metric CSV files were not found in `results/metrics/`. Please run `python -m src.pipeline`.")
        return

    # Top KPI cards highlighting the 4 core models on Held-Out Test Set (H = 34)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            label="Model 1: ARIMA(0,1,1) Baseline",
            value="0.1115 Ah",
            subtitle="34-Step RMSE | MAPE: 8.203%",
            accent="rose",
        )
    with c2:
        render_kpi_card(
            label="Model 2: ARIMA(0,1,1) + Drift",
            value="0.0152 Ah",
            subtitle="34-Step RMSE (-86.4%) | MAPE: 1.030%",
            accent="cyan",
        )
    with c3:
        render_kpi_card(
            label="Model 3: Holt Linear (Val-Tuned)",
            value="0.0274 Ah",
            subtitle="34-Step RMSE (-75.4%) | MAPE: 1.841%",
            accent="teal",
        )
    with c4:
        render_kpi_card(
            label="Model 4: Damped Holt (φ=0.98)",
            value="0.0751 Ah",
            subtitle="34-Step RMSE (-32.6%) | MAPE: 5.390%",
            accent="amber",
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Held-Out Multi-Step Comparison (H=34)",
            "2. Forecast Horizon Analysis (h=5, 10, 20, 34)",
            "3. Rolling-Origin & Online 1-Step Evaluation",
            "4. Residual Diagnostics & Ljung–Box Tests",
        ]
    )

    with tab1:
        render_section_header("Held-Out Test Set Multi-Step Performance (Cycles 135–168, Horizon H = 34 Cycles)")
        render_callout(
            "<b>Methodological Separation of Evaluation Regimes:</b> The table below reports strictly "
            "<b>open-loop 34-cycle multi-step forecasts</b> launched from Origin Cycle 134 onto the untouched held-out test set "
            "(<code>Cycles 135–168</code>), alongside each model's 1-step in-sample Training RMSE (<code>Cycles 1–134</code>) "
            "and 24-step Inner Validation RMSE (<code>Cycles 111–134</code>). Online 1-step recalibration is reported separately in Tab 3."
        )

        st.plotly_chart(plot_model_comparison_bars(comp_df), use_container_width=True)

        st.dataframe(
            comp_df.style.format(
                {
                    "MAE_Ah": "{:.5f}",
                    "RMSE_Ah": "{:.5f}",
                    "MAPE_Percent": "{:.3f}%",
                    "sMAPE_Percent": "{:.3f}%",
                    "Val_RMSE_Ah": "{:.5f}",
                    "Train_RMSE_Ah": "{:.5f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if split_df is not None:
            with st.expander("View Complete Train / Inner Validation / Held-Out Test Split Breakdown (`train_val_test_metrics.csv`)", expanded=False):
                st.dataframe(
                    split_df.style.format(
                        {
                            "MAE_Ah": "{:.5f}",
                            "RMSE_Ah": "{:.5f}",
                            "MAPE_Percent": "{:.3f}%",
                            "sMAPE_Percent": "{:.3f}%",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    with tab2:
        render_section_header("Multi-Step Forecast Horizon Analysis (h = 5, 10, 20, and 34 Cycles)")
        if horiz_df is not None:
            st.plotly_chart(plot_horizon_comparison_chart(horiz_df), use_container_width=True)
            render_callout(
                "<b>Key Finding on Horizon Scaling:</b> Validation-tuned <b>Holt's Linear Trend (α=0.10, β=0.05)</b> "
                "achieves the lowest error across short and medium horizons: <code>0.306% MAPE</code> at $h=5$, "
                "<code>0.591% MAPE</code> at $h=10$, and <code>0.983% MAPE</code> at $h=20$. Meanwhile, un-drifted "
                "<code>ARIMA(0,1,1)</code> degrades rapidly from <code>2.469% MAPE</code> ($h=5$) to <code>8.203% MAPE</code> ($h=34$)."
            )
            st.dataframe(
                horiz_df.style.format(
                    {
                        "MAE_Ah": "{:.5f}",
                        "RMSE_Ah": "{:.5f}",
                        "MAPE_Percent": "{:.3f}%",
                        "sMAPE_Percent": "{:.3f}%",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tab3:
        render_section_header("Leakage-Free Rolling-Origin Evaluation (Origins 90, 105, 120, 134, 148; Horizon h = 20)")
        if roll_eval_df is not None and roll_sum_df is not None:
            st.plotly_chart(plot_rolling_origin_chart(roll_eval_df), use_container_width=True)
            render_callout(
                "<b>Causal Rolling-Origin Protocol:</b> At each cutoff $c \\in \\{90, 105, 120, 134, 148\\}$, every model "
                "is fitted and causally tuned using <b>only observations $1 \\dots c$</b> before forecasting the next $h = 20$ cycles. "
                "<b>Holt's Linear Trend</b> achieves the lowest 5-origin mean RMSE (<code>0.02992 ± 0.02943 Ah</code>, "
                "<code>1.924% ± 2.032% MAPE</code>), followed closely by <b>ARIMA(0,1,1) with Drift</b> "
                "(<code>0.03337 ± 0.02826 Ah</code>, <code>2.097% ± 1.985% MAPE</code>)."
            )
            st.markdown("##### 5-Origin Summary Statistics (`results/metrics/rolling_origin_metrics.csv`)")
            st.dataframe(
                roll_sum_df.style.format(
                    {
                        "Mean_MAE_Ah": "{:.5f}",
                        "Std_MAE_Ah": "{:.5f}",
                        "Mean_RMSE_Ah": "{:.5f}",
                        "Std_RMSE_Ah": "{:.5f}",
                        "Mean_MAPE_Percent": "{:.3f}%",
                        "Std_MAPE_Percent": "{:.3f}%",
                        "Mean_sMAPE_Percent": "{:.3f}%",
                        "Std_sMAPE_Percent": "{:.3f}%",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("##### Per-Origin Breakdown (`results/metrics/rolling_origin_evaluation.csv`)")
            st.dataframe(
                roll_eval_df.style.format(precision=4),
                use_container_width=True,
                hide_index=True,
            )

        if online_df is not None:
            render_section_header("Online 1-Step-Ahead Expanding-Window Recalibration (Cycles 135–168, h = 1)")
            st.markdown(
                "When a Battery Management System (BMS) appends each newly measured cycle capacity $Q_t$ and forecasts "
                "only **1 step ahead ($h = 1$)**, all three models achieve sub-1% MAPE:"
            )
            st.dataframe(
                online_df.style.format(
                    {
                        "MAE_Ah": "{:.5f}",
                        "RMSE_Ah": "{:.5f}",
                        "MAPE_Percent": "{:.3f}%",
                        "sMAPE_Percent": "{:.3f}%",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tab4:
        render_section_header("Training-Set Residual Diagnostics & Ljung–Box Portmanteau Tests (Lags 5, 10, 15, 20)")
        if resid_df is not None:
            render_callout(
                "<b>Statistical Interpretation of Ljung–Box Hypothesis Tests:</b> Under $H_0$, residual autocorrelations "
                "$\\rho_1, \\dots, \\rho_m$ are jointly zero. <b>Model 1 (ARIMA)</b>, <b>Model 2 (ARIMA+Drift)</b>, and "
                "<b>Model 3b (Train-MLE Holt)</b> yield $p > 0.23$ at all tested lags ($5, 10, 15, 20$), meaning we "
                "<b>fail to reject $H_0$ at $\\alpha = 0.05$</b> (consistent with uncorrelated residuals, though failing to reject $H_0$ "
                "does not prove strict white noise, and Shapiro–Wilk tests [$p < 10^{-10}$] reject normality due to heavy-tailed "
                "regeneration spikes). Conversely, <b>Model 3 (Val-Tuned Holt, α=0.10, β=0.05)</b> rejects $H_0$ on 1-step residuals "
                "because low $\\alpha = 0.10$ acts as a low-pass filter against regeneration jumps to stabilize multi-step slope projection."
            )
            st.dataframe(
                resid_df.style.format(
                    {
                        "Mean_Ah": "{:.6f}",
                        "Std_Ah": "{:.5f}",
                        "Skewness": "{:.3f}",
                        "Kurtosis": "{:.3f}",
                        "LjungBox_Stat_Lag5": "{:.3f}",
                        "LjungBox_p_Lag5": "{:.4e}",
                        "LjungBox_Stat_Lag10": "{:.3f}",
                        "LjungBox_p_Lag10": "{:.4e}",
                        "LjungBox_Stat_Lag15": "{:.3f}",
                        "LjungBox_p_Lag15": "{:.4e}",
                        "LjungBox_Stat_Lag20": "{:.3f}",
                        "LjungBox_p_Lag20": "{:.4e}",
                        "Shapiro_Stat": "{:.4f}",
                        "Shapiro_p": "{:.4e}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            col_f1, col_f2 = st.columns(2)
            fig14_path = FIGURES_DIR / "fig14_comprehensive_residual_diagnostics.png"
            fig15_path = FIGURES_DIR / "fig15_actual_vs_predicted_six_panel.png"
            if fig14_path.exists():
                with col_f1:
                    st.image(
                        str(fig14_path),
                        caption="Figure 14: Comprehensive 5-Model Residual Diagnostics (Cycles 1–134)",
                        use_container_width=True,
                    )
            if fig15_path.exists():
                with col_f2:
                    st.image(
                        str(fig15_path),
                        caption="Figure 15: Six-Panel Actual-vs-Predicted Comparison on Identical Scales",
                        use_container_width=True,
                    )

