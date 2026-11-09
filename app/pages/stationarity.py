"""
PAGE 3 — STATIONARITY ANALYSIS
Displays original capacity series, first-order differenced series, exact ADF statistics,
p-values, critical values, and stationarity interpretation justifying d = 1 for ARIMA.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import plot_stationarity_comparison
from app.components.metrics import (
    render_callout,
    render_kpi_card,
    render_page_header,
    render_section_header,
)
from app.utils.data_loader import (
    load_processed_battery_data,
    load_stationarity_results,
)


def render_stationarity_page() -> None:
    df = load_processed_battery_data()
    stat_res = load_stationarity_results()

    render_page_header(
        eyebrow="Phase 4 — Unit-Root Testing & Order of Integration",
        title="Stationarity Analysis & Differencing Justification (d = 1)",
        subtitle=(
            "Formal Augmented Dickey–Fuller (ADF) unit-root diagnostics on the chronological training set "
            "(Cycles 1–134) and full trajectory (Cycles 1–168) before and after first-order differencing."
        ),
    )

    tr_lvl = stat_res["adf_train_level"]
    tr_dif = stat_res["adf_train_diff1"]
    fl_lvl = stat_res["adf_full_level"]
    fl_dif = stat_res["adf_full_diff1"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            label="Train Level Q_t (d = 0)",
            value=f"ADF = {tr_lvl['adf_statistic']:.4f}",
            subtitle=f"p = {tr_lvl['p_value']:.4f} (Non-Stationary)",
            accent="rose",
        )
    with c2:
        render_kpi_card(
            label="Train 1st Diff ΔQ_t (d = 1)",
            value=f"ADF = {tr_dif['adf_statistic']:.4f}",
            subtitle=f"p = {tr_dif['p_value']:.2e} (Stationary)",
            accent="teal",
        )
    with c3:
        render_kpi_card(
            label="5% Critical Value",
            value=f"{tr_lvl['critical_values']['5%']:.4f}",
            subtitle=f"1% Crit: {tr_lvl['critical_values']['1%']:.4f}",
            accent="cyan",
        )
    with c4:
        render_kpi_card(
            label="Selected Integration Order",
            value=f"d = {stat_res['recommended_d']}",
            subtitle="First Differencing Sufficient",
            accent="amber",
        )

    render_section_header("Visual Comparison: Level Series Q_t (d = 0) vs. First Difference ΔQ_t (d = 1)")
    scope_choice = st.radio(
        "Select Series Scope for Visual Inspection:",
        options=["Training Set Only (Cycles 1–134 — Used for Model Identification)", "Full Series (Cycles 1–168)"],
        horizontal=True,
    )
    train_only_flag = scope_choice.startswith("Training")
    st.plotly_chart(
        plot_stationarity_comparison(df, train_only=train_only_flag),
        use_container_width=True,
    )

    render_section_header("Augmented Dickey–Fuller (ADF) Unit-Root Test Summary Table")
    adf_rows = []
    for key in ["adf_train_level", "adf_train_diff1", "adf_full_level", "adf_full_diff1"]:
        item = stat_res[key]
        adf_rows.append(
            {
                "Series Tested": item["series_name"],
                "Integration Order (d)": 0 if "d=0" in item["series_name"] else 1,
                "ADF Statistic": item["adf_statistic"],
                "p-Value": item["p_value"],
                "Lags Used (AIC)": item["used_lag"],
                "Observations (N)": item["n_obs"],
                "1% Critical Value": item["critical_values"]["1%"],
                "5% Critical Value": item["critical_values"]["5%"],
                "10% Critical Value": item["critical_values"]["10%"],
                "Stationary at α = 0.05?": "Yes (Reject H₀)" if item["is_stationary_at_5pct"] else "No (Fail to Reject H₀)",
            }
        )
    adf_df = pd.DataFrame(adf_rows)
    st.dataframe(
        adf_df.style.format(
            {
                "ADF Statistic": "{:.4f}",
                "p-Value": "{:.4e}",
                "1% Critical Value": "{:.4f}",
                "5% Critical Value": "{:.4f}",
                "10% Critical Value": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    col_left, col_right = st.columns([1.15, 0.85])
    with col_left:
        render_section_header("Why Differencing Is Required for the ARIMA Methodology")
        st.markdown(
            """
            1. **Unit-Root Non-Stationarity in Levels ($d = 0$):**
               - An Autoregressive Moving Average ($\\text{ARMA}(p, q)$) process requires weak stationarity: a constant unconditional mean $\\mathbb{E}[Q_t] = \\mu$, finite time-invariant variance $\\text{Var}(Q_t) = \\sigma^2$, and autocovariance $\\gamma(k)$ depending only on lag $k$.
               - On the training set (`Cycles 1–134`), the level series $Q_t$ yields $\\text{ADF} = -1.3520 > -2.8840$ ($p = 0.6051$). Because the test statistic is less negative than the $5\\%$ critical value, we **fail to reject the null hypothesis $H_0$ of a unit root**. Fitting an undifferenced $\\text{ARMA}$ model directly to $Q_t$ would produce spurious regression and mean-reverting forecasts toward the historical average ($1.56\\text{ Ah}$).
            2. **Stationarity Achieved via First-Order Differencing ($d = 1$):**
               - Applying the first-difference operator $\\Delta Q_t = (1 - B)Q_t = Q_t - Q_{t-1}$ transforms the non-stationary capacity trajectory into cycle-by-cycle capacity increments.
               - On $\\Delta Q_t$ (`Cycles 2–134`), the ADF statistic drops to $\\text{ADF} = -8.5062 \\ll -3.4817$ ($p = 1.20 \\times 10^{-13}$), **decisively rejecting $H_0$ at the $1\\%$ significance level**.
               - Therefore, **$d = 1$ is both necessary and sufficient**. Second-order differencing ($d = 2$) is unnecessary and would introduce artificial moving-average over-differencing noise.
            """
        )
    with col_right:
        render_section_header("Training Segment Mean Shift Verification")
        seg = stat_res.get("segment_stats_train", {})
        if seg:
            seg_df = pd.DataFrame(
                [
                    {
                        "Training Segment": "Segment 1 (Cycles 1–44)",
                        "Mean Capacity (Ah)": seg["segment_1_mean"],
                        "Std Dev (Ah)": seg["segment_1_std"],
                    },
                    {
                        "Training Segment": "Segment 2 (Cycles 45–89)",
                        "Mean Capacity (Ah)": seg["segment_2_mean"],
                        "Std Dev (Ah)": seg["segment_2_std"],
                    },
                    {
                        "Training Segment": "Segment 3 (Cycles 90–134)",
                        "Mean Capacity (Ah)": seg["segment_3_mean"],
                        "Std Dev (Ah)": seg["segment_3_std"],
                    },
                ]
            )
            st.dataframe(
                seg_df.style.format({"Mean Capacity (Ah)": "{:.4f}", "Std Dev (Ah)": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
        render_callout(
            "<b>Crucial Implication for Multi-Step Forecasting:</b> While first differencing ($\\Delta Q_t$) makes "
            "the increments stationary around a negative mean ($\\overline{\\Delta Q} = -0.00525\\text{ Ah/cycle}$), "
            "standard Box–Jenkins <code>ARIMA(0, 1, 1)</code> without a drift term (<code>trend='n'</code>) forces "
            "$\\mathbb{E}[\\Delta Q_{t+h}] = 0$ for $h \\ge 2$, causing multi-step level forecasts to plateau. "
            "Capturing ongoing degradation requires either adding explicit drift (<code>Model 2: ARIMA with Drift</code>) "
            "or maintaining an adaptive slope state $T_t$ (<code>Model 3: Holt's Linear Trend</code>)."
        )

