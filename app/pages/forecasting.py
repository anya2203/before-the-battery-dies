"""
PAGE 4 — FORECASTING LAB
Interactive multi-step forecasting workbench allowing the user to select:
- ARIMA (0,1,1)
- ARIMA (0,1,1) with drift
- Holt Linear Trend
- Damped Holt
Supports forecast origin and horizon selection, causal model fitting (zero future lookahead),
actual vs. predicted capacity visualization, model parameter inspection, and forecast tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components.charts import plot_forecast_lab_chart
from app.components.metrics import (
    render_callout,
    render_kpi_card,
    render_page_header,
    render_section_header,
)
from app.utils.data_loader import run_causal_forecast_at_origin


MODEL_OPTIONS = [
    "ARIMA (0,1,1)",
    "ARIMA (0,1,1) with drift",
    "Holt Linear Trend",
    "Damped Holt",
]


def render_forecasting_page() -> None:
    render_page_header(
        eyebrow="Phase 5 & Phase 6 — Causal Statistical Forecasting Workbench",
        title="Interactive Forecasting Lab: ARIMA vs. Trend-Aware Exponential Smoothing",
        subtitle=(
            "Configure forecast origin, multi-step horizon, and model specifications. Every model is fitted and tuned "
            "strictly on historical observations up to the selected forecast origin (zero lookahead leakage)."
        ),
    )

    # Interactive Controls Panel
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.8, 1.0, 1.0, 1.1])
    with ctrl_col1:
        selected_models = st.multiselect(
            "Select Forecasting Models:",
            options=MODEL_OPTIONS,
            default=MODEL_OPTIONS,
        )
    with ctrl_col2:
        origin_cycle = st.slider(
            "Forecast Origin Cycle:",
            min_value=60,
            max_value=158,
            value=134,
            step=1,
            help="Default Cycle 134 corresponds to the primary 80%/20% chronological split (Train: 1–134, Test: 135–168).",
        )
    with ctrl_col3:
        max_default_h = 34 if origin_cycle == 134 else min(34, 168 - origin_cycle)
        horizon = st.slider(
            "Forecast Horizon (Cycles):",
            min_value=5,
            max_value=60,
            value=max(5, max_default_h),
            step=1,
            help="Number of future discharge cycles h to forecast ahead from the selected origin.",
        )
    with ctrl_col4:
        st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
        show_ci = st.checkbox("Show 95% ARIMA Intervals", value=True)
        show_fitted = st.checkbox("Show In-Sample Fit", value=False)

    if not selected_models:
        st.warning("Please select at least one forecasting model above to display predictions.")
        return

    forecast_bundle = run_causal_forecast_at_origin(origin_cycle=origin_cycle, horizon=horizon)
    future_actual_df = forecast_bundle["future_actual_df"]
    n_eval = len(future_actual_df)

    # Top Summary Callout
    if origin_cycle == 134 and horizon == 34:
        render_callout(
            "<b>Primary Benchmark Configuration Active (Origin Cycle 134, Horizon H = 34 Cycles):</b> "
            "Models are trained on <code>Cycles 1–134</code> (with inner chronological validation on <code>Cycles 111–134</code>) "
            "and evaluated on the untouched held-out test set (<code>Cycles 135–168</code>)."
        )
    else:
        render_callout(
            f"<b>Custom Causal Forecast Origin Active (Origin Cycle {origin_cycle}, Horizon H = {horizon} Cycles):</b> "
            f"All selected models are fitted and causally tuned strictly on <code>Cycles 1–{origin_cycle}</code> with zero access "
            f"to post-origin observations. Overlapping actual evaluation window: <code>{n_eval} cycles</code>."
        )

    # KPI Row for Selected Models (evaluated over overlapping actual cycles)
    if n_eval > 0:
        cols = st.columns(len(selected_models))
        for idx, m_key in enumerate(selected_models):
            m_info = forecast_bundle["models"][m_key]
            m_metrics = m_info["eval_metrics"]
            accent_color = (
                "teal"
                if m_key == "Holt Linear Trend"
                else ("cyan" if "drift" in m_key else ("amber" if "Damped" in m_key else "rose"))
            )
            with cols[idx]:
                render_kpi_card(
                    label=f"{m_key} ({n_eval}-Step RMSE)",
                    value=f"{m_metrics['RMSE']:.4f} Ah",
                    subtitle=f"MAE: {m_metrics['MAE']:.4f} Ah | MAPE: {m_metrics['MAPE']:.2f}%",
                    accent=accent_color,
                )

    render_section_header("Actual vs. Predicted Discharge Capacity Trajectory")
    st.plotly_chart(
        plot_forecast_lab_chart(
            forecast_bundle=forecast_bundle,
            selected_models=selected_models,
            show_ci=show_ci,
            show_fitted=show_fitted,
        ),
        use_container_width=True,
    )

    st.caption(
        "Note on Uncertainty Intervals: Analytical 95% state-space prediction intervals are shown for "
        "ARIMA(0,1,1) and ARIMA(0,1,1) with Drift (`statsmodels` `get_forecast().conf_int(alpha=0.05)`). "
        "Classical `statsmodels.tsa.holtwinters.Holt` provides point forecasts without closed-form parametric intervals."
    )

    render_section_header("Fitted Model Parameters & Mathematical Explanations")
    param_cols = st.columns(min(2, len(selected_models)))
    for idx, m_key in enumerate(selected_models):
        m_info = forecast_bundle["models"][m_key]
        with param_cols[idx % len(param_cols)]:
            st.markdown(f"#### {m_info['full_name']}")
            st.markdown(m_info["explanation"])
            p_df = pd.DataFrame(
                [{"Parameter / Diagnostic": k, "Estimated Value": v} for k, v in m_info["params"].items()]
            )
            st.dataframe(p_df, use_container_width=True, hide_index=True)

    render_section_header(f"Multi-Step Forecast Table (Cycles {origin_cycle + 1} – {origin_cycle + horizon})")
    future_cycles = forecast_bundle["future_cycles"]
    q_init = forecast_bundle["q_init"]

    actual_map = dict(
        zip(
            future_actual_df["cycle_number"].astype(int),
            future_actual_df["discharge_capacity_ah"].astype(float),
        )
    )

    table_rows = []
    for step_idx, cyc in enumerate(future_cycles):
        c_int = int(cyc)
        act_q = actual_map.get(c_int, np.nan)
        row_dict = {
            "Discharge_Cycle": c_int,
            "Horizon_Step_h": step_idx + 1,
            "Actual_Capacity_Ah": act_q,
            "Actual_SOH_Percent": (act_q / q_init * 100.0) if not np.isnan(act_q) else np.nan,
        }
        for m_key in selected_models:
            m_info = forecast_bundle["models"][m_key]
            pred_q = float(m_info["forecast"][step_idx])
            col_prefix = (
                m_key.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "_")
            )
            row_dict[f"{col_prefix}_Pred_Ah"] = pred_q
            row_dict[f"{col_prefix}_Pred_SOH_%"] = (pred_q / q_init) * 100.0
            row_dict[f"{col_prefix}_Error_Ah"] = (act_q - pred_q) if not np.isnan(act_q) else np.nan
        table_rows.append(row_dict)

    forecast_table_df = pd.DataFrame(table_rows)
    st.dataframe(
        forecast_table_df.style.format(precision=4, na_rep="— (Future Beyond Dataset)"),
        use_container_width=True,
        height=340,
        hide_index=True,
    )

    csv_bytes = forecast_table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download Forecast Table (Origin {origin_cycle}, H={horizon}) as CSV",
        data=csv_bytes,
        file_name=f"battery_forecasts_origin{origin_cycle}_h{horizon}.csv",
        mime="text/csv",
    )

