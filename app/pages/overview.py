"""
PAGE 1 — OVERVIEW
Displays project overview, NASA B0006 dataset KPIs, interactive capacity & SOH trend charts,
dataset summary, and the central & supporting research questions.
"""

from __future__ import annotations

import streamlit as st

from app.components.charts import plot_capacity_degradation, plot_soh_trend
from app.components.metrics import (
    render_callout,
    render_kpi_card,
    render_page_header,
    render_section_header,
)
from app.utils.data_loader import load_processed_battery_data


def render_overview_page() -> None:
    df = load_processed_battery_data()

    n_cycles = int(len(df))
    q_init = float(df["discharge_capacity_ah"].iloc[0])
    q_min = float(df["discharge_capacity_ah"].min())
    min_cycle = int(df.loc[df["discharge_capacity_ah"].idxmin(), "cycle_number"])
    q_final = float(df["discharge_capacity_ah"].iloc[-1])

    soh_min = float(df["soh_percent"].min())
    soh_final = float(df["soh_percent"].iloc[-1])

    max_fade_pct = float(df["capacity_fade_percent"].max())
    final_fade_pct = float(df["capacity_fade_percent"].iloc[-1])

    render_page_header(
        eyebrow="Industrial Battery Health & Prognostics Analytics",
        title="Before the Battery Dies: Forecasting Lithium-Ion Battery Degradation Using Statistical Time-Series Models",
        subtitle=(
            "End-to-end statistical time-series investigation comparing conventional Box–Jenkins ARIMA against "
            "trend-aware models (ARIMA with Linear Drift, Holt's Linear Trend, and Damped Holt) on cycle-indexed "
            "discharge capacity measurements from NASA Ames PCoE Cell B0006."
        ),
    )

    # Row of 6 KPI Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_kpi_card(
            label="Dataset / Cell ID",
            value="NASA B0006",
            subtitle="Mendeley V2 (24 °C)",
            accent="teal",
        )
    with c2:
        render_kpi_card(
            label="Discharge Cycles",
            value=f"{n_cycles}",
            subtitle="Ops 2–614 (50,285 rows)",
            accent="cyan",
        )
    with c3:
        render_kpi_card(
            label="Initial Capacity",
            value=f"{q_init:.3f} Ah",
            subtitle="Cycle 1 (100.00% SOH)",
            accent="teal",
        )
    with c4:
        render_kpi_card(
            label="Min / Final Capacity",
            value=f"{q_min:.3f} Ah",
            subtitle=f"Min C{min_cycle} | C168: {q_final:.3f} Ah",
            accent="amber",
        )
    with c5:
        render_kpi_card(
            label="Min / Latest SOH",
            value=f"{soh_min:.2f}%",
            subtitle=f"Min C{min_cycle} | C168: {soh_final:.2f}%",
            accent="rose",
        )
    with c6:
        render_kpi_card(
            label="Capacity Degradation",
            value=f"{max_fade_pct:.2f}%",
            subtitle=f"Peak Fade | C168: {final_fade_pct:.2f}%",
            accent="amber",
        )

    render_callout(
        "<b>Central Research Question:</b> <i>Can trend-aware statistical forecasting models improve the prediction "
        "of lithium-ion battery capacity degradation compared with conventional ARIMA forecasting?</i><br>"
        "<span style='color:#94a3b8; font-size:0.85rem;'>"
        "Evaluated under a strict chronological 80%/20% split (Training: Cycles 1–134, Inner Validation: Cycles 111–134, "
        "Held-Out Test: Cycles 135–168, Horizon H = 34 cycles) with zero test-data leakage."
        "</span>"
    )

    render_section_header("Cycle-Level Capacity & State of Health Trajectories")
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            plot_capacity_degradation(df, show_split=True),
            use_container_width=True,
        )
    with col_right:
        st.plotly_chart(
            plot_soh_trend(df),
            use_container_width=True,
        )

    render_section_header("Dataset Provenance, Experimental Protocol & Supporting Research Questions")
    info_col1, info_col2 = st.columns([1.05, 1.0])
    with info_col1:
        st.markdown(
            """
            #### Dataset Summary (`NASA_B0006` / Mendeley Data V2)
            - **Source Repository:** *EV Lithium Ion Battery State of Health Dataset*, Mendeley Data V2 (`DOI: 10.17632/7ctkcfnsdw.2`), derived from the NASA Ames Prognostics Center of Excellence (PCoE) Lithium-Ion Battery Aging Dataset.
            - **Cell Chemistry & Rating:** Commercial 18650 $\\text{LiCoO}_2$ / graphite cylindrical cell with a $2.00\\text{ Ah}$ nominal rating ($Q_1 = 2.0520\\text{ Ah}$ initial measured capacity).
            - **Cycling Protocol:** Charged in CC-CV mode at $1.5\\text{ A}$ to $4.2\\text{ V}$ ($20\\text{ mA}$ cutoff); discharged at a constant current of $2.0\\text{ A}$ down to $2.5\\text{ V}$ inside a $24.0^\\circ\\text{C}$ isothermal chamber.
            - **Cycle-Indexed Time Series:** Indexed by sequential discharge cycle number $t \\in \\{1, 2, \\dots, 168\\}$. Rest pauses for Electrochemical Impedance Spectroscopy (EIS) induce transient **capacity regeneration spikes** (most notably $+0.1518\\text{ Ah}$ at Cycle 90).
            """
        )
    with info_col2:
        st.markdown(
            """
            #### Supporting Research Questions Investigated
            1. **Stationarity & Differencing:** Does $Q_t$ exhibit stochastic non-stationarity, and is first-order differencing ($\\Delta Q_t$, $d=1$) necessary and sufficient for stationarity?
            2. **Explicit Drift / Slope States:** Why does un-drifted $\\text{ARIMA}(0,1,1)$ plateau over multi-step horizons, and how much error reduction do $\\text{ARIMA}(0,1,1)$ with Drift, Holt's Linear Trend, and Damped Holt achieve?
            3. **Horizon Scaling:** How do $\\text{MAE}$, $\\text{RMSE}$, $\\text{MAPE}$, and $\\text{sMAPE}$ scale across $h \\in \\{5, 10, 20, 34\\}$ cycles?
            4. **Rolling-Origin Stability:** Do trend-aware models maintain superiority across 5 chronological origins (`90, 105, 120, 134, 148`) under causal tuning?
            5. **Residual Diagnostics:** Do model residuals fail to reject the Ljung–Box null hypothesis of zero autocorrelation across lags 5–20?
            """
        )

    with st.expander("Inspect Processed Cycle-Level Dataset (`data/processed/battery_capacity.csv` — 168 Rows)", expanded=False):
        st.dataframe(
            df.style.format(
                {
                    "discharge_capacity_ah": "{:.4f}",
                    "soh_percent": "{:.2f}",
                    "soh_rated_percent": "{:.2f}",
                    "capacity_fade_ah": "{:.4f}",
                    "capacity_fade_percent": "{:.2f}",
                    "discharge_duration_s": "{:.1f}",
                    "mean_voltage_v": "{:.4f}",
                    "min_voltage_v": "{:.4f}",
                    "mean_current_a": "{:.4f}",
                }
            ),
            use_container_width=True,
            height=320,
        )

