"""
PAGE 6 — REMAINING USEFUL LIFE (RUL)
Allows the user to select forecast origin and End-of-Life (EOL) capacity threshold.
Displays current capacity, selected threshold, forecasted capacity trajectory,
predicted threshold-crossing cycle, and predicted RUL in cycles.
Compares against actual RUL only when future observations are available for retrospective evaluation.
Clearly labels RUL estimates as experimental and dependent on threshold and model assumptions.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import plot_rul_trajectory_chart
from app.components.metrics import (
    render_callout,
    render_kpi_card,
    render_page_header,
    render_section_header,
)
from app.utils.data_loader import (
    load_processed_battery_data,
    run_interactive_rul_analysis,
)


RUL_MODELS = [
    "Holt Linear Trend",
    "ARIMA (0,1,1) with drift",
    "Damped Holt",
    "ARIMA (0,1,1)",
]


def render_rul_page() -> None:
    df = load_processed_battery_data()

    render_page_header(
        eyebrow="Phase 9 Extension — Experimental Prognostics & Threshold Crossing",
        title="Remaining Useful Life (RUL) Estimation via Statistical Trend Extrapolation",
        subtitle=(
            "Project cycle-level discharge capacity trajectories forward from a chosen forecast origin to estimate "
            "when the cell will cross a user-defined End-of-Life (EOL) capacity threshold."
        ),
    )

    render_callout(
        "<b>Experimental Prognostics Disclaimer:</b> Remaining Useful Life (RUL) projections shown on this page are "
        "<b>experimental statistical extrapolations</b> that depend strongly on the user-selected EOL capacity threshold "
        "and local linear slope assumptions. In the NASA Ames PCoE Cell B0006 aging experiment, <b>1.40 Ah</b> "
        "(30% degradation from the 2.00 Ah nominal rating; 68.23% of the 2.0520 Ah initial measured capacity) is the "
        "documented experimental stopping criterion—<i>not</i> a universal EOL threshold across all battery applications.",
        variant="warn",
    )

    ctrl1, ctrl2, ctrl3 = st.columns([1.1, 1.1, 1.4])
    with ctrl1:
        origin_cycle = st.slider(
            "Forecast Origin Cycle:",
            min_value=40,
            max_value=150,
            value=60,
            step=1,
            help="Models are fitted and causally tuned using ONLY Cycles 1..origin_cycle. Default Cycle 60 is the pre-EOL benchmark origin.",
        )
    with ctrl2:
        eol_threshold_ah = st.slider(
            "EOL Capacity Threshold (Ah):",
            min_value=1.20,
            max_value=1.80,
            value=1.40,
            step=0.01,
            help="Default 1.40 Ah corresponds to the documented NASA B0006 experiment threshold (crossed at Cycle 112).",
        )
    with ctrl3:
        selected_models = st.multiselect(
            "Models to Project:",
            options=RUL_MODELS,
            default=RUL_MODELS,
        )

    rul_bundle = run_interactive_rul_analysis(
        origin_cycle=origin_cycle,
        eol_threshold_ah=eol_threshold_ah,
        max_horizon=120,
    )

    q_origin = rul_bundle["origin_capacity_ah"]
    act_eol = rul_bundle["actual_eol_cycle"]
    act_rul = rul_bundle["actual_rul_cycles"]
    already_crossed = rul_bundle["already_crossed_in_history"]
    retro_available = rul_bundle["retrospective_available"]

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card(
            label=f"Capacity at Origin (C{origin_cycle})",
            value=f"{q_origin:.4f} Ah",
            subtitle=f"Trained strictly on C1–{origin_cycle}",
            accent="cyan",
        )
    with k2:
        render_kpi_card(
            label="Selected EOL Threshold",
            value=f"{eol_threshold_ah:.2f} Ah",
            subtitle="NASA B0006 Default: 1.40 Ah" if abs(eol_threshold_ah - 1.40) < 1e-4 else "Custom User Threshold",
            accent="rose",
        )
    with k3:
        if already_crossed:
            render_kpi_card(
                label="Actual Retrospective RUL",
                value="Already Crossed",
                subtitle=f"Crossed at Cycle {act_eol} (≤ Origin {origin_cycle})",
                accent="amber",
            )
        elif retro_available:
            render_kpi_card(
                label="Actual Retrospective RUL",
                value=f"{act_rul} Cycles",
                subtitle=f"Actual Crossing: Cycle {act_eol}",
                accent="teal",
            )
        else:
            render_kpi_card(
                label="Actual Retrospective RUL",
                value="Not Reached",
                subtitle="Threshold not crossed within 168 cycles",
                accent="amber",
            )
    with k4:
        h_rul = rul_bundle["holt_predicted_rul_cycles"]
        h_eol = rul_bundle["holt_predicted_eol_cycle"]
        h_err = rul_bundle["holt_rul_error_cycles"]
        render_kpi_card(
            label="Holt Linear Predicted RUL",
            value=f"{h_rul} Cycles" if h_rul is not None else "Never Crosses",
            subtitle=(
                f"Pred Cycle {h_eol} (Error: {h_err:+d} cyc)"
                if (h_rul is not None and retro_available and not already_crossed)
                else (f"Pred Cycle {h_eol}" if h_eol is not None else "Horizon > 120 cyc")
            ),
            accent="teal",
        )
    with k5:
        d_rul = rul_bundle["arima_drift_predicted_rul_cycles"]
        d_eol = rul_bundle["arima_drift_predicted_eol_cycle"]
        d_err = rul_bundle["arima_drift_rul_error_cycles"]
        render_kpi_card(
            label="ARIMA+Drift Predicted RUL",
            value=f"{d_rul} Cycles" if d_rul is not None else "Never Crosses",
            subtitle=(
                f"Pred Cycle {d_eol} (Error: {d_err:+d} cyc)"
                if (d_rul is not None and retro_available and not already_crossed)
                else (f"Pred Cycle {d_eol}" if d_eol is not None else "Horizon > 120 cyc")
            ),
            accent="cyan",
        )

    if already_crossed:
        st.info(
            f"Note: The battery already crossed the selected threshold ({eol_threshold_ah:.2f} Ah) at **Cycle {act_eol}**, "
            f"which is at or before your selected forecast origin (**Cycle {origin_cycle}**). Decrease the origin cycle "
            f"(e.g., to Cycle 60) or lower the threshold to evaluate forward-looking RUL prediction."
        )

    render_section_header("Causal Capacity Trajectory Extrapolation & EOL Threshold Crossing")
    st.plotly_chart(
        plot_rul_trajectory_chart(df, rul_bundle, selected_models=selected_models),
        use_container_width=True,
    )

    render_section_header("Model-by-Model Experimental RUL Comparison Table")
    rows = [
        {
            "Forecasting Model": "Model 3: Holt's Linear Trend (Causally Tuned)",
            "Causal Tuning / Fit Scope": f"Cycles 1–{origin_cycle} (α={rul_bundle['causal_holt_alpha_at_origin']:.2f}, β={rul_bundle['causal_holt_beta_at_origin']:.2f})",
            "Origin Capacity (Ah)": f"{q_origin:.4f}",
            "Selected Threshold (Ah)": f"{eol_threshold_ah:.2f}",
            "Predicted EOL Cycle": str(rul_bundle["holt_predicted_eol_cycle"]) if rul_bundle["holt_predicted_eol_cycle"] else "None (Does Not Cross)",
            "Predicted RUL (Cycles)": str(rul_bundle["holt_predicted_rul_cycles"]) if rul_bundle["holt_predicted_rul_cycles"] is not None else "None",
            "Actual EOL Cycle (Retrospective)": str(act_eol) if (retro_available and not already_crossed) else ("Already Crossed" if already_crossed else "Not Observed in 168 Cycles"),
            "Actual RUL (Cycles)": str(act_rul) if (retro_available and not already_crossed) else "—",
            "RUL Error (Pred − Actual)": (
                f"{rul_bundle['holt_rul_error_cycles']:+d} cycles"
                if (rul_bundle["holt_rul_error_cycles"] is not None and retro_available and not already_crossed)
                else "—"
            ),
        },
        {
            "Forecasting Model": "Model 2: ARIMA(0, 1, 1) with Linear Drift",
            "Causal Tuning / Fit Scope": f"Cycles 1–{origin_cycle} (trend='t')",
            "Origin Capacity (Ah)": f"{q_origin:.4f}",
            "Selected Threshold (Ah)": f"{eol_threshold_ah:.2f}",
            "Predicted EOL Cycle": str(rul_bundle["arima_drift_predicted_eol_cycle"]) if rul_bundle["arima_drift_predicted_eol_cycle"] else "None (Does Not Cross)",
            "Predicted RUL (Cycles)": str(rul_bundle["arima_drift_predicted_rul_cycles"]) if rul_bundle["arima_drift_predicted_rul_cycles"] is not None else "None",
            "Actual EOL Cycle (Retrospective)": str(act_eol) if (retro_available and not already_crossed) else ("Already Crossed" if already_crossed else "Not Observed in 168 Cycles"),
            "Actual RUL (Cycles)": str(act_rul) if (retro_available and not already_crossed) else "—",
            "RUL Error (Pred − Actual)": (
                f"{rul_bundle['arima_drift_rul_error_cycles']:+d} cycles"
                if (rul_bundle["arima_drift_rul_error_cycles"] is not None and retro_available and not already_crossed)
                else "—"
            ),
        },
        {
            "Forecasting Model": "Model 4: Damped Holt's Trend (Causally Tuned)",
            "Causal Tuning / Fit Scope": (
                f"Cycles 1–{origin_cycle} (α={rul_bundle['damped_causal_alpha']:.2f}, "
                f"β={rul_bundle['damped_causal_beta']:.2f}, φ={rul_bundle['damped_causal_phi']:.2f})"
            ),
            "Origin Capacity (Ah)": f"{q_origin:.4f}",
            "Selected Threshold (Ah)": f"{eol_threshold_ah:.2f}",
            "Predicted EOL Cycle": str(rul_bundle["damped_predicted_eol_cycle"]) if rul_bundle["damped_predicted_eol_cycle"] else "None (Plateaus Above Threshold)",
            "Predicted RUL (Cycles)": str(rul_bundle["damped_predicted_rul_cycles"]) if rul_bundle["damped_predicted_rul_cycles"] is not None else "None",
            "Actual EOL Cycle (Retrospective)": str(act_eol) if (retro_available and not already_crossed) else ("Already Crossed" if already_crossed else "Not Observed in 168 Cycles"),
            "Actual RUL (Cycles)": str(act_rul) if (retro_available and not already_crossed) else "—",
            "RUL Error (Pred − Actual)": (
                f"{rul_bundle['damped_rul_error_cycles']:+d} cycles"
                if (rul_bundle["damped_rul_error_cycles"] is not None and retro_available and not already_crossed)
                else "—"
            ),
        },
        {
            "Forecasting Model": "Model 1: ARIMA(0, 1, 1) — Baseline",
            "Causal Tuning / Fit Scope": f"Cycles 1–{origin_cycle} (trend='n')",
            "Origin Capacity (Ah)": f"{q_origin:.4f}",
            "Selected Threshold (Ah)": f"{eol_threshold_ah:.2f}",
            "Predicted EOL Cycle": str(rul_bundle["arima_predicted_eol_cycle"]) if rul_bundle["arima_predicted_eol_cycle"] else "None (Horizontal Plateau)",
            "Predicted RUL (Cycles)": str(rul_bundle["arima_predicted_rul_cycles"]) if rul_bundle["arima_predicted_rul_cycles"] is not None else "None",
            "Actual EOL Cycle (Retrospective)": str(act_eol) if (retro_available and not already_crossed) else ("Already Crossed" if already_crossed else "Not Observed in 168 Cycles"),
            "Actual RUL (Cycles)": str(act_rul) if (retro_available and not already_crossed) else "—",
            "RUL Error (Pred − Actual)": (
                f"{rul_bundle['arima_rul_error_cycles']:+d} cycles"
                if (rul_bundle["arima_rul_error_cycles"] is not None and retro_available and not already_crossed)
                else "—"
            ),
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    render_callout(
        "<b>Physical Interpretation of the Default Origin-60 Benchmark (1.40 Ah Threshold):</b> At Cycle 60 "
        "(<code>Q₆₀ = 1.6447 Ah</code>), only Cycles 1–60 are known. Causally tuned <b>Holt's Linear Trend</b> "
        "projects threshold crossing at <b>Cycle 98</b> (<code>Predicted RUL = 38 cycles</code>) and <b>ARIMA(0,1,1) with Drift</b> "
        "projects crossing at <b>Cycle 96</b> (<code>Predicted RUL = 36 cycles</code>), whereas actual crossing occurs at "
        "<b>Cycle 112</b> (<code>Actual RUL = 52 cycles</code>). The 14-to-16 cycle conservative (early) warning margin occurs "
        "because an unscheduled rest pause at <b>Cycle 90</b> induced a large <code>+0.1518 Ah</code> capacity regeneration jump "
        "that temporarily restored cell capacity and delayed physical threshold crossing by ~14 cycles. Meanwhile, "
        "un-drifted <b>ARIMA(0,1,1)</b> plateaus horizontally at <code>1.643 Ah</code> and never predicts EOL."
    )

