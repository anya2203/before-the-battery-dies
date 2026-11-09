"""
PAGE 2 — EXPLORATORY DATA ANALYSIS
Displays interactive Plotly charts and diagnostic interpretations for:
- Capacity vs. cycle
- SOH vs. cycle
- Capacity fade & cycle-to-cycle increments
- Rolling mean and standard deviation
- Capacity distribution
- Training-only ACF and PACF plots
Plus optional inspection of the static publication figures (fig01–fig08).
"""

from __future__ import annotations

import streamlit as st

from app.components.charts import (
    plot_acf_pacf_interactive,
    plot_capacity_degradation,
    plot_capacity_distribution,
    plot_capacity_fade_and_diff,
    plot_rolling_statistics,
    plot_soh_trend,
)
from app.components.metrics import (
    render_callout,
    render_kpi_card,
    render_page_header,
    render_section_header,
)
from app.utils.data_loader import (
    FIGURES_DIR,
    compute_acf_pacf_arrays,
    load_processed_battery_data,
)


def render_eda_page() -> None:
    df = load_processed_battery_data()
    acf_pacf_data = compute_acf_pacf_arrays(max_lags=35, train_only=True)

    render_page_header(
        eyebrow="Phase 3 — Exploratory Time-Series Diagnostics",
        title="Exploratory Data Analysis (EDA) of Cycle-Level Battery Degradation",
        subtitle=(
            "Interactive characterization of capacity fade, State of Health (SOH), rest-induced regeneration spikes, "
            "rolling moment evolution, distributional skewness, and training-set autocorrelation structure."
        ),
    )

    diffs = df["discharge_capacity_ah"].diff().dropna()
    pos_jumps = int((diffs > 0).sum())
    neg_steps = int((diffs <= 0).sum())
    max_jump = float(diffs.max())
    max_jump_cycle = int(df.loc[diffs.idxmax(), "cycle_number"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            label="Mean Capacity (Ah)",
            value=f"{df['discharge_capacity_ah'].mean():.4f} Ah",
            subtitle=f"Std Dev: {df['discharge_capacity_ah'].std():.4f} Ah",
            accent="teal",
        )
    with c2:
        render_kpi_card(
            label="Mean Cycle Drift ΔQ",
            value=f"{diffs.mean():.5f} Ah",
            subtitle=f"Median: {diffs.median():.5f} Ah/cycle",
            accent="cyan",
        )
    with c3:
        render_kpi_card(
            label="Regeneration Spikes",
            value=f"{pos_jumps} / 167",
            subtitle=f"Max Jump: +{max_jump:.4f} Ah (Cycle {max_jump_cycle})",
            accent="amber",
        )
    with c4:
        render_kpi_card(
            label="Threshold Crossings",
            value="C61 / C103 / C112",
            subtitle="80% SOH / 70% SOH / 1.40 Ah Rated",
            accent="rose",
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1. Capacity & SOH Trends",
            "2. Capacity Fade & ΔQ_t",
            "3. Rolling Mean & Std Dev",
            "4. Capacity Distributions",
            "5. ACF & PACF Diagnostics",
        ]
    )

    with tab1:
        render_section_header("Discharge Capacity (Ah) and State of Health (SOH %) vs. Cycle")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_capacity_degradation(df, show_split=True), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_soh_trend(df), use_container_width=True)
        render_callout(
            "<b>Diagnostic Interpretation:</b> Discharge capacity declines from <code>2.0520 Ah</code> (Cycle 1) to "
            "a minimum of <code>1.1721 Ah</code> (Cycle 164, <code>57.12% SOH</code>), finishing at <code>1.2038 Ah</code> "
            "(Cycle 168, <code>58.66% SOH</code>). Upward diamonds highlight self-recharge regeneration spikes following "
            "rest/EIS intervals at Cycles 20 (<code>+0.1220 Ah</code>), 48 (<code>+0.1190 Ah</code>), 90 (<code>+0.1518 Ah</code>), "
            "and 151 (<code>+0.0380 Ah</code>)."
        )

    with tab2:
        render_section_header("Cumulative Capacity Fade & Cycle-over-Cycle Differences ΔQ_t")
        st.plotly_chart(plot_capacity_fade_and_diff(df), use_container_width=True)
        render_callout(
            "<b>Diagnostic Interpretation:</b> Cumulative fade reaches <code>0.8799 Ah</code> (<code>42.88%</code> of initial capacity) "
            "at Cycle 164. The lower panel shows that cycle-to-cycle increments $\\Delta Q_t = Q_t - Q_{t-1}$ fluctuate "
            "around a persistent negative drift (<code>-0.00508 Ah/cycle</code>) interrupted by discrete positive regeneration "
            "impulses (amber bars). There is no fixed periodic seasonality."
        )

    with tab3:
        render_section_header("10-Cycle and 20-Cycle Rolling Mean and Standard Deviation")
        st.plotly_chart(plot_rolling_statistics(df), use_container_width=True)
        render_callout(
            "<b>Diagnostic Interpretation:</b> Both the 10-cycle and 20-cycle rolling means decline monotonically "
            "from ~<code>1.98 Ah</code> to ~<code>1.20 Ah</code>, visually confirming mean non-stationarity. The rolling "
            "standard deviation stays low (~<code>0.015 Ah</code>) during smooth degradation phases and bursts upward "
            "(~<code>0.055 Ah</code>) around rest-regeneration events at Cycles 20, 48, and 90."
        )

    with tab4:
        render_section_header("Distributional Properties of Level Capacity Q_t and Increments ΔQ_t")
        st.plotly_chart(plot_capacity_distribution(df), use_container_width=True)
        render_callout(
            "<b>Diagnostic Interpretation:</b> Level capacity $Q_t$ spans a wide range (<code>1.17–2.05 Ah</code>) with a "
            "higher density in the mid-to-late degradation regime (<code>1.20–1.50 Ah</code>) where the fade slope flattens slightly. "
            "Conversely, $\\Delta Q_t$ is tightly unimodal around <code>-0.0067 Ah/cycle</code> (median) with a heavy positive "
            "right tail caused by discrete rest-regeneration spikes."
        )

    with tab5:
        render_section_header("Training-Set Autocorrelation (ACF) and Partial Autocorrelation (PACF)")
        st.markdown(
            "Computed strictly on the **chronological training set (Cycles 1–134)** to prevent test-set lookahead bias "
            "during Box–Jenkins $(p, d, q)$ identification."
        )
        st.plotly_chart(plot_acf_pacf_interactive(acf_pacf_data, series_mode="level"), use_container_width=True)
        st.plotly_chart(plot_acf_pacf_interactive(acf_pacf_data, series_mode="diff1"), use_container_width=True)
        render_callout(
            "<b>Diagnostic Interpretation:</b> On the level series $Q_t$ ($d=0$), the ACF decays very slowly and nearly linearly "
            "($r_1 = 0.974 \\to r_{10} = 0.772$), while the PACF exhibits a single dominant spike at Lag 1 ($+0.974$) and cuts off "
            "into the 95% Bartlett band—the classic signature of an integrated $I(1)$ process. After first differencing ($\\Delta Q_t$, $d=1$), "
            "long-range autocorrelation vanishes ($r_1 = -0.080, r_2 = -0.145$), justifying low-order $\\text{ARIMA}(p, 1, q)$ candidates."
        )

    with st.expander("View Saved Publication Figures (`results/figures/fig01` – `fig08`)", expanded=False):
        fig_files = [
            ("fig01_capacity_vs_cycle.png", "Figure 1: Discharge Capacity vs. Discharge Cycle"),
            ("fig02_soh_vs_cycle.png", "Figure 2: State of Health (SOH %) vs. Discharge Cycle"),
            ("fig03_capacity_fade_vs_cycle.png", "Figure 3: Cumulative Capacity Fade & Cycle Increments"),
            ("fig04_rolling_statistics.png", "Figure 4: Rolling Mean & Rolling Standard Deviation"),
            ("fig05_acf_plot.png", "Figure 5: Training-Set Autocorrelation Function (ACF)"),
            ("fig06_pacf_plot.png", "Figure 6: Training-Set Partial Autocorrelation Function (PACF)"),
            ("fig07_capacity_distribution.png", "Figure 7: Capacity & First-Difference Distributions"),
            ("fig08_temperature_and_telemetry_vs_capacity.png", "Figure 8: Ambient Temperature & Discharge Duration Telemetry"),
        ]
        cols = st.columns(2)
        for idx, (fname, caption) in enumerate(fig_files):
            fpath = FIGURES_DIR / fname
            if fpath.exists():
                with cols[idx % 2]:
                    st.image(str(fpath), caption=caption, use_container_width=True)

