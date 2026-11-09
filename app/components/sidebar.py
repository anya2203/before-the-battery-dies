"""
Sidebar navigation and dataset status panel for the Streamlit dashboard.
"""

from __future__ import annotations

import streamlit as st


PAGES = [
    "1. Overview",
    "2. Exploratory Data Analysis",
    "3. Stationarity Analysis",
    "4. Forecasting Lab",
    "5. Model Evaluation",
    "6. Remaining Useful Life",
]


def render_sidebar() -> str:
    """
    Render the industrial battery analytics sidebar navigation and return the selected page key.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.4rem 0 0.8rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 0.9rem;">
                <div style="color: #14b8a6; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                    BATTERY PROGNOSTICS LAB
                </div>
                <div style="color: #f8fafc; font-size: 1.18rem; font-weight: 700; margin-top: 0.15rem;">
                    Before the Battery Dies
                </div>
                <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 0.15rem;">
                    Statistical Time-Series Forecasting of Li-Ion Capacity Degradation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            "Navigation",
            options=PAGES,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div style="margin-top: 1.2rem; padding: 0.85rem; background-color: #131b26; border: 1px solid #1e293b; border-radius: 6px; font-size: 0.79rem; color: #94a3b8;">
                <div style="color: #2dd4bf; font-weight: 700; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.06em; margin-bottom: 0.45rem;">
                    Active Telemetry Profile
                </div>
                <div><b style="color:#e2e8f0;">Cell ID:</b> NASA PCoE B0006</div>
                <div><b style="color:#e2e8f0;">Chemistry:</b> 18650 LiCoO₂ / Graphite</div>
                <div><b style="color:#e2e8f0;">Chamber Temp:</b> 24.0 °C (Isothermal)</div>
                <div><b style="color:#e2e8f0;">Discharge Cycles:</b> 168 (Ops 2–614)</div>
                <div><b style="color:#e2e8f0;">Primary Split:</b> Train 1–134 | Test 135–168</div>
                <div><b style="color:#e2e8f0;">Source DOI:</b> <code>10.17632/7ctkcfnsdw.2</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page

