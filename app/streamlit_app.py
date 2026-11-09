"""
Main Streamlit Entry Point — Before the Battery Dies:
Forecasting Lithium-Ion Battery Degradation Using Statistical Time-Series Models.

Run from the project root via:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.metrics import inject_custom_css
from app.components.sidebar import render_sidebar
from app.pages.eda import render_eda_page
from app.pages.evaluation import render_evaluation_page
from app.pages.forecasting import render_forecasting_page
from app.pages.overview import render_overview_page
from app.pages.rul import render_rul_page
from app.pages.stationarity import render_stationarity_page


def main() -> None:
    st.set_page_config(
        page_title="Before the Battery Dies | Battery Prognostics Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    selected_page = render_sidebar()

    if selected_page.startswith("1."):
        render_overview_page()
    elif selected_page.startswith("2."):
        render_eda_page()
    elif selected_page.startswith("3."):
        render_stationarity_page()
    elif selected_page.startswith("4."):
        render_forecasting_page()
    elif selected_page.startswith("5."):
        render_evaluation_page()
    elif selected_page.startswith("6."):
        render_rul_page()


if __name__ == "__main__":
    main()

