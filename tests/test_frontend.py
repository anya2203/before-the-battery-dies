"""
Automated verification tests for the Streamlit frontend (app/) of Before the Battery Dies.
Verifies data loaders, causal forecast/RUL engines, Plotly figure builders, and all 6 dashboard pages.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.data_loader import (
    load_processed_battery_data,
    load_stationarity_results,
    compute_acf_pacf_arrays,
    load_evaluation_artifacts,
    run_causal_forecast_at_origin,
    run_interactive_rul_analysis,
)
from app.components.charts import (
    plot_capacity_degradation,
    plot_soh_trend,
    plot_capacity_fade_and_diff,
    plot_rolling_statistics,
    plot_capacity_distribution,
    plot_acf_pacf_interactive,
    plot_stationarity_comparison,
    plot_forecast_lab_chart,
    plot_model_comparison_bars,
    plot_horizon_comparison_chart,
    plot_rolling_origin_chart,
    plot_rul_trajectory_chart,
)


METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def _experiment_summary() -> dict:
    return json.loads((METRICS_DIR / "experiment_summary.json").read_text(encoding="utf-8"))


def test_frontend_data_loaders_match_csv_artifacts():
    """Verify that frontend data loaders return exactly the values stored in processed CSVs and JSONs."""
    df = load_processed_battery_data()
    raw_csv = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "battery_capacity.csv")
    assert len(df) == 168
    pd.testing.assert_series_equal(
        df["discharge_capacity_ah"].reset_index(drop=True),
        raw_csv.sort_values("cycle_number")["discharge_capacity_ah"].reset_index(drop=True),
    )
    # Values documented in the report / overview page (B0006)
    assert np.isclose(df["discharge_capacity_ah"].iloc[0], 2.05197, atol=1e-4)
    assert np.isclose(df["discharge_capacity_ah"].min(), 1.17211, atol=1e-4)
    assert np.isclose(df["discharge_capacity_ah"].iloc[-1], 1.20377, atol=1e-4)
    assert np.isclose(df["soh_percent"].min(), 57.121, atol=1e-2)
    assert np.isclose(df["soh_percent"].iloc[-1], 58.664, atol=1e-2)

    stat_res = load_stationarity_results()
    assert stat_res["adf_train_level"]["is_stationary_at_5pct"] is False
    assert stat_res["adf_train_diff1"]["is_stationary_at_5pct"] is True
    assert stat_res["recommended_d"] == 1

    artifacts = load_evaluation_artifacts()
    comp_df = artifacts["model_comparison"]
    assert comp_df is not None and len(comp_df) >= 5
    drift_row = comp_df[comp_df["Model"].str.contains("Drift", case=False)].iloc[0]
    summary = _experiment_summary()
    assert np.isclose(
        float(drift_row["RMSE_Ah"]),
        summary["model_2_arima_with_drift"]["metrics_full_test"]["RMSE"],
        atol=1e-6,
    )


def test_frontend_causal_forecast_and_rul_engines():
    """Verify causal forecast at origin 134 and causal RUL at origin 60 match validated pipeline outputs."""
    fc_134 = run_causal_forecast_at_origin(origin_cycle=134, horizon=34)
    assert fc_134["origin_cycle"] == 134
    assert fc_134["horizon"] == 34
    models_out = fc_134["models"]

    comp_df = pd.read_csv(METRICS_DIR / "model_comparison_metrics.csv")

    def csv_rmse(prefix: str) -> float:
        return float(comp_df.loc[comp_df["Model"].str.startswith(prefix), "RMSE_Ah"].iloc[0])

    assert np.isclose(float(models_out["ARIMA (0,1,1)"]["eval_metrics"]["RMSE"]), csv_rmse("Model 1:"), atol=1e-5)
    assert np.isclose(float(models_out["ARIMA (0,1,1) with drift"]["eval_metrics"]["RMSE"]), csv_rmse("Model 2:"), atol=1e-5)
    assert np.isclose(float(models_out["Holt Linear Trend"]["eval_metrics"]["RMSE"]), csv_rmse("Model 3:"), atol=1e-5)
    assert np.isclose(float(models_out["Damped Holt"]["eval_metrics"]["RMSE"]), csv_rmse("Model 4:"), atol=1e-5)

    # Causal RUL at Origin 60 with 1.40 Ah threshold must match the pipeline's rul_analysis
    rul_60 = run_interactive_rul_analysis(origin_cycle=60, eol_threshold_ah=1.40, max_horizon=115)
    expected = _experiment_summary()["rul_analysis"]
    for key in (
        "actual_eol_cycle",
        "actual_rul_cycles",
        "holt_predicted_eol_cycle",
        "holt_predicted_rul_cycles",
        "arima_drift_predicted_eol_cycle",
        "arima_drift_predicted_rul_cycles",
        "arima_predicted_eol_cycle",
    ):
        assert rul_60[key] == expected[key], f"{key}: {rul_60[key]} != {expected[key]}"
    assert rul_60["actual_eol_cycle"] == 112


def test_frontend_plotly_chart_builders():
    """Verify all Plotly chart builder functions return valid Figure objects."""
    df = load_processed_battery_data()
    acf_pacf = compute_acf_pacf_arrays(max_lags=25, train_only=True)
    artifacts = load_evaluation_artifacts()
    fc_134 = run_causal_forecast_at_origin(origin_cycle=134, horizon=34)
    rul_60 = run_interactive_rul_analysis(origin_cycle=60, eol_threshold_ah=1.40, max_horizon=115)

    figs = [
        plot_capacity_degradation(df),
        plot_soh_trend(df),
        plot_capacity_fade_and_diff(df),
        plot_rolling_statistics(df),
        plot_capacity_distribution(df),
        plot_acf_pacf_interactive(acf_pacf, series_mode="level"),
        plot_acf_pacf_interactive(acf_pacf, series_mode="diff1"),
        plot_stationarity_comparison(df, train_only=True),
        plot_forecast_lab_chart(
            forecast_bundle=fc_134,
            selected_models=["ARIMA (0,1,1)", "ARIMA (0,1,1) with drift", "Holt Linear Trend"],
            show_ci=True,
        ),
        plot_model_comparison_bars(artifacts["model_comparison"]),
        plot_horizon_comparison_chart(artifacts["horizon_comparison"]),
        plot_rolling_origin_chart(artifacts["rolling_origin_eval"]),
        plot_rul_trajectory_chart(
            df=df,
            rul_bundle=rul_60,
            selected_models=["ARIMA (0,1,1)", "ARIMA (0,1,1) with drift", "Holt Linear Trend", "Damped Holt"],
        ),
    ]
    for fig in figs:
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


def _loopback_socketpair_available(timeout_s: float = 5.0) -> bool:
    """
    Streamlit's AppTest creates an asyncio event loop. On Windows, asyncio builds its internal
    self-pipe via a loopback TCP `socket.socketpair()`. In network-isolated environments (e.g. a
    sandbox that blocks loopback), `accept()` blocks forever, hanging the test indefinitely.
    Probe this in a daemon thread with a hard timeout so the suite can never hang.
    """
    import socket
    import threading

    result = {"ok": False}

    def _probe() -> None:
        try:
            a, b = socket.socketpair()
            a.close()
            b.close()
            result["ok"] = True
        except OSError:
            result["ok"] = False

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout_s)
    return (not t.is_alive()) and result["ok"]


def test_all_page_render_functions_execute_in_bare_mode():
    """
    Execute every dashboard page render function directly (Streamlit bare mode, no event loop).
    Widgets return their default values, so this exercises each page's full default code path
    (data loading, causal model fitting, chart building) and fails on any raised exception.
    """
    from app.components.metrics import inject_custom_css
    from app.components.sidebar import PAGES, render_sidebar
    from app.pages.eda import render_eda_page
    from app.pages.evaluation import render_evaluation_page
    from app.pages.forecasting import render_forecasting_page
    from app.pages.overview import render_overview_page
    from app.pages.rul import render_rul_page
    from app.pages.stationarity import render_stationarity_page

    assert len(PAGES) == 6
    inject_custom_css()
    assert render_sidebar() == PAGES[0]
    for render_fn in (
        render_overview_page,
        render_eda_page,
        render_stationarity_page,
        render_forecasting_page,
        render_evaluation_page,
        render_rul_page,
    ):
        render_fn()


def test_streamlit_all_pages_render_without_error():
    """Use Streamlit AppTest to render all 6 dashboard pages and verify zero exceptions."""
    if not _loopback_socketpair_available():
        pytest.skip(
            "Loopback sockets unavailable (network-isolated environment): asyncio cannot create its "
            "event-loop self-pipe, so Streamlit AppTest would block indefinitely in socket.accept()."
        )

    from streamlit.testing.v1 import AppTest

    app_path = str(PROJECT_ROOT / "app" / "streamlit_app.py")
    at = AppTest.from_file(app_path, default_timeout=60)
    at.run(timeout=60)
    assert not at.exception, f"Overview page raised exception: {at.exception}"

    # Inspect radio options in sidebar and navigate to each page
    radio_widget = at.sidebar.radio[0]
    assert len(radio_widget.options) == 6
    for option_label in radio_widget.options:
        at.sidebar.radio[0].set_value(option_label).run(timeout=60)
        assert not at.exception, f"Page '{option_label}' raised exception: {at.exception}"
