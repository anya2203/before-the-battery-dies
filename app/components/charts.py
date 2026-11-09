"""
Interactive Plotly chart builders for the 'Before the Battery Dies' Streamlit dashboard.
Uses a dark charcoal + teal/cyan industrial color palette and consistent axis labels.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = {
    "bg_plot": "#0e1520",
    "bg_paper": "#131b26",
    "grid": "#1e293b",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "teal": "#14b8a6",
    "cyan": "#06b6d4",
    "sky": "#38bdf8",
    "amber": "#f59e0b",
    "rose": "#f43f5e",
    "violet": "#a855f7",
    "emerald": "#10b981",
    "slate": "#64748b",
}

MODEL_COLORS = {
    "ARIMA (0,1,1)": "#f43f5e",
    "Model 1: ARIMA(0, 1, 1) — Baseline": "#f43f5e",
    "Model 1: ARIMA(0, 1, 1) Baseline": "#f43f5e",
    "ARIMA (0,1,1) with drift": "#38bdf8",
    "Model 2: ARIMA(0, 1, 1) with Drift": "#38bdf8",
    "Holt Linear Trend": "#14b8a6",
    "Model 3: Holt's Linear Trend (Val-Tuned: alpha=0.10, beta=0.05)": "#14b8a6",
    "Model 3: Holt's Linear Trend (Val-Tuned)": "#14b8a6",
    "Model 3: Holt's Linear Trend (Causally Tuned)": "#14b8a6",
    "Model 3b: Holt's Linear Trend (Train-MLE: alpha=0.88, beta=0.00)": "#10b981",
    "Model 3b: Holt's Linear Trend (Train-MLE)": "#10b981",
    "Damped Holt": "#f59e0b",
    "Model 4: Damped Holt's Trend (Val-Tuned: alpha=0.60, beta=0.05, phi=0.98)": "#f59e0b",
    "Model 4: Damped Holt's Trend": "#f59e0b",
    "Model 4: Damped Holt's Trend (Causally Tuned)": "#f59e0b",
}


def _apply_industrial_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str = "Discharge Cycle",
    yaxis_title: str = "Discharge Capacity (Ah)",
    height: int = 430,
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#f8fafc"), x=0.01),
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        font=dict(color=PALETTE["text"], size=12),
        xaxis=dict(
            title=xaxis_title,
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            linecolor="#334155",
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=PALETTE["grid"],
            zerolinecolor=PALETTE["grid"],
            linecolor="#334155",
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.82)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(size=11, color="#e2e8f0"),
        ),
        margin=dict(l=55, r=28, t=52, b=48),
        height=height,
        hovermode="x unified",
    )
    return fig


def plot_capacity_degradation(df: pd.DataFrame, show_split: bool = True) -> go.Figure:
    """Plot discharge capacity (Ah) vs. sequential discharge cycle with regeneration & EOL markers."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["cycle_number"],
            y=df["discharge_capacity_ah"],
            mode="lines+markers",
            name="Measured Capacity Q_t (Ah)",
            line=dict(color=PALETTE["teal"], width=2.4),
            marker=dict(size=4.5, color=PALETTE["cyan"]),
            hovertemplate="Cycle %{x}<br>Capacity: %{y:.4f} Ah<extra></extra>",
        )
    )

    # Highlight major regeneration spikes (> +0.035 Ah)
    diffs = df["discharge_capacity_ah"].diff().fillna(0.0)
    regen_df = df[diffs > 0.035]
    if not regen_df.empty:
        fig.add_trace(
            go.Scatter(
                x=regen_df["cycle_number"],
                y=regen_df["discharge_capacity_ah"],
                mode="markers",
                name="Rest Regeneration Spikes",
                marker=dict(size=9, color=PALETTE["amber"], symbol="diamond", line=dict(width=1, color="#fff")),
                hovertemplate="Cycle %{x} (Regeneration)<br>Capacity: %{y:.4f} Ah<extra></extra>",
            )
        )

    # NASA B0006 experimental 1.40 Ah reference line
    fig.add_hline(
        y=1.40,
        line_dash="dash",
        line_color=PALETTE["rose"],
        annotation_text="NASA B0006 Stopping Threshold (1.40 Ah — Crossed at Cycle 112)",
        annotation_position="top right",
        annotation_font=dict(color=PALETTE["rose"], size=11),
    )

    if show_split:
        fig.add_vline(
            x=134.5,
            line_dash="dot",
            line_color=PALETTE["amber"],
            annotation_text="Chronological Split (Train 1–134 | Test 135–168)",
            annotation_position="top left",
            annotation_font=dict(color=PALETTE["amber"], size=11),
        )

    return _apply_industrial_layout(
        fig,
        title="NASA Cell B0006 — Discharge Capacity Trajectory Across 168 Cycles",
        xaxis_title="Discharge Cycle",
        yaxis_title="Discharge Capacity (Ah)",
        height=430,
    )


def plot_soh_trend(df: pd.DataFrame) -> go.Figure:
    """Plot primary SOH (%) and nominal rated SOH (%) across discharge cycles."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["cycle_number"],
            y=df["soh_percent"],
            mode="lines",
            name="SOH (%) [Normalized to Q₁ = 2.0520 Ah]",
            line=dict(color=PALETTE["cyan"], width=2.5),
            hovertemplate="Cycle %{x}<br>SOH (Initial Ref): %{y:.2f}%<extra></extra>",
        )
    )

    if "soh_rated_percent" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["cycle_number"],
                y=df["soh_rated_percent"],
                mode="lines",
                name="Rated SOH (%) [Normalized to 2.0000 Ah Nominal]",
                line=dict(color=PALETTE["teal"], width=1.8, dash="dot"),
                hovertemplate="Cycle %{x}<br>SOH (2.0 Ah Rated): %{y:.2f}%<extra></extra>",
            )
        )

    fig.add_hline(
        y=80.0,
        line_dash="dash",
        line_color=PALETTE["amber"],
        annotation_text="80% Initial SOH (Cycle 61)",
        annotation_position="top right",
        annotation_font=dict(color=PALETTE["amber"], size=11),
    )
    fig.add_hline(
        y=70.0,
        line_dash="dash",
        line_color=PALETTE["rose"],
        annotation_text="70% Initial SOH (Cycle 103)",
        annotation_position="bottom right",
        annotation_font=dict(color=PALETTE["rose"], size=11),
    )

    return _apply_industrial_layout(
        fig,
        title="State of Health (SOH %) Degradation Trajectory",
        xaxis_title="Discharge Cycle",
        yaxis_title="State of Health (%)",
        height=410,
    )


def plot_capacity_fade_and_diff(df: pd.DataFrame) -> go.Figure:
    """Plot cumulative capacity fade (Ah) and cycle-to-cycle increments ΔQ_t."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Cumulative Capacity Fade: Fade_t = Q₁ − Q_t (Ah)",
            "Cycle-to-Cycle Capacity Change: ΔQ_t = Q_t − Q_{t−1} (Ah/cycle)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df["cycle_number"],
            y=df["capacity_fade_ah"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(20, 184, 166, 0.16)",
            line=dict(color=PALETTE["teal"], width=2.2),
            name="Cumulative Fade (Ah)",
        ),
        row=1,
        col=1,
    )

    delta_q = df["discharge_capacity_ah"].diff()
    colors = [PALETTE["amber"] if v > 0 else PALETTE["cyan"] for v in delta_q.fillna(0.0)]
    fig.add_trace(
        go.Bar(
            x=df["cycle_number"],
            y=delta_q,
            marker_color=colors,
            name="ΔQ_t (Ah/cycle)",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line_color="#475569", line_width=1, row=2, col=1)

    fig.update_layout(
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        font=dict(color=PALETTE["text"], size=12),
        height=540,
        margin=dict(l=55, r=28, t=55, b=45),
        showlegend=True,
        legend=dict(bgcolor="rgba(15, 23, 42, 0.82)", bordercolor="#1e293b", borderwidth=1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], title_text="Discharge Cycle", row=2, col=1)
    fig.update_xaxes(gridcolor=PALETTE["grid"], row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="Capacity Fade (Ah)", row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="ΔQ_t (Ah)", row=2, col=1)
    return fig


def plot_rolling_statistics(df: pd.DataFrame) -> go.Figure:
    """Plot rolling 10-cycle and 20-cycle mean and standard deviation."""
    q = df["discharge_capacity_ah"]
    roll_mean_10 = q.rolling(10, min_periods=1).mean()
    roll_mean_20 = q.rolling(20, min_periods=1).mean()
    roll_std_10 = q.rolling(10, min_periods=2).std()
    roll_std_20 = q.rolling(20, min_periods=2).std()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Rolling Mean vs. Raw Discharge Capacity (Confirming Mean Non-Stationarity)",
            "Rolling Standard Deviation (Localized Variance Bursts at Regeneration Cycles)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df["cycle_number"],
            y=q,
            mode="lines",
            name="Raw Capacity Q_t",
            line=dict(color="#64748b", width=1.4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["cycle_number"],
            y=roll_mean_10,
            mode="lines",
            name="10-Cycle Rolling Mean",
            line=dict(color=PALETTE["teal"], width=2.3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["cycle_number"],
            y=roll_mean_20,
            mode="lines",
            name="20-Cycle Rolling Mean",
            line=dict(color=PALETTE["amber"], width=2.2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["cycle_number"],
            y=roll_std_10,
            mode="lines",
            name="10-Cycle Rolling Std Dev",
            line=dict(color=PALETTE["cyan"], width=2.0),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["cycle_number"],
            y=roll_std_20,
            mode="lines",
            name="20-Cycle Rolling Std Dev",
            line=dict(color=PALETTE["rose"], width=2.0, dash="dot"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        font=dict(color=PALETTE["text"], size=12),
        height=540,
        margin=dict(l=55, r=28, t=55, b=45),
        legend=dict(bgcolor="rgba(15, 23, 42, 0.82)", bordercolor="#1e293b", borderwidth=1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], title_text="Discharge Cycle", row=2, col=1)
    fig.update_xaxes(gridcolor=PALETTE["grid"], row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="Discharge Capacity (Ah)", row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="Rolling Std (Ah)", row=2, col=1)
    return fig


def plot_capacity_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot distributions of level capacity Q_t and first differences ΔQ_t."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Distribution of Discharge Capacity Q_t (Ah)",
            "Distribution of Cycle Increments ΔQ_t (Ah/cycle)",
        ),
    )

    fig.add_trace(
        go.Histogram(
            x=df["discharge_capacity_ah"],
            nbinsx=22,
            marker_color=PALETTE["teal"],
            opacity=0.85,
            name="Level Q_t (Ah)",
        ),
        row=1,
        col=1,
    )

    delta_q = df["discharge_capacity_ah"].diff().dropna()
    fig.add_trace(
        go.Histogram(
            x=delta_q,
            nbinsx=28,
            marker_color=PALETTE["cyan"],
            opacity=0.85,
            name="First Diff ΔQ_t (Ah)",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        font=dict(color=PALETTE["text"], size=12),
        height=400,
        margin=dict(l=50, r=28, t=55, b=45),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], title_text="Discharge Capacity (Ah)", row=1, col=1)
    fig.update_xaxes(gridcolor=PALETTE["grid"], title_text="ΔQ_t (Ah/cycle)", row=1, col=2)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="Cycle Count", row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="Cycle Count", row=1, col=2)
    return fig


def plot_acf_pacf_interactive(acf_pacf_dict: Dict[str, Any], series_mode: str = "level") -> go.Figure:
    """Plot interactive ACF and PACF stem/bar charts with 95% confidence bands."""
    if series_mode == "level":
        acf_vals = acf_pacf_dict["acf_level"]
        pacf_vals = acf_pacf_dict["pacf_level"]
        ci = acf_pacf_dict["ci_level"]
        subtitle_prefix = "Level Series Q_t (d = 0, Training Cycles 1–134)"
    else:
        acf_vals = acf_pacf_dict["acf_diff1"]
        pacf_vals = acf_pacf_dict["pacf_diff1"]
        ci = acf_pacf_dict["ci_diff1"]
        subtitle_prefix = "First-Differenced Series ΔQ_t (d = 1, Training Cycles 2–134)"

    lags = np.arange(len(acf_vals))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"ACF — {subtitle_prefix}",
            f"PACF — {subtitle_prefix}",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf_vals,
            marker_color=PALETTE["teal"],
            width=0.35,
            name="ACF",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=lags,
            y=pacf_vals,
            marker_color=PALETTE["cyan"],
            width=0.35,
            name="PACF",
        ),
        row=1,
        col=2,
    )

    for col_idx in (1, 2):
        fig.add_hline(y=ci, line_dash="dash", line_color=PALETTE["amber"], row=1, col=col_idx)
        fig.add_hline(y=-ci, line_dash="dash", line_color=PALETTE["amber"], row=1, col=col_idx)
        fig.add_hline(y=0.0, line_color="#475569", row=1, col=col_idx)

    fig.update_layout(
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        font=dict(color=PALETTE["text"], size=12),
        height=400,
        margin=dict(l=50, r=28, t=55, b=45),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], title_text="Lag (Cycles)", row=1, col=1)
    fig.update_xaxes(gridcolor=PALETTE["grid"], title_text="Lag (Cycles)", row=1, col=2)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="Autocorrelation", range=[-0.45, 1.08], row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="Partial Autocorrelation", range=[-0.45, 1.08], row=1, col=2)
    return fig


def plot_stationarity_comparison(df: pd.DataFrame, train_only: bool = True) -> go.Figure:
    """Plot original level capacity Q_t vs. first-order differenced series ΔQ_t."""
    sub = df[df["cycle_number"] <= 134].copy() if train_only else df.copy()
    scope_label = "Training Set (Cycles 1–134)" if train_only else "Full Series (Cycles 1–168)"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            f"Original Level Series Q_t (d = 0) — {scope_label} [Non-Stationary]",
            f"First-Order Differenced Series ΔQ_t = Q_t − Q_{{t−1}} (d = 1) — {scope_label} [Stationary]",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=sub["cycle_number"],
            y=sub["discharge_capacity_ah"],
            mode="lines+markers",
            name="Level Series Q_t (Ah)",
            line=dict(color=PALETTE["teal"], width=2.2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    diff_q = sub["discharge_capacity_ah"].diff()
    fig.add_trace(
        go.Scatter(
            x=sub["cycle_number"].iloc[1:],
            y=diff_q.iloc[1:],
            mode="lines+markers",
            name="First Difference ΔQ_t (Ah/cycle)",
            line=dict(color=PALETTE["cyan"], width=1.9),
            marker=dict(size=4, color=PALETTE["amber"]),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=float(diff_q.iloc[1:].mean()),
        line_dash="dash",
        line_color=PALETTE["rose"],
        annotation_text=f"Mean ΔQ = { float(diff_q.iloc[1:].mean()):.5f} Ah/cycle",
        annotation_position="bottom right",
        row=2,
        col=1,
    )

    fig.update_layout(
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        font=dict(color=PALETTE["text"], size=12),
        height=540,
        margin=dict(l=55, r=28, t=55, b=45),
        legend=dict(bgcolor="rgba(15, 23, 42, 0.82)", bordercolor="#1e293b", borderwidth=1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], title_text="Discharge Cycle", row=2, col=1)
    fig.update_xaxes(gridcolor=PALETTE["grid"], row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="Discharge Capacity (Ah)", row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="ΔQ_t (Ah/cycle)", row=2, col=1)
    return fig


def plot_forecast_lab_chart(
    forecast_bundle: Dict[str, Any],
    selected_models: List[str],
    show_ci: bool = True,
    show_fitted: bool = False,
) -> go.Figure:
    """
    Plot historical training observations, held-out actual observations, and multi-step forecasts
    for all user-selected models from `origin_cycle`.
    """
    train_df = forecast_bundle["train_df"]
    future_actual_df = forecast_bundle["future_actual_df"]
    future_cycles = forecast_bundle["future_cycles"]
    origin_cycle = forecast_bundle["origin_cycle"]
    horizon = forecast_bundle["horizon"]

    fig = go.Figure()

    # Historical training observations
    fig.add_trace(
        go.Scatter(
            x=train_df["cycle_number"],
            y=train_df["discharge_capacity_ah"],
            mode="lines",
            name=f"Historical Training Actual (Cycles 1–{origin_cycle})",
            line=dict(color="#94a3b8", width=2.2),
            hovertemplate="Cycle %{x}<br>Training Actual: %{y:.4f} Ah<extra></extra>",
        )
    )

    # Held-out actual observations in forecast window (if available)
    if not future_actual_df.empty:
        fig.add_trace(
            go.Scatter(
                x=future_actual_df["cycle_number"],
                y=future_actual_df["discharge_capacity_ah"],
                mode="lines+markers",
                name=f"Held-Out Actual Capacity (Cycles {origin_cycle + 1}–{int(future_actual_df['cycle_number'].max())})",
                line=dict(color="#f8fafc", width=2.6),
                marker=dict(size=5.5, color="#f8fafc"),
                hovertemplate="Cycle %{x}<br>Actual Capacity: %{y:.4f} Ah<extra></extra>",
            )
        )

    for m_key in selected_models:
        if m_key not in forecast_bundle["models"]:
            continue
        m_info = forecast_bundle["models"][m_key]
        color = MODEL_COLORS.get(m_key, PALETTE["teal"])

        # Optional 95% confidence interval for ARIMA models that provide analytical state-space intervals
        if show_ci and m_info["has_ci"] and m_info["ci_lower"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([future_cycles, future_cycles[::-1]]),
                    y=np.concatenate([m_info["ci_upper"], m_info["ci_lower"][::-1]]),
                    fill="toself",
                    fillcolor=(
                        "rgba(244, 63, 94, 0.12)"
                        if "drift" not in m_key.lower()
                        else "rgba(56, 189, 248, 0.14)"
                    ),
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"{m_key} 95% Prediction Interval",
                )
            )

        if show_fitted:
            fitted = m_info["fitted_values"]
            start_idx = 1 if "ARIMA" in m_key else 0
            fig.add_trace(
                go.Scatter(
                    x=train_df["cycle_number"].iloc[start_idx:],
                    y=fitted[start_idx:],
                    mode="lines",
                    name=f"{m_key} (In-Sample Fit)",
                    line=dict(color=color, width=1.3, dash="dot"),
                    opacity=0.65,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=future_cycles,
                y=m_info["forecast"],
                mode="lines+markers",
                name=f"{m_key} Forecast (h={horizon})",
                line=dict(color=color, width=2.6, dash="dash"),
                marker=dict(size=5, color=color),
                hovertemplate=f"Cycle %{{x}}<br>{m_key}: %{{y:.4f}} Ah<extra></extra>",
            )
        )

    fig.add_vline(
        x=origin_cycle + 0.5,
        line_dash="dot",
        line_color=PALETTE["amber"],
        annotation_text=f"Forecast Origin (Cycle {origin_cycle})",
        annotation_position="top left",
        annotation_font=dict(color=PALETTE["amber"], size=11),
    )

    return _apply_industrial_layout(
        fig,
        title=f"Multi-Step Capacity Forecasts from Origin Cycle {origin_cycle} (Horizon H = {horizon} Cycles)",
        xaxis_title="Discharge Cycle",
        yaxis_title="Discharge Capacity (Ah)",
        height=470,
    )


def plot_model_comparison_bars(comp_df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart of MAE, RMSE, MAPE, and sMAPE across held-out test models."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Absolute & Squared Error on Held-Out Test Set (Ah)",
            "Percentage Error on Held-Out Test Set (%)",
        ),
    )

    short_labels = [
        m.replace("Model 1: ", "M1: ")
        .replace("Model 2: ", "M2: ")
        .replace("Model 3: ", "M3: ")
        .replace("Model 3b: ", "M3b: ")
        .replace("Model 4: ", "M4: ")
        .split(" (")[0]
        + (" (MLE)" if "3b" in m else (" (Val)" if "Model 3:" in m else ""))
        for m in comp_df["Model"]
    ]

    fig.add_trace(
        go.Bar(
            x=short_labels,
            y=comp_df["MAE_Ah"],
            name="MAE (Ah)",
            marker_color=PALETTE["teal"],
            text=[f"{v:.4f}" for v in comp_df["MAE_Ah"]],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=short_labels,
            y=comp_df["RMSE_Ah"],
            name="RMSE (Ah)",
            marker_color=PALETTE["cyan"],
            text=[f"{v:.4f}" for v in comp_df["RMSE_Ah"]],
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=short_labels,
            y=comp_df["MAPE_Percent"],
            name="MAPE (%)",
            marker_color=PALETTE["amber"],
            text=[f"{v:.2f}%" for v in comp_df["MAPE_Percent"]],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=short_labels,
            y=comp_df["sMAPE_Percent"],
            name="sMAPE (%)",
            marker_color=PALETTE["rose"],
            text=[f"{v:.2f}%" for v in comp_df["sMAPE_Percent"]],
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        barmode="group",
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        font=dict(color=PALETTE["text"], size=11),
        height=440,
        margin=dict(l=50, r=28, t=55, b=85),
        legend=dict(bgcolor="rgba(15, 23, 42, 0.82)", bordercolor="#1e293b", borderwidth=1),
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], tickangle=-22, row=1, col=1)
    fig.update_xaxes(gridcolor=PALETTE["grid"], tickangle=-22, row=1, col=2)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="Error (Ah)", range=[0, 0.13], row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="Error (%)", range=[0, 9.8], row=1, col=2)
    return fig


def plot_horizon_comparison_chart(horiz_df: pd.DataFrame) -> go.Figure:
    """Plot RMSE and MAPE scaling across forecast horizons h = 5, 10, 20, 34 cycles."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "RMSE Scaling Across Forecast Horizons (h = 5, 10, 20, 34)",
            "MAPE (%) Scaling Across Forecast Horizons (h = 5, 10, 20, 34)",
        ),
    )

    for model_name, group in horiz_df.groupby("Model", sort=False):
        color = MODEL_COLORS.get(model_name, PALETTE["teal"])
        fig.add_trace(
            go.Scatter(
                x=group["Horizon_h"],
                y=group["RMSE_Ah"],
                mode="lines+markers",
                name=model_name,
                line=dict(color=color, width=2.4),
                marker=dict(size=7, color=color),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=group["Horizon_h"],
                y=group["MAPE_Percent"],
                mode="lines+markers",
                name=model_name,
                line=dict(color=color, width=2.4),
                marker=dict(size=7, color=color),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        font=dict(color=PALETTE["text"], size=11),
        height=420,
        margin=dict(l=55, r=28, t=55, b=48),
        legend=dict(bgcolor="rgba(15, 23, 42, 0.82)", bordercolor="#1e293b", borderwidth=1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], title_text="Forecast Horizon h (Cycles)", tickvals=[5, 10, 20, 34], row=1, col=1)
    fig.update_xaxes(gridcolor=PALETTE["grid"], title_text="Forecast Horizon h (Cycles)", tickvals=[5, 10, 20, 34], row=1, col=2)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="RMSE (Ah)", row=1, col=1)
    fig.update_yaxes(gridcolor=PALETTE["grid"], title_text="MAPE (%)", row=1, col=2)
    return fig


def plot_rolling_origin_chart(roll_eval_df: pd.DataFrame) -> go.Figure:
    """Plot 20-cycle ahead RMSE across the 5 chronological forecast origins (90, 105, 120, 134, 148)."""
    fig = go.Figure()

    origins = [f"Origin {int(c)}<br>(C{int(c)+1}–{int(c)+20})" for c in roll_eval_df["train_end_cycle"]]

    series_spec = [
        ("Model 1: ARIMA(0,1,1) Baseline", "arima_rmse", PALETTE["rose"]),
        ("Model 2: ARIMA(0,1,1) with Drift", "arima_drift_rmse", PALETTE["sky"]),
        ("Model 3: Holt's Linear Trend (Causal)", "holt_rmse", PALETTE["teal"]),
        ("Model 4: Damped Holt's Trend (Causal)", "damped_holt_rmse", PALETTE["amber"]),
    ]

    for label, col_name, color in series_spec:
        if col_name in roll_eval_df.columns:
            fig.add_trace(
                go.Bar(
                    x=origins,
                    y=roll_eval_df[col_name],
                    name=label,
                    marker_color=color,
                    text=[f"{v:.4f}" for v in roll_eval_df[col_name]],
                    textposition="outside",
                )
            )

    fig.update_layout(
        barmode="group",
        title=dict(
            text="Leakage-Free Rolling-Origin Cross-Validation (20-Cycle Horizon Across 5 Origins)",
            font=dict(size=15, color="#f8fafc"),
            x=0.01,
        ),
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        font=dict(color=PALETTE["text"], size=12),
        xaxis=dict(title="Chronological Forecast Origin & Evaluation Window", gridcolor=PALETTE["grid"]),
        yaxis=dict(title="20-Cycle Out-of-Sample RMSE (Ah)", gridcolor=PALETTE["grid"], range=[0, 0.175]),
        legend=dict(bgcolor="rgba(15, 23, 42, 0.82)", bordercolor="#1e293b", borderwidth=1),
        height=430,
        margin=dict(l=55, r=28, t=55, b=48),
    )
    return fig


def plot_rul_trajectory_chart(
    df: pd.DataFrame,
    rul_bundle: Dict[str, Any],
    selected_models: List[str],
) -> go.Figure:
    """Plot RUL threshold-crossing trajectories launched causally from `origin_cycle`."""
    origin_cycle = rul_bundle["origin_cycle"]
    eol_thresh = rul_bundle["eol_threshold_ah"]
    future_cycles = rul_bundle["future_cycles"]

    hist_df = df[df["cycle_number"] <= origin_cycle]
    post_df = df[df["cycle_number"] > origin_cycle]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_df["cycle_number"],
            y=hist_df["discharge_capacity_ah"],
            mode="lines",
            name=f"Observed History Used for Fitting (Cycles 1–{origin_cycle})",
            line=dict(color="#94a3b8", width=2.4),
        )
    )

    if not post_df.empty:
        fig.add_trace(
            go.Scatter(
                x=post_df["cycle_number"],
                y=post_df["discharge_capacity_ah"],
                mode="lines",
                name=f"Actual Post-Origin Trajectory (Cycles {origin_cycle + 1}–168, Retrospective)",
                line=dict(color="#f8fafc", width=2.2, dash="dot"),
            )
        )

    traj_map = {
        "ARIMA (0,1,1)": (
            rul_bundle["arima_trajectory"],
            rul_bundle["arima_predicted_eol_cycle"],
            PALETTE["rose"],
        ),
        "ARIMA (0,1,1) with drift": (
            rul_bundle["arima_drift_trajectory"],
            rul_bundle["arima_drift_predicted_eol_cycle"],
            PALETTE["sky"],
        ),
        "Holt Linear Trend": (
            rul_bundle["holt_trajectory"],
            rul_bundle["holt_predicted_eol_cycle"],
            PALETTE["teal"],
        ),
        "Damped Holt": (
            rul_bundle["damped_trajectory"],
            rul_bundle["damped_predicted_eol_cycle"],
            PALETTE["amber"],
        ),
    }

    for m_name in selected_models:
        if m_name not in traj_map:
            continue
        traj, pred_eol, color = traj_map[m_name]
        fig.add_trace(
            go.Scatter(
                x=future_cycles,
                y=traj,
                mode="lines",
                name=f"{m_name} Trajectory" + (f" (Pred EOL: Cycle {pred_eol})" if pred_eol else " (Never Crosses)"),
                line=dict(color=color, width=2.5, dash="dash"),
            )
        )
        if pred_eol is not None:
            idx = int(pred_eol - origin_cycle - 1)
            if 0 <= idx < len(traj):
                fig.add_trace(
                    go.Scatter(
                        x=[pred_eol],
                        y=[traj[idx]],
                        mode="markers",
                        name=f"{m_name} Crossing (Cycle {pred_eol})",
                        marker=dict(size=11, color=color, symbol="x", line=dict(width=2, color="#fff")),
                    )
                )

    fig.add_hline(
        y=eol_thresh,
        line_dash="dash",
        line_color=PALETTE["rose"],
        annotation_text=f"Selected End-of-Life Capacity Threshold ({eol_thresh:.2f} Ah)",
        annotation_position="bottom left",
        annotation_font=dict(color=PALETTE["rose"], size=11),
    )

    fig.add_vline(
        x=origin_cycle,
        line_dash="dot",
        line_color=PALETTE["amber"],
        annotation_text=f"Forecast Origin (Cycle {origin_cycle})",
        annotation_position="top left",
        annotation_font=dict(color=PALETTE["amber"], size=11),
    )

    if rul_bundle["actual_eol_cycle"] is not None:
        act_c = rul_bundle["actual_eol_cycle"]
        fig.add_vline(
            x=act_c,
            line_dash="dot",
            line_color="#e2e8f0",
            annotation_text=f"Actual Crossing (Cycle {act_c})",
            annotation_position="top right",
            annotation_font=dict(color="#e2e8f0", size=11),
        )

    return _apply_industrial_layout(
        fig,
        title=f"Experimental Remaining Useful Life (RUL) Projection from Origin Cycle {origin_cycle}",
        xaxis_title="Discharge Cycle",
        yaxis_title="Discharge Capacity (Ah)",
        height=470,
    )

