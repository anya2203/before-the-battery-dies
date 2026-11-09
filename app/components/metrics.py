"""
Styling, KPI metric cards, and structured callout components for the industrial battery analytics UI.
"""

from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    """
    Inject dark charcoal + teal/cyan industrial theme CSS for clean typography,
    consistent spacing, and polished metric cards.
    """
    st.markdown(
        """
        <style>
        /* Main background & typography */
        .stApp {
            background-color: #0b0f17;
            color: #e6edf3;
        }
        [data-testid="stSidebar"] {
            background-color: #101622;
            border-right: 1px solid #1e293b;
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
        }
        h1, h2, h3, h4 {
            color: #f0f6fc;
            letter-spacing: -0.015em;
            font-weight: 600;
        }
        /* Custom Header Banner */
        .bms-header {
            background: linear-gradient(135deg, #111927 0%, #0d1520 100%);
            border: 1px solid #1e293b;
            border-left: 4px solid #14b8a6;
            border-radius: 8px;
            padding: 1.15rem 1.4rem;
            margin-bottom: 1.25rem;
        }
        .bms-header-eyebrow {
            color: #14b8a6;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }
        .bms-header-title {
            color: #f8fafc;
            font-size: 1.55rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
        }
        .bms-header-sub {
            color: #94a3b8;
            font-size: 0.92rem;
            margin: 0;
            line-height: 1.45;
        }
        /* KPI Metric Card */
        .kpi-card {
            background-color: #131b26;
            border: 1px solid #1e293b;
            border-top: 3px solid #14b8a6;
            border-radius: 8px;
            padding: 0.95rem 1.1rem;
            height: 100%;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
        }
        .kpi-card-cyan {
            border-top-color: #06b6d4;
        }
        .kpi-card-amber {
            border-top-color: #f59e0b;
        }
        .kpi-card-rose {
            border-top-color: #f43f5e;
        }
        .kpi-label {
            color: #94a3b8;
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
        }
        .kpi-value {
            color: #f8fafc;
            font-size: 1.48rem;
            font-weight: 700;
            font-family: 'Consolas', 'SFMono-Regular', monospace;
            margin-bottom: 0.2rem;
        }
        .kpi-delta {
            color: #2dd4bf;
            font-size: 0.78rem;
            font-weight: 500;
        }
        .kpi-delta-muted {
            color: #64748b;
            font-size: 0.78rem;
        }
        /* Info / Research Callout Box */
        .bms-callout {
            background-color: #111823;
            border: 1px solid #1e293b;
            border-left: 3px solid #06b6d4;
            border-radius: 6px;
            padding: 0.9rem 1.15rem;
            margin: 0.75rem 0 1.1rem 0;
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.55;
        }
        .bms-callout-warn {
            border-left-color: #f59e0b;
            background-color: #161512;
        }
        .bms-section-title {
            color: #e2e8f0;
            font-size: 1.12rem;
            font-weight: 600;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 0.45rem;
            margin-top: 1.2rem;
            margin-bottom: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(eyebrow: str, title: str, subtitle: str) -> None:
    """Render a consistent top-of-page industrial banner."""
    st.markdown(
        f"""
        <div class="bms-header">
            <div class="bms-header-eyebrow">{eyebrow}</div>
            <div class="bms-header-title">{title}</div>
            <div class="bms-header-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    label: str,
    value: str,
    subtitle: str = "",
    accent: str = "teal",
) -> None:
    """Render a styled KPI metric card (`teal`, `cyan`, `amber`, or `rose`)."""
    accent_class = {
        "teal": "kpi-card",
        "cyan": "kpi-card kpi-card-cyan",
        "amber": "kpi-card kpi-card-amber",
        "rose": "kpi-card kpi-card-rose",
    }.get(accent, "kpi-card")

    st.markdown(
        f"""
        <div class="{accent_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str) -> None:
    """Render a clean section divider heading."""
    st.markdown(f'<div class="bms-section-title">{title}</div>', unsafe_allow_html=True)


def render_callout(html_content: str, variant: str = "info") -> None:
    """Render an industrial callout panel (`info` or `warn`)."""
    cls = "bms-callout bms-callout-warn" if variant == "warn" else "bms-callout"
    st.markdown(f'<div class="{cls}">{html_content}</div>', unsafe_allow_html=True)

