"""
Phase 3: Exploratory Data Analysis (EDA) Module.

Generates 8 publication-grade figures in `results/figures/` and computes
quantitative summaries and structured interpretations for each visualization.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Ensure Matplotlib uses a writable cache directory inside scratch
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / ".matplotlib"),
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)


def generate_eda_figures(df: pd.DataFrame, figures_dir: str | Path) -> Dict[str, Any]:
    """
    Generate all 8 required EDA figures and return quantitative summary facts
    plus plot file paths.
    """
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cycles = df["cycle_number"].values
    cap = df["discharge_capacity_ah"].values
    soh = df["soh_percent"].values
    soh_rated = df["soh_rated_percent"].values
    fade_ah = df["capacity_fade_ah"].values
    fade_pct = df["capacity_fade_percent"].values
    diff_cap = pd.Series(cap).diff()

    # Identify top regeneration spikes (where capacity jumped upward noticeably between consecutive cycles)
    regen_df = df.copy()
    regen_df["delta_q"] = regen_df["discharge_capacity_ah"].diff()
    top_regen = regen_df.sort_values("delta_q", ascending=False).head(5)

    # 1. Figure 1: Discharge Capacity vs Cycle Number
    fig1_path = fig_dir / "fig01_capacity_vs_cycle.png"
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    ax.plot(cycles, cap, color="#1f4e78", lw=2.2, label="Measured Discharge Capacity (Ah)")
    ax.axhline(1.40, color="#c0392b", ls="--", lw=1.6, label="NASA 30% Rated Fade Threshold (1.40 Ah)")
    ax.axhline(cap[0] * 0.80, color="#e67e22", ls=":", lw=1.6, label=f"80% Initial SOH Threshold ({cap[0]*0.80:.3f} Ah)")
    # Highlight top 3 regeneration spikes
    for _, row in top_regen.head(3).iterrows():
        ax.scatter(row["cycle_number"], row["discharge_capacity_ah"], color="#27ae60", s=55, zorder=5)
        ax.annotate(
            f"Regen Cycle {int(row['cycle_number'])}\n(+{row['delta_q']:.3f} Ah)",
            xy=(row["cycle_number"], row["discharge_capacity_ah"]),
            xytext=(row["cycle_number"] + 4, row["discharge_capacity_ah"] + 0.04),
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.0),
        )
    ax.set_title("Figure 1: Lithium-Ion Battery (NASA B0006) Discharge Capacity vs. Discharge Cycle", fontweight="bold")
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel("Discharge Capacity (Ah)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(fig1_path)
    plt.close(fig)

    # 2. Figure 2: SOH vs Cycle Number
    fig2_path = fig_dir / "fig02_soh_vs_cycle.png"
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    ax.plot(cycles, soh, color="#2980b9", lw=2.2, label=f"SOH Relative to Initial Capacity ({cap[0]:.4f} Ah)")
    ax.plot(cycles, soh_rated, color="#8e44ad", lw=1.8, ls="-.", label="SOH Relative to Nominal Rated Capacity (2.0000 Ah)")
    ax.axhline(80.0, color="#e67e22", ls=":", lw=1.6, label="80% SOH Benchmark")
    ax.axhline(70.0, color="#c0392b", ls="--", lw=1.6, label="70% SOH (NASA 30% Fade Stopping Criterion)")
    ax.set_title("Figure 2: State of Health (SOH %) Trajectory Across 168 Discharge Cycles", fontweight="bold")
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel("State of Health (%)")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(fig2_path)
    plt.close(fig)

    # 3. Figure 3: Capacity Fade vs Cycle Number
    fig3_path = fig_dir / "fig03_capacity_fade_vs_cycle.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), dpi=180, sharex=True)
    ax1.plot(cycles, fade_ah, color="#c0392b", lw=2.2, label="Cumulative Capacity Fade (Ah)")
    ax1.fill_between(cycles, 0, fade_ah, color="#c0392b", alpha=0.12)
    ax1.set_ylabel("Cumulative Fade (Ah)")
    ax1.set_title("Figure 3: Cumulative Capacity Fade and Cycle-to-Cycle Capacity Change", fontweight="bold")
    ax1.legend(loc="upper left")

    colors_bar = ["#27ae60" if v > 0 else "#7f8c8d" for v in diff_cap.fillna(0)]
    ax2.bar(cycles, diff_cap.fillna(0), color=colors_bar, width=0.85, alpha=0.85)
    ax2.axhline(0, color="black", lw=1.0)
    ax2.set_xlabel("Discharge Cycle")
    ax2.set_ylabel("Cycle-to-Cycle ΔQ (Ah)")
    ax2.set_title("Cycle-over-Cycle Capacity Difference ΔQ_t = Q_t - Q_{t-1} (Green = Self-Recharge / Regeneration)")
    fig.tight_layout()
    fig.savefig(fig3_path)
    plt.close(fig)

    # 4. Figure 4: Rolling Mean and Standard Deviation
    fig4_path = fig_dir / "fig04_rolling_statistics.png"
    s_cap = pd.Series(cap, index=cycles)
    roll_mean_10 = s_cap.rolling(window=10).mean()
    roll_std_10 = s_cap.rolling(window=10).std()
    roll_mean_20 = s_cap.rolling(window=20).mean()
    roll_std_20 = s_cap.rolling(window=20).std()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), dpi=180, sharex=True)
    ax1.plot(cycles, cap, color="#bdc3c7", lw=1.5, label="Observed Capacity (Ah)")
    ax1.plot(cycles, roll_mean_10, color="#1f4e78", lw=2.2, label="10-Cycle Rolling Mean")
    ax1.plot(cycles, roll_mean_20, color="#e67e22", lw=2.2, ls="--", label="20-Cycle Rolling Mean")
    ax1.set_ylabel("Discharge Capacity (Ah)")
    ax1.set_title("Figure 4: Rolling Mean and Rolling Standard Deviation (Stationarity Diagnostics)", fontweight="bold")
    ax1.legend(loc="upper right")

    ax2.plot(cycles, roll_std_10, color="#2980b9", lw=2.0, label="10-Cycle Rolling Std Dev")
    ax2.plot(cycles, roll_std_20, color="#8e44ad", lw=2.0, ls="--", label="20-Cycle Rolling Std Dev")
    ax2.set_xlabel("Discharge Cycle")
    ax2.set_ylabel("Rolling Std Dev (Ah)")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fig4_path)
    plt.close(fig)

    # Use strictly the chronological training series (Cycles 1..134) for ACF/PACF lag identification
    train_cap = cap[:134]
    train_diff_cap = pd.Series(train_cap).diff().dropna()

    # 5. Figure 5: ACF Plot (Level vs First-Differenced on Training Series)
    fig5_path = fig_dir / "fig05_acf_plot.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    plot_acf(train_cap, lags=35, ax=ax1, color="#1f4e78", vlines_kwargs={"colors": "#1f4e78"})
    ax1.set_title("ACF of Training Capacity Series Q_t (Cycles 1–134)", fontweight="bold")
    ax1.set_xlabel("Lag (Cycles)")
    ax1.set_ylabel("Autocorrelation")

    plot_acf(train_diff_cap, lags=35, ax=ax2, color="#27ae60", vlines_kwargs={"colors": "#27ae60"})
    ax2.set_title("ACF of First-Differenced Training Series ΔQ_t", fontweight="bold")
    ax2.set_xlabel("Lag (Cycles)")
    ax2.set_ylabel("Autocorrelation")
    fig.suptitle("Figure 5: Training-Set Autocorrelation Function (ACF) Before and After First Differencing", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig5_path, bbox_inches="tight")
    plt.close(fig)

    # 6. Figure 6: PACF Plot (Level vs First-Differenced on Training Series)
    fig6_path = fig_dir / "fig06_pacf_plot.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    plot_pacf(train_cap, lags=35, method="ywm", ax=ax1, color="#c0392b", vlines_kwargs={"colors": "#c0392b"})
    ax1.set_title("PACF of Training Capacity Series Q_t (Cycles 1–134)", fontweight="bold")
    ax1.set_xlabel("Lag (Cycles)")
    ax1.set_ylabel("Partial Autocorrelation")

    plot_pacf(train_diff_cap, lags=35, method="ywm", ax=ax2, color="#8e44ad", vlines_kwargs={"colors": "#8e44ad"})
    ax2.set_title("PACF of First-Differenced Training Series ΔQ_t", fontweight="bold")
    ax2.set_xlabel("Lag (Cycles)")
    ax2.set_ylabel("Partial Autocorrelation")
    fig.suptitle("Figure 6: Training-Set Partial Autocorrelation Function (PACF) Before and After First Differencing", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig6_path, bbox_inches="tight")
    plt.close(fig)

    # 7. Figure 7: Capacity Distribution
    fig7_path = fig_dir / "fig07_capacity_distribution.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    sns.histplot(cap, bins=20, kde=True, color="#1f4e78", ax=ax1)
    ax1.axvline(np.mean(cap), color="#c0392b", ls="--", lw=1.8, label=f"Mean = {np.mean(cap):.3f} Ah")
    ax1.axvline(np.median(cap), color="#27ae60", ls=":", lw=1.8, label=f"Median = {np.median(cap):.3f} Ah")
    ax1.set_title("Distribution of Discharge Capacity Q_t (Ah)", fontweight="bold")
    ax1.set_xlabel("Discharge Capacity (Ah)")
    ax1.set_ylabel("Frequency (Cycles)")
    ax1.legend()

    sns.histplot(diff_cap.dropna(), bins=25, kde=True, color="#e67e22", ax=ax2)
    ax2.axvline(diff_cap.mean(), color="#c0392b", ls="--", lw=1.8, label=f"Mean ΔQ = {diff_cap.mean():.4f} Ah/cycle")
    ax2.set_title("Distribution of Cycle-to-Cycle Change ΔQ_t (Ah)", fontweight="bold")
    ax2.set_xlabel("First Difference ΔQ_t (Ah)")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    fig.suptitle("Figure 7: Distributional Analysis of Level Capacity and Cycle-to-Cycle Capacity Changes", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig7_path, bbox_inches="tight")
    plt.close(fig)

    # 8. Figure 8: Ambient Temperature & Cycle Telemetry vs Capacity
    fig8_path = fig_dir / "fig08_temperature_and_telemetry_vs_capacity.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    ax1.scatter(df["cycle_number"], df["ambient_temperature_c"], color="#e74c3c", s=25, alpha=0.8)
    ax1.set_ylim(20, 28)
    ax1.set_title("Ambient Temperature vs. Cycle Number (Constant 24 °C)", fontweight="bold")
    ax1.set_xlabel("Sequential Discharge Cycle Number (t)")
    ax1.set_ylabel("Ambient Temperature (°C)")
    ax1.annotate(
        "Isothermal Chamber Experiment:\nAmbient_Temperature = 24 °C (Std = 0.0 °C)\nacross all 168 cycles",
        xy=(84, 24),
        xytext=(35, 25.8),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fef9e7", ec="#f39c12"),
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2),
    )

    sc = ax2.scatter(
        df["discharge_duration_s"] / 60.0,
        df["discharge_capacity_ah"],
        c=df["cycle_number"],
        cmap="viridis",
        s=35,
        alpha=0.9,
    )
    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label("Cycle Number (t)")
    ax2.set_title("Discharge Duration (min) vs. Capacity (Ah)", fontweight="bold")
    ax2.set_xlabel("Cycle Discharge Duration (minutes)")
    ax2.set_ylabel("Discharge Capacity (Ah)")
    fig.suptitle("Figure 8: Operating Temperature Verification and Discharge Duration vs. Capacity", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig8_path, bbox_inches="tight")
    plt.close(fig)

    # Compute exact threshold crossing cycles in observed data
    cross_80_soh = df[df["soh_percent"] < 80.0]["cycle_number"].min()
    cross_70_soh = df[df["soh_percent"] < 70.0]["cycle_number"].min()
    cross_1_4_ah = df[df["discharge_capacity_ah"] < 1.40]["cycle_number"].min()

    return {
        "figures": {
            "fig01": str(fig1_path),
            "fig02": str(fig2_path),
            "fig03": str(fig3_path),
            "fig04": str(fig4_path),
            "fig05": str(fig5_path),
            "fig06": str(fig6_path),
            "fig07": str(fig7_path),
            "fig08": str(fig8_path),
        },
        "summary_stats": {
            "n_cycles": int(len(df)),
            "q_initial_ah": float(cap[0]),
            "q_final_ah": float(cap[-1]),
            "q_min_ah": float(np.min(cap)),
            "q_min_cycle": int(df.loc[df["discharge_capacity_ah"].idxmin(), "cycle_number"]),
            "q_mean_ah": float(np.mean(cap)),
            "q_std_ah": float(np.std(cap, ddof=1)),
            "q_median_ah": float(np.median(cap)),
            "total_fade_ah": float(fade_ah[-1]),
            "total_fade_percent": float(fade_pct[-1]),
            "mean_cycle_delta_q_ah": float(diff_cap.mean()),
            "std_cycle_delta_q_ah": float(diff_cap.std()),
            "positive_regen_cycles_count": int((diff_cap > 0).sum()),
            "max_regen_jump_ah": float(diff_cap.max()),
            "max_regen_cycle": int(df.loc[diff_cap.idxmax(), "cycle_number"]),
            "first_cycle_below_80pct_soh": int(cross_80_soh),
            "first_cycle_below_70pct_soh": int(cross_70_soh),
            "first_cycle_below_1_40ah": int(cross_1_4_ah),
            "ambient_temp_unique_c": df["ambient_temperature_c"].unique().tolist(),
        },
    }

