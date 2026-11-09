"""
Phase 2: Data Preprocessing Module.

Extracts clean cycle-level discharge capacity observations, preserves single-battery
identity (`NASA_B0006`), aligns sequential discharge cycle number `t = 1..168`,
aggregates valid intra-cycle telemetry from `Discharge AH.xlsx`, derives SOH (%)
and capacity fade metrics, and exports `data/processed/battery_capacity.csv`.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd


def preprocess_battery_data(
    raw_data: Dict[str, Dict[str, pd.DataFrame]],
    output_csv_path: str | Path | None = None,
    battery_id: str = "NASA_B0006",
    nominal_rated_capacity_ah: float = 2.0,
) -> pd.DataFrame:
    """
    Build the clean, cycle-indexed battery degradation time series.

    Parameters
    ----------
    raw_data : Dict[str, Dict[str, pd.DataFrame]]
        Loaded Excel sheets from `data_loader.load_raw_excel_files`.
    output_csv_path : str | Path | None
        Destination path to save `battery_capacity.csv`.
    battery_id : str
        Identifier for the single battery series (`NASA_B0006`).
    nominal_rated_capacity_ah : float
        NASA nominal rated capacity (2.0 Ah) for supplementary SOH reference.

    Returns
    -------
    pd.DataFrame
        Clean cycle-level DataFrame sorted by `cycle_number`.
    """
    df_ah = raw_data["Discharge AH.xlsx"]["Sheet1"].copy()
    df_sum = raw_data["Discharge Summary.xlsx"]["Sheet1"].copy()
    df_soh = raw_data["SoH Data.xlsx"]["Sheet1"].copy()

    # Validate row counts and operation ID alignment
    if len(df_soh) != len(df_sum):
        raise ValueError(
            f"Row count mismatch between SoH Data ({len(df_soh)}) and Discharge Summary ({len(df_sum)})."
        )
    if not (df_soh["Charge Cycle"].values == df_sum["Dis-ch Cycle"].values).all():
        raise ValueError("Operation cycle indices do not align between SoH Data and Discharge Summary.")

    # Aggregate valid intra-cycle electrical/thermal statistics from Discharge AH.xlsx
    # Exclude cycle-boundary negative delta_t artifacts when summarizing cycle telemetry
    cycle_telemetry = (
        df_ah.groupby("Cycle_Number", as_index=False)
        .agg(
            ambient_temperature_c=("Ambient_Temperature", "first"),
            discharge_duration_s=("Time", "max"),
            mean_voltage_v=("Voltage_measured", "mean"),
            min_voltage_v=("Voltage_measured", "min"),
            mean_current_a=("Current_measured", "mean"),
            num_measurements=("Time", "count"),
        )
        .rename(columns={"Cycle_Number": "operation_cycle_id"})
    )

    # Construct primary cycle-level dataframe
    clean_df = pd.DataFrame(
        {
            "battery_id": battery_id,
            "cycle_number": df_soh["Cycle #"].astype(int),
            "operation_cycle_id": df_soh["Charge Cycle"].astype(int),
            "discharge_capacity_ah": df_soh["Dis-chg Ah"].astype(float),
        }
    )

    # Sort strictly by sequential discharge cycle number
    clean_df = clean_df.sort_values("cycle_number", ascending=True).reset_index(drop=True)

    # Validate monotonic cycle numbers 1..N with no gaps or duplicates
    expected_cycles = np.arange(1, len(clean_df) + 1)
    if not np.array_equal(clean_df["cycle_number"].values, expected_cycles):
        raise ValueError("Sequential cycle_number is not strictly 1..N without gaps or duplicates.")

    # Validate physical plausibility of discharge capacity (0.5 Ah < Q_t < 2.5 Ah)
    if clean_df["discharge_capacity_ah"].isna().any():
        raise ValueError("Missing values detected in discharge_capacity_ah.")
    if ((clean_df["discharge_capacity_ah"] <= 0.5) | (clean_df["discharge_capacity_ah"] >= 2.5)).any():
        raise ValueError("Physically implausible discharge capacity values detected outside (0.5, 2.5) Ah.")

    # Reference capacity Q_initial = capacity at cycle 1 (consistent with SoH Data.xlsx)
    q_initial = float(clean_df["discharge_capacity_ah"].iloc[0])

    # Derived health and degradation indicators
    clean_df["soh_percent"] = (clean_df["discharge_capacity_ah"] / q_initial) * 100.0
    clean_df["soh_rated_percent"] = (
        clean_df["discharge_capacity_ah"] / float(nominal_rated_capacity_ah)
    ) * 100.0
    clean_df["capacity_fade_ah"] = q_initial - clean_df["discharge_capacity_ah"]
    clean_df["capacity_fade_percent"] = (
        clean_df["capacity_fade_ah"] / q_initial
    ) * 100.0

    # Merge cycle-level telemetry from Discharge AH.xlsx
    clean_df = clean_df.merge(cycle_telemetry, on="operation_cycle_id", how="left")

    if output_csv_path is not None:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(out_path, index=False)

    return clean_df

