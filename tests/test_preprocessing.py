"""
Unit Test Suite for 'Before the Battery Dies' Preprocessing, Splitting, Models, Residual Diagnostics, and Evaluation Metrics.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_excel_files, run_full_data_audit
from src.preprocessing import preprocess_battery_data
from src.stationarity import analyze_stationarity
from src.models import (
    chronological_train_test_split,
    select_and_fit_arima,
    select_and_fit_holt,
    compute_residual_diagnostics,
    run_rolling_origin_evaluation,
)
from src.evaluation import calculate_metrics


RAW_DIR = PROJECT_ROOT / "data" / "raw"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def test_raw_excel_files_exist_and_load():
    raw_data = load_raw_excel_files(RAW_DIR)
    assert "Discharge AH.xlsx" in raw_data
    assert "Discharge Summary.xlsx" in raw_data
    assert "SoH Data.xlsx" in raw_data

    df_ah = raw_data["Discharge AH.xlsx"]["Sheet1"]
    df_sum = raw_data["Discharge Summary.xlsx"]["Sheet1"]
    df_soh = raw_data["SoH Data.xlsx"]["Sheet1"]

    assert df_ah.shape == (50285, 9)
    assert df_sum.shape == (168, 2)
    assert df_soh.shape == (168, 4)


def test_preprocessing_produces_valid_single_battery_series(tmp_path):
    raw_data, _ = run_full_data_audit(RAW_DIR)
    out_csv = tmp_path / "battery_capacity.csv"
    clean_df = preprocess_battery_data(raw_data, output_csv_path=out_csv)

    assert out_csv.exists()
    assert len(clean_df) == 168
    assert clean_df["battery_id"].nunique() == 1
    assert clean_df["battery_id"].iloc[0] == "NASA_B0006"

    # Verify strict sequential cycle order 1..168
    assert list(clean_df["cycle_number"]) == list(range(1, 169))
    assert clean_df["discharge_capacity_ah"].isna().sum() == 0
    assert (clean_df["discharge_capacity_ah"] > 1.0).all()
    assert (clean_df["discharge_capacity_ah"] < 2.2).all()

    # Verify SOH calculation uses only Cycle 1 (no test leakage)
    q_init = clean_df["discharge_capacity_ah"].iloc[0]
    expected_soh = (clean_df["discharge_capacity_ah"] / q_init) * 100.0
    np.testing.assert_allclose(clean_df["soh_percent"].values, expected_soh.values, rtol=1e-6)
    assert np.isclose(clean_df["soh_percent"].iloc[0], 100.0)


def test_chronological_split_has_no_leakage_or_shuffling():
    raw_data = load_raw_excel_files(RAW_DIR)
    clean_df = preprocess_battery_data(raw_data)
    train_df, test_df = chronological_train_test_split(clean_df, train_ratio=0.80)

    assert len(train_df) == 134
    assert len(test_df) == 34
    assert train_df["cycle_number"].max() == 134
    assert test_df["cycle_number"].min() == 135
    assert train_df["cycle_number"].is_monotonic_increasing
    assert test_df["cycle_number"].is_monotonic_increasing
    assert set(train_df["cycle_number"]).isdisjoint(set(test_df["cycle_number"]))


def test_calculate_metrics_exact_values_and_validation():
    y_true = np.array([2.0, 1.5, 1.0])
    y_pred = np.array([1.8, 1.5, 1.2])

    metrics = calculate_metrics(y_true, y_pred)
    assert np.isclose(metrics["MAE"], 0.4 / 3.0)
    assert np.isclose(metrics["RMSE"], np.sqrt(0.08 / 3.0))
    assert np.isclose(metrics["MAPE"], 10.0)
    expected_smape = ((0.4 / 3.8) + 0.0 + (0.4 / 2.2)) / 3.0 * 100.0
    assert np.isclose(metrics["sMAPE"], expected_smape)

    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError, match="empty"):
        calculate_metrics(np.array([]), np.array([]))

    with pytest.raises(ValueError, match="zero values"):
        calculate_metrics(np.array([0.0, 1.0]), np.array([0.1, 1.0]))


def test_stationarity_models_and_residual_diagnostics():
    raw_data = load_raw_excel_files(RAW_DIR)
    clean_df = preprocess_battery_data(raw_data)
    train_df, test_df = chronological_train_test_split(clean_df, train_ratio=0.80)

    stat_res = analyze_stationarity(clean_df, train_end_idx=len(train_df))
    assert stat_res["recommended_d"] == 1
    assert not stat_res["adf_train_level"]["is_stationary_at_5pct"]
    assert stat_res["adf_train_diff1"]["is_stationary_at_5pct"]

    arima_res = select_and_fit_arima(train_df["discharge_capacity_ah"], test_horizon=len(test_df), d=1)
    holt_res = select_and_fit_holt(train_df["discharge_capacity_ah"], test_horizon=len(test_df))

    # Verify Model 1 (ARIMA) and Model 2 (ARIMA with Drift)
    assert len(arima_res["forecast"]) == len(test_df)
    assert len(arima_res["arima_with_drift_forecast"]) == len(test_df)
    assert arima_res["residual_diagnostics"]["White_Noise_At_5Pct"] is True
    assert arima_res["arima_with_drift_diagnostics"]["White_Noise_At_5Pct"] is True

    # Verify Model 3 (Holt Linear) and Model 4 (Damped Holt)
    assert len(holt_res["forecast"]) == len(test_df)
    assert len(holt_res["mle_holt_forecast"]) == len(test_df)
    assert len(holt_res["damped_holt_forecast"]) == len(test_df)
    assert holt_res["mle_holt_diagnostics"]["White_Noise_At_5Pct"] is True


def test_generated_metrics_csv_files_and_rolling_origin_integrity():
    assert (METRICS_DIR / "model_comparison_metrics.csv").exists()
    assert (METRICS_DIR / "train_val_test_metrics.csv").exists()
    assert (METRICS_DIR / "online_recalibration_metrics.csv").exists()
    assert (METRICS_DIR / "horizon_comparison.csv").exists()
    assert (METRICS_DIR / "rolling_origin_metrics.csv").exists()
    assert (METRICS_DIR / "residual_diagnostics.csv").exists()

    comp_df = pd.read_csv(METRICS_DIR / "model_comparison_metrics.csv")
    assert (comp_df["Evaluation_Set"] == "Held-Out Test (Cycles 135-168)").all()
    assert (comp_df["Horizon_Cycles"] == 34).all()
    assert {"MAE_Ah", "RMSE_Ah", "MAPE_Percent", "sMAPE_Percent"}.issubset(comp_df.columns)

    horiz_df = pd.read_csv(METRICS_DIR / "horizon_comparison.csv")
    assert set(horiz_df["Horizon_h"].unique()) == {5, 10, 20, 34}

    roll_summary_df = pd.read_csv(METRICS_DIR / "rolling_origin_metrics.csv")
    assert len(roll_summary_df) == 4
    assert (roll_summary_df["Num_Origins"] == 5).all()
    # Verify Model 3 (Holt) and Model 2 (ARIMA+Drift) beat Model 1 (ARIMA Baseline) across rolling origins
    m1_rmse = roll_summary_df.loc[roll_summary_df["Model"].str.contains("Model 1"), "Mean_RMSE_Ah"].iloc[0]
    m2_rmse = roll_summary_df.loc[roll_summary_df["Model"].str.contains("Model 2"), "Mean_RMSE_Ah"].iloc[0]
    m3_rmse = roll_summary_df.loc[roll_summary_df["Model"].str.contains("Model 3"), "Mean_RMSE_Ah"].iloc[0]
    assert m3_rmse < m1_rmse
    assert m2_rmse < m1_rmse
