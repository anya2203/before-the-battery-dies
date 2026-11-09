"""
End-to-End Reproducible Pipeline for 'Before the Battery Dies'.

Executes:
- Phase 1: Raw Excel data audit & schema inspection (`reports/data_inspection.md`)
- Phase 2: Preprocessing & SOH calculation (`data/processed/battery_capacity.csv`)
- Phase 3: Exploratory Data Analysis (`results/figures/fig01..fig08`)
- Phase 4: Stationarity analysis (ADF, rolling stats, ACF/PACF)
- Phase 5: Baseline ARIMA(0,1,1) & ARIMA(0,1,1) with Drift modeling
- Phase 6: Enhanced Holt's Linear Trend & Damped Holt modeling
- Phase 7: Fair Evaluation (MAE, RMSE, MAPE, sMAPE across Train/Val/Test splits,
  multi-horizon h=5,10,20,34, leakage-free rolling-origin cross-validation, and
  formal residual white-noise diagnostics)
- Phase 9: Causal RUL threshold-crossing estimation
"""

import json
from pathlib import Path
from typing import Dict, Any

from src.data_loader import run_full_data_audit, save_data_inspection_report
from src.preprocessing import preprocess_battery_data
from src.eda import generate_eda_figures
from src.stationarity import analyze_stationarity
from src.models import (
    chronological_train_test_split,
    select_and_fit_arima,
    select_and_fit_holt,
    run_rolling_origin_evaluation,
    run_rolling_one_step_recalibration,
    estimate_rul_threshold_crossing,
)
from src.evaluation import evaluate_all_models
from src.visualization import generate_model_and_evaluation_figures


def run_complete_pipeline(project_root: str | Path | None = None) -> Dict[str, Any]:
    """
    Execute the complete battery degradation forecasting experiment from raw data.
    """
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "raw"
    processed_csv = root / "data" / "processed" / "battery_capacity.csv"
    figures_dir = root / "results" / "figures"
    results_dir = root / "results"
    inspection_md = root / "reports" / "data_inspection.md"

    # Phase 1: Data Inspection
    raw_data, audit_results = run_full_data_audit(raw_dir)
    save_data_inspection_report(audit_results, inspection_md)

    # Phase 2: Preprocessing
    clean_df = preprocess_battery_data(raw_data, output_csv_path=processed_csv)

    # Phase 3: Exploratory Data Analysis
    eda_results = generate_eda_figures(clean_df, figures_dir)

    # Chronological Train/Test Split (80% train = 134 cycles, 20% test = 34 cycles)
    train_df, test_df = chronological_train_test_split(clean_df, train_ratio=0.80)

    # Phase 4: Stationarity Analysis
    stationarity_results = analyze_stationarity(clean_df, train_end_idx=len(train_df))
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics" / "stationarity_results.json", "w", encoding="utf-8") as f:
        json.dump(stationarity_results, f, indent=2)

    # Phase 5: Baseline ARIMA(0,1,1) & ARIMA(0,1,1) with Drift
    d_order = int(stationarity_results["recommended_d"])
    arima_results = select_and_fit_arima(
        train_df["discharge_capacity_ah"],
        test_horizon=len(test_df),
        d=d_order,
        val_size=24,
    )

    # Phase 6: Holt's Linear Trend & Damped Holt's Trend
    holt_results = select_and_fit_holt(
        train_df["discharge_capacity_ah"],
        test_horizon=len(test_df),
        val_size=24,
    )

    # Leakage-Free Rolling-Origin Evaluation (parameters tuned causally at each origin)
    rolling_origin_df = run_rolling_origin_evaluation(
        clean_df,
        arima_order=arima_results["best_order"],
        cutoffs=[90, 105, 120, 134, 148],
        horizon=20,
    )

    # Online 1-Step Recalibration Extension
    rolling_1step_res = run_rolling_one_step_recalibration(
        train_df["discharge_capacity_ah"],
        test_df["discharge_capacity_ah"],
        arima_order=arima_results["best_order"],
        holt_alpha=holt_results["smoothing_level_alpha"],
        holt_beta=holt_results["smoothing_trend_beta"],
    )

    # Phase 9: Leakage-Free RUL Threshold-Crossing Estimation from Origin Cycle 60
    rul_results = estimate_rul_threshold_crossing(
        clean_df,
        origin_cycle=60,
        max_horizon=108,
        eol_capacity_ah=1.40,
        arima_order=arima_results["best_order"],
    )

    # Phase 7: Comprehensive Fair Evaluation
    eval_results = evaluate_all_models(
        train_df=train_df,
        test_df=test_df,
        arima_res=arima_results,
        holt_res=holt_results,
        rolling_1step_res=rolling_1step_res,
        rolling_origin_df=rolling_origin_df,
        rul_res=rul_results,
        results_dir=results_dir,
        val_size=24,
    )

    # Visualization of Models, Residual Diagnostics & 6-Panel Actual vs. Predicted Comparison
    model_figs = generate_model_and_evaluation_figures(
        df=clean_df,
        train_df=train_df,
        test_df=test_df,
        arima_res=arima_results,
        holt_res=holt_results,
        eval_res=eval_results,
        rolling_origin_df=rolling_origin_df,
        rul_res=rul_results,
        figures_dir=figures_dir,
    )

    return {
        "audit": audit_results,
        "eda": eda_results,
        "stationarity": stationarity_results,
        "arima": arima_results,
        "holt": holt_results,
        "evaluation": eval_results,
        "rolling_origin": rolling_origin_df,
        "rul": rul_results,
        "model_figures": model_figs,
    }


if __name__ == "__main__":
    results = run_complete_pipeline()
    print("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print("\n--- 1. Held-Out Test Set Multi-Step Model Comparison (Cycles 135-168, h=34) ---")
    print(results["evaluation"]["comparison_df"].to_string(index=False))
    print("\n--- 2. Train / Validation / Test Split Breakdown ---")
    print(results["evaluation"]["split_breakdown_df"].to_string(index=False))
    print("\n--- 3. Forecast Horizon Analysis (h = 5, 10, 20, 34 Cycles) ---")
    print(results["evaluation"]["horizon_df"].to_string(index=False))
    print("\n--- 4. Rolling-Origin Summary Metrics (5 Origins, h=20 Cycles) ---")
    print(results["evaluation"]["rolling_origin_summary_df"].to_string(index=False))
    print("\n--- 5. Residual Diagnostics Table ---")
    print(results["evaluation"]["residual_diagnostics_df"].to_string(index=False))
    print("\n--- 6. Online 1-Step Recalibration Metrics (h=1) ---")
    print(results["evaluation"]["online_df"].to_string(index=False))
    print("\n--- 7. Leakage-Free RUL Analysis (Origin Cycle 60, EOL=1.40 Ah) ---")
    for k, v in results["evaluation"]["summary"]["rul_analysis"].items():
        print(f"  {k}: {v}")
