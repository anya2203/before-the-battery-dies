# Phase 1 — Critical Methodology & Codebase Audit Report

**Project:** Before the Battery Dies — Forecasting Lithium-Ion Battery Degradation Using Statistical Time-Series Models  
**Dataset:** EV Lithium Ion Battery State of Health Dataset (Mendeley Data V2, DOI: `10.17632/7ctkcfnsdw.2`) / NASA Ames PCoE Cell **B0006** ($N = 168$ sequential discharge cycles)  
**Scope of Audit:** Complete inspection of `src/data_loader.py`, `src/preprocessing.py`, `src/eda.py`, `src/stationarity.py`, `src/models.py`, `src/evaluation.py`, `src/visualization.py`, `src/pipeline.py`, `notebooks/battery_degradation_analysis.ipynb`, `reports/project_report.md`, `reports/technical_blog.md`, `research/literature_review.md`, and `results/metrics/*.csv`.

---

## Executive Summary of Audit

The existing project successfully loads the raw Mendeley V2 / NASA B0006 Excel files, isolates the single battery trajectory (`NASA_B0006`, 168 cycles at $24^\circ\text{C}$), avoids random shuffling, and runs end-to-end. However, a rigorous scientific audit identified **seven methodological weaknesses** relating to subtle hyperparameter lookahead leakage in rolling-origin and RUL evaluations, conflation of multi-step and 1-step online metrics in the main comparison table, incomplete evaluation of `ARIMA(0,1,1) with drift` across horizons and rolling origins, missing `sMAPE` metrics, and incomplete residual white-noise diagnostics for the exponential smoothing models.

Each issue is documented below with its scientific impact, exact correction procedure, and target files.

---

## Detailed Audit Findings

### Issue 1: Inner Validation vs. Full-Training Information Criteria in ARIMA Selection
- **What is wrong:** In `src/models.py` (`select_and_fit_arima`, lines 78–103), candidate `ARIMA(p, 1, q)` models are evaluated on the inner split (`sub_train`: Cycles 1–110, `val`: Cycles 111–134) to compute `val_rmse`, but `m_full = ARIMA(y_train, order=order).fit()` is simultaneously fitted on the entire training sequence (Cycles 1–134, which includes the validation window) and `grid_df` is sorted primarily by full-training `aic` (`sort_values(["aic", "val_rmse"])`). Furthermore, `ARIMA(0,1,1) with drift` (`trend='t'`) was treated only as a secondary diagnostic array rather than a first-class competitor with full validation, residual, and confidence-interval diagnostics.
- **Why it matters scientifically:** When an inner training/validation split (`1–110` vs. `111–134`) is used to select model specifications, in-sample information criteria (`sub_train_aic`, `sub_train_bic`) and out-of-sample validation errors (`val_mae`, `val_rmse`) must be computed strictly on the inner split before refitting on the full training set (`1–134`). Additionally, because battery degradation exhibits a clear downward trend, **Model 2 (`ARIMA(0,1,1) with drift`)** must be elevated to a primary competitor alongside **Model 1 (`ARIMA(0,1,1)` baseline)**, **Model 3 (`Holt's Linear Trend`)**, and **Model 4 (`Damped Holt's Trend`)**.
- **How it should be corrected:**
  1. Compute `sub_train_aic` and `sub_train_bic` strictly on Cycles 1–110 alongside inner validation `val_mae`, `val_rmse`, `val_mape`, and `val_smape` on Cycles 111–134.
  2. Refit each selected model on the complete 134-cycle training set (`Cycles 1–134`) and record full-training `train_aic`, `train_bic`, `train_mae`, `train_rmse`, residual series, Ljung–Box statistics, and 34-cycle test forecasts for both **Model 1 (`ARIMA(0,1,1)` Baseline)** and **Model 2 (`ARIMA(0,1,1)` with Drift)**.
- **Files requiring modification:** `src/models.py`, `src/evaluation.py`, `reports/project_report.md`.

---

### Issue 2: Temporal Lookahead Leakage in Rolling-Origin Evaluation and RUL Estimation
- **What is wrong:**
  1. In `src/pipeline.py` (lines 81–88) and `src/models.py` (`run_rolling_origin_evaluation`, lines 286–333), smoothing parameters $\alpha = 0.10, \beta = 0.05$—which were tuned on `Cycles 111–134`—were passed as fixed arguments into earlier forecast origins at **Cycle 90** (`Cycles 91–110`) and **Cycle 105** (`Cycles 106–125`).
  2. Similarly, `estimate_rul_threshold_crossing` at `origin_cycle = 60` (`src/pipeline.py`, lines 98–106) used $\alpha = 0.10, \beta = 0.05$ selected using validation data from Cycles 111–134.
  3. Rolling-origin evaluation only compared `ARIMA(0,1,1)` and `Holt Linear`, omitting `ARIMA(0,1,1) with Drift` and `Damped Holt's Trend`, and did not report the **standard deviation of errors** across origins or save to `results/metrics/rolling_origin_metrics.csv`.
- **Why it matters scientifically:** Using hyperparameters tuned on Cycles 111–134 when evaluating a forecast launched at Cycle 60, Cycle 90, or Cycle 105 introduces **temporal lookahead leakage**, because information from Cycles 111–134 was not available at Cycle 60 or Cycle 90.
- **How it should be corrected:**
  1. In `run_rolling_origin_evaluation`, at each chronological origin $c$, select Holt and Damped Holt smoothing parameters using **only observations available up to cycle $c$** ($1 \dots c$, via an inner validation window within $1 \dots c$).
  2. Evaluate all four core models (`ARIMA(0,1,1) Baseline`, `ARIMA(0,1,1) with Drift`, `Holt's Linear Trend`, and `Damped Holt's Trend`) at every origin.
  3. Compute and save both per-origin metrics (`results/metrics/rolling_origin_evaluation.csv`) and summary statistics across origins (`n_origins`, `horizon`, `mean_MAE`, `std_MAE`, `mean_RMSE`, `std_RMSE`, `mean_MAPE`, `std_MAPE`, `mean_sMAPE`, `std_sMAPE`) to `results/metrics/rolling_origin_metrics.csv`.
  4. In `estimate_rul_threshold_crossing(origin_cycle=60)`, tune Holt parameters strictly on Cycles 1–60 (e.g., sub-train 1–45, validation 46–60) so zero post-Cycle-60 information enters the RUL projection.
- **Files requiring modification:** `src/models.py`, `src/evaluation.py`, `src/pipeline.py`.

---

### Issue 3: Conflation of Multi-Step Test Forecasts and 1-Step Online Recalibration in Main Comparison Table
- **What is wrong:** In `src/evaluation.py` (`evaluate_all_models`, lines 100–126), `results/metrics/model_comparison_metrics.csv` combined 34-cycle multi-step forecasts (which never see test observations) with `Rolling 1-Step ARIMA` and `Rolling 1-Step Holt` (which observe true test capacity $Q_{t-1}$ at every step $t \in [135, 168]$) and labeled both as `Horizon_Cycles = 34`. Moreover, Training (`Cycles 1–134`) and Inner Validation (`Cycles 111–134`) metrics were not reported alongside Held-Out Test (`Cycles 135–168`) metrics.
- **Why it matters scientifically:** Mixing $h=1$ expanding-window online updates with $h=34$ open-loop multi-step forecasts in one table under `Horizon_Cycles = 34` compares fundamentally different information sets. Furthermore, clearly distinguishing **Training**, **Validation**, and **Held-Out Test** performance is essential to verify that models are neither underfitted nor overfitted.
- **How it should be corrected:**
  1. Structure `results/metrics/model_comparison_metrics.csv` so each core multi-step model (`Model 1: ARIMA(0,1,1) Baseline`, `Model 2: ARIMA(0,1,1) with Drift`, `Model 3: Holt's Linear Trend (Val-Tuned)`, `Model 3b: Holt's Linear Trend (Train-MLE)`, `Model 4: Damped Holt's Trend`) has explicitly labeled rows for `Evaluation_Set = Train (Cycles 1-134)`, `Validation (Cycles 111-134)`, and `Held-Out Test (Cycles 135-168)` (plus a clean held-out test summary table).
  2. Export the 1-step online recalibration experiment to a separate, dedicated table (`results/metrics/online_recalibration_metrics.csv`) with `Horizon_Cycles = 1`.
- **Files requiring modification:** `src/evaluation.py`, `reports/project_report.md`, `notebooks/battery_degradation_analysis.ipynb`.

---

### Issue 4: Missing `sMAPE` Metric and Incomplete Model Coverage in Horizon Analysis
- **What is wrong:**
  1. `src/evaluation.py` (`calculate_metrics`, lines 17–44) computed `MAE`, `RMSE`, and `MAPE`, but omitted Symmetric Mean Absolute Percentage Error (`sMAPE`).
  2. The multi-horizon analysis (`h = 5, 10, 20, 34` cycles) excluded **Model 2 (`ARIMA(0,1,1) with Drift`)** and was saved only to `horizon_sensitivity_metrics.csv` instead of `results/metrics/horizon_comparison.csv`.
- **Why it matters scientifically:** Standard `MAPE` penalizes over-predictions (where $\hat{Q}_t > Q_t$, which occurs when an un-drifted model stays flat while actual capacity drops) asymmetrically compared to under-predictions. Adding `sMAPE` and including all four core models across $h \in \{5, 10, 20, 34\}$ in `results/metrics/horizon_comparison.csv` provides a complete, unbiased multi-horizon comparison.
- **How it should be corrected:**
  1. Add `sMAPE` to `calculate_metrics` in `src/evaluation.py`.
  2. Evaluate all four core models (`ARIMA(0,1,1) Baseline`, `ARIMA(0,1,1) with Drift`, `Holt's Linear Trend`, `Damped Holt's Trend`) at horizons $h = 5, 10, 20, 34$ from the identical forecast origin (Cycle 134) and save to `results/metrics/horizon_comparison.csv`.
- **Files requiring modification:** `src/evaluation.py`, `tests/test_preprocessing.py`.

---

### Issue 5: Incomplete Residual Diagnostics & Unverified White-Noise Assumptions for Smoothing Models
- **What is wrong:** `src/visualization.py` (`Figure 9`) only plotted the residual histogram and residual ACF for `ARIMA(0,1,1)`, omitting a residual time-series plot over training cycles and omitting residual diagnostics and Ljung–Box tests for **Model 2 (`ARIMA(0,1,1) with Drift`)**, **Model 3 (`Holt's Linear Trend`)**, and **Model 4 (`Damped Holt's Trend`)**.
- **Why it matters scientifically:** While `ARIMA(0,1,1)` and `ARIMA(0,1,1) with Drift` have high Ljung–Box $p$-values ($p > 0.60$, consistent with white-noise innovations), validation-tuned `Holt's Linear Trend` ($\alpha = 0.10, \beta = 0.05$) applies heavy exponential smoothing to filter out capacity regeneration spikes. Heavy smoothing ($\alpha = 0.10$) intentionally trades 1-step in-sample tracking for multi-step trend stability, which leaves positive serial correlation in 1-step training residuals! Conversely, MLE-fitted Holt ($\alpha = 0.88, \beta = 0.00$) yields uncorrelated training residuals ($p > 0.60$). This trade-off must be formally tested via Ljung–Box diagnostics and honestly explained rather than assuming all models have white-noise residuals.
- **How it should be corrected:**
  1. Compute Ljung–Box test statistics and $p$-values at lags $5, 10, 15, 20$, residual mean, residual standard deviation, and Shapiro–Wilk test statistics for all final models, saving to `results/metrics/residual_diagnostics.csv`.
  2. Generate a comprehensive multi-model residual diagnostic figure (`results/figures/fig14_comprehensive_residual_diagnostics.png`) displaying the **residual time-series plot**, **residual distribution (histogram + KDE)**, and **residual ACF plot** for each core model, and update `fig09_arima_diagnostics_and_forecast.png` to include the residual time series.
- **Files requiring modification:** `src/models.py`, `src/evaluation.py`, `src/visualization.py`, `reports/project_report.md`.

---

### Issue 6: Standardizing Axis Scales, Labels, and Dedicated Model Forecast Plots
- **What is wrong:** In `src/visualization.py`, axis labels varied across figures (`Sequential Discharge Cycle Number (t)` vs. `Test Cycle Number (t)`), `ARIMA(0,1,1) with Drift` was not plotted in the main forecast comparison figure (`Figure 11`), and individual model vs. actual comparison plots were not arranged on consistent axis scales.
- **Why it matters scientifically:** Visual comparisons of competing forecasting models must share identical coordinate scales, standardized axis titles (`Discharge Cycle` and `Discharge Capacity (Ah)`), and clear legends so differences in slope and bias are immediately apparent without visual distortion.
- **How it should be corrected:**
  1. Update `src/visualization.py` so all forecast comparison plots use standardized labels (`X-axis: Discharge Cycle`, `Y-axis: Discharge Capacity (Ah)`) and shared Y-axis limits.
  2. Generate a dedicated 6-panel publication figure (`results/figures/fig15_actual_vs_predicted_six_panel.png`) containing:
     - Panel 1: Training capacity (Cycles 1–134) and actual test capacity (Cycles 135–168).
     - Panel 2: Model 1 — `ARIMA(0,1,1)` Baseline forecast vs. actual test observations.
     - Panel 3: Model 2 — `ARIMA(0,1,1)` with Drift forecast vs. actual test observations.
     - Panel 4: Model 3 — `Holt's Linear Trend` forecast vs. actual test observations.
     - Panel 5: Combined comparison of all four core models (`ARIMA(0,1,1)`, `ARIMA with Drift`, `Holt Linear`, `Damped Holt`).
     - Panel 6: Forecast error comparison ($e_t = Q_t - \hat{Q}_t$) across the 34 test cycles.
- **Files requiring modification:** `src/visualization.py`.

---

### Issue 7: Full-Series vs. Training-Series ACF/PACF in EDA
- **What is wrong:** In `src/eda.py` (`Figures 5 & 6`), ACF and PACF plots were generated using all 168 cycles (`cap`) rather than the 134-cycle training series (`Cycles 1–134`).
- **Why it matters scientifically:** Although `src/stationarity.py` already computed ADF and numerical ACF/PACF on the 134-cycle training series, plotting ACF/PACF on the training series in `src/eda.py` ensures that visual lag identification ($p, q$) is 100% free of test-set information.
- **How it should be corrected:** Update `src/eda.py` (`Figures 5 & 6`) to compute and label the ACF and PACF on the chronological training series (`Cycles 1–134`, `n = 134`).
- **Files requiring modification:** `src/eda.py`.

