# Final Validation Checklist — Before the Battery Dies

**Project:** Forecasting Lithium-Ion Battery Degradation Using Statistical Time-Series Models  
**Primary Dataset:** EV Lithium Ion Battery State of Health Dataset (Mendeley Data V2, DOI: `10.17632/7ctkcfnsdw.2`) / NASA Ames PCoE Cell **B0006** (`168` discharge cycles)  
**Audit Date:** November 9, 2026  

---

## 1. Summary Checklist Table

| # | Audit Requirement | Status | Primary Evidence / Verified Files |
| :- | :--- | :---: | :--- |
| **1.1** | **Holt MLE vs. Validation-Tuned Holt Parameter & Result Divergence Explained** | **PASS** | `src/models.py` (`select_and_fit_holt`, lines 280–449), `results/metrics/holt_parameter_tuning.csv`, `reports/project_report.md` (Section 13.2) |
| **1.2** | **Zero Test-Data Leakage in Both Holt Configurations** | **PASS** | `src/models.py` (lines 288–375): both fit strictly on `y_train` (`Cycles 1–134`) or inner `sub_train` (`Cycles 1–110`) |
| **1.3** | **Primary Enhanced Model Selection Rule Justified** | **PASS** | `Model 3: Holt's Linear Trend (Val-Tuned: alpha=0.10, beta=0.05)` selected via multi-step inner validation RMSE (`0.01732 Ah` vs. `0.04683 Ah` for `MLE_Linear`) prior to test evaluation |
| **2.1** | **Rolling-Origin Validation Across All 5 Origins (`90, 105, 120, 134, 148`)** | **PASS** | `src/models.py` (`run_rolling_origin_evaluation`, lines 452–539), `results/metrics/rolling_origin_evaluation.csv`, `results/metrics/rolling_origin_metrics.csv` |
| **2.2** | **Causal Fitting & Parameter Tuning Using Only `Cycles 1..c` at Each Origin** | **PASS** | `src/models.py` (`_tune_holt_causal`, lines 84–129; `train_y = cap[:c]` at line 482) |
| **2.3** | **Identical Forecast Horizons (`h = 20`) & Non-Lookahead Evaluation Windows** | **PASS** | `test_y = cap[c : c + 20]` (`Cycles 91–110`, `106–125`, `121–140`, `135–154`, `149–168`) |
| **3.1** | **Residual Generation & Ljung–Box Implementation Verified** | **PASS** | `src/models.py` (`compute_residual_diagnostics`, lines 42–81; `best_model.resid[d:]` with $N=133$ for $d=1$ ARIMA models; `N=134` for Holt models) |
| **3.2** | **Statistically Accurate Hypothesis-Testing Language (White Noise Not Claimed as "Proven")** | **PASS** | `results/metrics/residual_diagnostics.csv`, `reports/project_report.md` (Section 15.5, Table 8), `reports/technical_blog.md` |
| **3.3** | **All Tested Lags (`5, 10, 15, 20`) & Shapiro–Wilk $p$-Values Reported for All Models** | **PASS** | `results/metrics/residual_diagnostics.csv` and `reports/project_report.md` (Table 8) |
| **4.1** | **Cycle-60 RUL Forecast Uses Only `Cycles 1–60` for Fitting & Tuning** | **PASS** | `src/models.py` (`estimate_rul_threshold_crossing`, lines 581–657; `train_sub = cap[:60]`, `_tune_holt_causal(train_sub, val_size=15)`) |
| **4.2** | **EOL Threshold (`1.40 Ah`), Actual RUL (`52`), and Predicted RUL Verified** | **PASS** | `results/metrics/experiment_summary.json` (`rul_analysis`): Actual EOL = `Cycle 112` (`RUL = 52`); Holt = `Cycle 98` (`RUL = 38`, error `-14`); Drift ARIMA = `Cycle 96` (`RUL = 36`, error `-16`); Baseline ARIMA = `None` |
| **5.1** | **Dataset & Methodological Limitations Discussed in Report** | **PASS** | `reports/project_report.md` (Section 17): single `NASA_B0006` cell, $N=168$ cycles, constant $24^\circ\text{C}$, limited generalization, rest regeneration, linear extrapolation near aging knees |
| **6.1** | **Literature Review References, Authors, Years, Venues, and DOIs Verified** | **PASS** | `research/literature_review.md` and `reports/project_report.md` (Sections 6 & 19): all 7 scholarly papers + 2 dataset citations verified against published records (including corrected authors for `10.3390/en11040820` and exact DOI `10.1016/j.est.2026.121800`) |
| **7.1** | **End-to-End Pipeline Execution from Raw Excel Files (`python -m src.pipeline`)** | **PASS** | Executed from `data/raw/*.xlsx`; regenerates `data/processed/battery_capacity.csv`, `results/metrics/*.csv`, `results/forecasts/*.csv`, and `results/figures/fig01..fig15.png` |
| **7.2** | **Full Automated Unit Test Suite (`pytest`) & Metric Synchronization** | **PASS** | `tests/test_preprocessing.py` (`6 passed`), `reports/project_report.md`, `reports/technical_blog.md`, `notebooks/battery_degradation_analysis.ipynb` |

---

## 2. Detailed Verification Evidence by Audit Item

### Item 1: Holt MLE vs. Validation-Tuned Holt
- **Why Parameters Differ:**
  - `Model 3b (Train-MLE Holt)` (`Holt(y_train, initialization_method="estimated").fit(optimized=True)` in `src/models.py`, line 373) minimizes **1-step-ahead in-sample SSE** ($\text{SSE} = 0.08284\text{ Ah}^2$, $\text{AIC} = -982.08$) on `Cycles 1–134`. Because `Cycles 1–134` include sharp rest-regeneration jumps (`Cycle 20: +0.1220 Ah`, `Cycle 48: +0.1190 Ah`, `Cycle 90: +0.1518 Ah`), 1-step SSE minimization selects a high level-tracking weight ($\alpha = 0.8756$) and zero slope adaptation ($\beta = 0.0000$), freezing the slope at the initial global average $T_0 = -0.005249\text{ Ah/cycle}$.
  - `Model 3 (Val-Tuned Holt)` (`src/models.py`, lines 298–353) selects $(\alpha, \beta)$ by minimizing **24-step-ahead out-of-sample RMSE** on the inner validation window (`Sub-train: Cycles 1–110` $\to$ `Validation: Cycles 111–134`). Because the fade rate slowed between Cycles 91 and 134 ($-0.0033\text{ Ah/cycle}$) relative to Cycles 1–90 ($-0.0061\text{ Ah/cycle}$), `MLE_Linear` ($\alpha=0.88, \beta=0.00$) over-projected the steep early slope onto `Cycles 111–134` (`val_rmse = 0.04683 Ah`, rank 39 of 62 in `results/metrics/holt_parameter_tuning.csv`), whereas $(\alpha = 0.10, \beta = 0.05)$ smoothed out the Cycle 90 regeneration spike and adapted to the local slope (`val_rmse = 0.01732 Ah`, rank 1 of 62).
- **Why Held-Out Test Results Differ Across Horizons (`Cycles 135–168`):**
  - On short-to-medium test horizons, `Model 3 (Val-Tuned, alpha=0.10, beta=0.05)` is substantially more accurate than `Model 3b (Train-MLE)`:
    - $h = 5$ (`Cycles 135–139`): Val-Tuned $\text{RMSE} = 0.00472\text{ Ah}$ ($\text{MAPE} = 0.306\%$) vs. Train-MLE $\text{RMSE} = 0.01467\text{ Ah}$ ($\text{MAPE} = 1.086\%$).
    - $h = 10$ (`Cycles 135–144`): Val-Tuned $\text{RMSE} = 0.00910\text{ Ah}$ ($\text{MAPE} = 0.591\%$) vs. Train-MLE $\text{RMSE} = 0.01572\text{ Ah}$ ($\text{MAPE} = 1.176\%$).
    - $h = 20$ (`Cycles 135–154`): Val-Tuned $\text{RMSE} = 0.01477\text{ Ah}$ ($\text{MAPE} = 0.983\%$) vs. Train-MLE $\text{RMSE} = 0.01682\text{ Ah}$ ($\text{MAPE} = 1.165\%$).
  - Only over the full 34-cycle horizon ($h = 34$, `Cycles 135–168`) does `Model 3b (Train-MLE)` yield a lower RMSE ($0.01514\text{ Ah}$, $\text{MAPE} = 1.022\%$) than `Model 3 (Val-Tuned)` ($0.02739\text{ Ah}$, $\text{MAPE} = 1.841\%$), because late-life degradation accelerated again between Cycles 152 and 164 ($-0.0057\text{ Ah/cycle}$), aligning with the global training slope ($-0.00525\text{ Ah/cycle}$).
- **Zero Test Leakage & Primary Model Designation:**
  - Neither model accesses `test_df` (`Cycles 135–168`) during parameter estimation or fitting (`select_and_fit_holt` takes only `train_series = train_df["discharge_capacity_ah"]`, `len = 134`).
  - Because hyperparameter selection for a multi-step forecasting task must be decided on the inner multi-step validation split *before* unsealing the test set, **Model 3 (Validation-Tuned Holt, $\alpha = 0.10, \beta = 0.05$) is the primary enhanced model**, while **Model 3b (Train-MLE Holt, $\alpha = 0.88, \beta = 0.00$)** is reported as a diagnostic benchmark.

---

### Item 2: Rolling-Origin Validation (`Origins 90, 105, 120, 134, 148`)
- **Code Inspection (`src/models.py`, lines 452–539):**
  - At each cutoff $c \in \{90, 105, 120, 134, 148\}$, `train_y = cap[:c]` slices strictly the first $c$ observations (`Cycles 1..c`), and `test_y = cap[c : c + 20]` slices the subsequent $h = 20$ observations (`Cycles c+1..c+20`).
  - `_tune_holt_causal(train_y, val_size=18)` splits `train_y` into `sub_tr = train_y[:-18]` and `sub_val = train_y[-18:]` (all $\le c$), selecting $(\alpha, \beta)$ for Model 3 and $(\alpha, \beta, \phi)$ for Model 4 with zero access to any cycle $> c$.
- **Per-Origin Verification (`results/metrics/rolling_origin_evaluation.csv`):**
  - `Origin 90` (`Train: 1–90`, `Test: 91–110`, $h=20$): Causal Holt $(\alpha=0.10, \beta=0.03)$, $\text{RMSE} = 0.08242\text{ Ah}$ vs. Baseline ARIMA $\text{RMSE} = 0.15364\text{ Ah}$.
  - `Origin 105` (`Train: 1–105`, `Test: 106–125`, $h=20$): Causal Holt $(\alpha=0.30, \beta=0.10)$, $\text{RMSE} = 0.01851\text{ Ah}$ vs. Baseline ARIMA $\text{RMSE} = 0.05412\text{ Ah}$.
  - `Origin 120` (`Train: 1–120`, `Test: 121–140`, $h=20$): Causal Holt $(\alpha=0.30, \beta=0.10)$, $\text{RMSE} = 0.01872\text{ Ah}$ vs. Baseline ARIMA $\text{RMSE} = 0.05097\text{ Ah}$.
  - `Origin 134` (`Train: 1–134`, `Test: 135–154`, $h=20$): Causal Holt $(\alpha=0.10, \beta=0.10)$, $\text{RMSE} = 0.01327\text{ Ah}$ vs. Baseline ARIMA $\text{RMSE} = 0.07092\text{ Ah}$.
  - `Origin 148` (`Train: 1–148`, `Test: 149–168`, $h=20$): Causal Holt $(\alpha=0.30, \beta=0.10)$, $\text{RMSE} = 0.01668\text{ Ah}$ vs. Baseline ARIMA $\text{RMSE} = 0.05802\text{ Ah}$.

---

### Item 3: Residual Diagnostics (`results/metrics/residual_diagnostics.csv`)
- **Residual Generation Verification:**
  - For `Model 1` (`ARIMA(0,1,1)`) and `Model 2` (`ARIMA(0,1,1)` with Drift), `best_model.resid[d:]` (`d=1`) drops the diffuse initialization residual at $t=1$ (`Q_1 - 0 = 2.0520 Ah`), yielding $N = 133$ valid 1-step in-sample innovations (`Cycles 2–134`).
  - For `Model 3`, `Model 3b`, and `Model 4` (`Holt` with `initialization_method="estimated"`), initial states $(L_0, T_0)$ are estimated at $t=0$, so `best_holt.resid` contains $N = 134$ valid 1-step residuals (`Cycles 1–134`).
- **Complete Ljung–Box & Shapiro–Wilk Table:**
  - **Model 1 (`ARIMA(0,1,1)`):** $Q(5)=6.3575$ ($p=0.2730$), $Q(10)=8.0060$ ($p=0.6282$), $Q(15)=9.4351$ ($p=0.8537$), $Q(20)=12.3627$ ($p=0.9030$); Shapiro–Wilk $W=0.6040$ ($p=2.29 \times 10^{-17}$). Decision: **Fail to reject $H_0$ of zero autocorrelation** at $\alpha=0.05$ (non-Gaussian due to regeneration spikes).
  - **Model 2 (`ARIMA(0,1,1)` with Drift):** $Q(5)=6.7096$ ($p=0.2431$), $Q(10)=8.4371$ ($p=0.5862$), $Q(15)=9.8568$ ($p=0.8287$), $Q(20)=12.6608$ ($p=0.8915$); Shapiro–Wilk $W=0.6021$ ($p=2.09 \times 10^{-17}$). Decision: **Fail to reject $H_0$ of zero autocorrelation** at $\alpha=0.05$.
  - **Model 3 (`Holt Linear`, $\alpha=0.10, \beta=0.05$):** $Q(5)=125.4536$ ($p=2.19 \times 10^{-25}$), $Q(10)=127.4681$ ($p=1.53 \times 10^{-22}$), $Q(15)=129.0987$ ($p=3.19 \times 10^{-20}$), $Q(20)=144.0514$ ($p=8.60 \times 10^{-21}$); Shapiro–Wilk $W=0.8382$ ($p=7.87 \times 10^{-11}$). Decision: **Reject $H_0$** (significant positive 1-step residual autocorrelation caused by heavy level smoothing $\alpha=0.10$).
  - **Model 3b (`Holt Linear Train-MLE`, $\alpha=0.88, \beta=0.00$):** $Q(5)=6.8265$ ($p=0.2339$), $Q(10)=8.5751$ ($p=0.5728$), $Q(15)=10.0024$ ($p=0.8196$), $Q(20)=12.7811$ ($p=0.8866$); Shapiro–Wilk $W=0.6014$ ($p=1.73 \times 10^{-17}$). Decision: **Fail to reject $H_0$ of zero autocorrelation** at $\alpha=0.05$.
  - **Model 4 (`Damped Holt`, $\alpha=0.60, \beta=0.05, \phi=0.98$):** $Q(5)=15.9040$ ($p=0.0071$), $Q(10)=19.5888$ ($p=0.0334$), $Q(15)=20.8006$ ($p=0.1433$), $Q(20)=23.3011$ ($p=0.2743$); Shapiro–Wilk $W=0.6448$ ($p=1.54 \times 10^{-16}$). Decision: **Reject $H_0$ at short lags ($m=5, 10$)**; fail to reject at lags $15, 20$.

---

### Item 4: RUL Validation at Origin Cycle 60
- **Causal Data Slicing (`src/models.py`, lines 604–622):**
  - `train_sub = cap[:60]` uses strictly `Cycles 1–60` ($Q_{60} = 1.644680\text{ Ah}$).
  - `_tune_holt_causal(train_sub, val_size=15)` tunes $(\alpha, \beta)$ on `Cycles 1–45` (sub-train) and `Cycles 46–60` (inner validation), selecting $\alpha = 0.10, \beta = 0.10$ without touching any cycle $> 60$.
- **Threshold Crossing Verification (`eol_capacity_ah = 1.40 Ah`):**
  - **Actual EOL Crossing:** In `data/processed/battery_capacity.csv`, Cycle 111 has $Q_{111} = 1.406872\text{ Ah} \ge 1.40\text{ Ah}$, and **Cycle 112** has $Q_{112} = 1.390955\text{ Ah} < 1.40\text{ Ah}$. Thus `actual_eol_cycle = 112` and `actual_rul_cycles = 112 - 60 = 52`.
  - **Model 3 (Causally Tuned Holt Linear):** Crosses $< 1.40\text{ Ah}$ at **Cycle 98** (`holt_predicted_rul_cycles = 38`, error $= -14$ cycles).
  - **Model 2 (`ARIMA(0,1,1)` with Drift):** Crosses $< 1.40\text{ Ah}$ at **Cycle 96** (`arima_drift_predicted_rul_cycles = 36`, error $= -16$ cycles).
  - **Model 1 (`ARIMA(0,1,1)` Baseline):** Plateaus at $1.643\text{ Ah}$ and never crosses $1.40\text{ Ah}$ (`arima_predicted_eol_cycle = None`).

