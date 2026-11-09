# Before the Battery Dies — Forecasting Lithium-Ion Battery Degradation Using Statistical Time-Series Models



**Domain:** Industrial / Energy Systems (Battery Health Monitoring & Prognostics)  

**Primary Dataset:** EV Lithium Ion Battery State of Health Dataset (Mendeley Data V2, DOI: `10.17632/7ctkcfnsdw.2`) / NASA Ames PCoE Cell **B0006**  

**Core Models Evaluated:**

- **Model 1:** Box–Jenkins $\text{ARIMA}(0, 1, 1)$ — Baseline

- **Model 2:** $\text{ARIMA}(0, 1, 1)$ with Linear Drift (`trend='t'`)

- **Model 3:** Holt's Linear Trend (Validation-Tuned $\alpha=0.10, \beta=0.05$ & Training-MLE $\alpha=0.88, \beta=0.00$)

- **Model 4:** Damped Holt's Trend (Validation-Tuned $\alpha=0.60, \beta=0.05, \phi=0.98$)



---



## 1. Project Overview



Lithium-ion batteries lose usable discharge capacity ($\text{Ah}$) over repeated charge–discharge cycles. This repository implements an end-to-end, scientifically audited, reproducible statistical time-series forecasting framework that:

1. Audits all raw spreadsheets (`Discharge AH.xlsx`, `Discharge Summary.xlsx`, `SoH Data.xlsx`) from the Mendeley Data V2 repository (`10.17632/7ctkcfnsdw.2`), verifying that all $50,285$ intra-cycle measurements and $168$ discharge cycles correspond to a single cell (**NASA PCoE Cell B0006** tested at $24^\circ\text{C}$).

2. Documents a comprehensive methodology audit (`reports/methodology_audit.md`) eliminating subtle lookahead leakage in ACF/PACF diagnostics, MLE parameter estimation, rolling-origin cross-validation, and RUL estimation.

3. Preprocesses the sequential cycle-indexed discharge capacity series ($t = 1, \dots, 168$), derives State of Health ($\text{SOH}\\%$) and capacity fade indicators, and exports `data/processed/battery_capacity.csv`. The index represents successive discharge cycles, not uniform calendar-time intervals; forecasts are therefore expressed in cycles.

4. Generates **15 publication-grade visualizations** in `results/figures/` with consistent axis labels (`Discharge Cycle` and `Discharge Capacity (Ah)`), covering Exploratory Data Analysis (`fig01`–`fig08`), Model Diagnostics, Forecasts, Horizon Decay, Rolling-Origin Stability, RUL Estimation (`fig09`–`fig13`), Comprehensive 5-Model Residual Diagnostics (`fig14`), and a 6-Panel Actual-vs-Predicted Comparison on Identical Scales (`fig15`).

5. Evaluates all models under a leakage-free 80%/20% chronological split (Training: Cycles 1–134; Inner Validation: Cycles 111–134; Held-Out Test: Cycles 135–168, $H = 34$ cycles), reporting `MAE`, `RMSE`, `MAPE`, and `sMAPE` separately across Training, Validation, Held-Out Multi-Step Test, and Online 1-Step Recalibration sets.



---



## 2. Key Empirical Findings



### 2.1 Held-Out Multi-Step Test Performance (`Cycles 135–168`, $H = 34$)



| Model | Train RMSE (Ah) [1-Step, C1–134] | Val RMSE (Ah) [24-Step, C111–134] | Test MAE (Ah) [34-Step, C135–168] | Test RMSE (Ah) [34-Step, C135–168] | Test MAPE (%) | Test sMAPE (%) |

| :--- | ---: | ---: | ---: | ---: | ---: | ---: |

| **Model 1: $\text{ARIMA}(0, 1, 1)$ — Baseline** | 0.02560 | 0.04286 | 0.10094 | 0.11153 | 8.203% | 7.805% |

| **Model 2: $\text{ARIMA}(0, 1, 1)$ with Drift** | 0.02496 | 0.04689 | **0.01298** | **0.01522** | **1.030%** | **1.028%** |

| **Model 3: Holt's Linear Trend (Val-Tuned: $\alpha=0.10, \beta=0.05$)** | 0.03533 | 0.01732 | **0.02253** | **0.02739** | **1.841%** | **1.817%** |

| **Model 3b: Holt's Linear Trend (Train-MLE: $\alpha=0.88, \beta=0.00$)** | 0.02486 | 0.04683 | **0.01288** | **0.01514** | **1.022%** | **1.020%** |

| **Model 4: Damped Holt's Trend (Val-Tuned: $\alpha=0.60, \beta=0.05, \phi=0.98$)** | 0.02594 | 0.01375 | 0.06616 | 0.07513 | 5.390% | 5.205% |

| Candidate $\text{ARIMA}(1, 1, 1)$ (Multi-Step) | 0.02553 | 0.04422 | 0.09890 | 0.10962 | 8.040% | 7.655% |

| Candidate $\text{ARIMA}(2, 1, 1)$ (Val-Optimal, Multi-Step) | 0.02560 | 0.02791 | 0.10096 | 0.11151 | 8.205% | 7.807% |



### 2.2 Leakage-Free Rolling-Origin Summary (`5 Origins: Cycles 90, 105, 120, 134, 148`, $h = 20$)



| Model | Mean MAE $\pm$ SD (Ah) | Mean RMSE $\pm$ SD (Ah) | Mean MAPE $\pm$ SD (%) | Mean sMAPE $\pm$ SD (%) |

| :--- | ---: | ---: | ---: | ---: |

| **Model 1: $\text{ARIMA}(0, 1, 1)$ Baseline** | $0.07231 \pm 0.04333$ | $0.07753 \pm 0.04322$ | $5.341\\% \pm 2.833\\%$ | $5.152\\% \pm 2.636\\%$ |

| **Model 2: $\text{ARIMA}(0, 1, 1)$ with Drift** | $0.02920 \pm 0.02955$ | $0.03337 \pm 0.02826$ | $2.097\\% \pm 1.985\\%$ | $2.072\\% \pm 1.909\\%$ |

| **Model 3: Holt's Linear Trend (Causally Tuned)** | **$0.02699 \pm 0.03028$** | **$0.02992 \pm 0.02943$** | **$1.924\\% \pm 2.032\\%$** | **$1.959\\% \pm 2.104\\%$** |

| **Model 4: Damped Holt's Trend (Causally Tuned)** | $0.04538 \pm 0.04800$ | $0.05030 \pm 0.05162$ | $3.275\\% \pm 3.261\\%$ | $3.167\\% \pm 3.067\\%$ |



---



## 3. Repository Structure



```text

before-the-battery-dies/

│

├── app/

│   ├── __init__.py

│   ├── streamlit_app.py

│   ├── pages/

│   │   ├── __init__.py

│   │   ├── overview.py

│   │   ├── eda.py

│   │   ├── stationarity.py

│   │   ├── forecasting.py

│   │   ├── evaluation.py

│   │   └── rul.py

│   ├── components/

│   │   ├── __init__.py

│   │   ├── charts.py

│   │   ├── metrics.py

│   │   └── sidebar.py

│   └── utils/

│       ├── __init__.py

│       └── data_loader.py

│

├── .streamlit/

│   └── config.toml

│

├── data/

│   ├── raw/

│   │   ├── Discharge AH.xlsx

│   │   ├── Discharge Summary.xlsx

│   │   └── SoH Data.xlsx

│   └── processed/

│       └── battery_capacity.csv

│

├── notebooks/

│   └── battery_degradation_analysis.ipynb

│

├── src/

│   ├── __init__.py

│   ├── data_loader.py

│   ├── preprocessing.py

│   ├── eda.py

│   ├── stationarity.py

│   ├── models.py

│   ├── evaluation.py

│   ├── visualization.py

│   └── pipeline.py

│

├── results/

│   ├── figures/

│   │   ├── fig01_capacity_vs_cycle.png ... fig13_rolling_origin_and_rul.png

│   │   ├── fig14_comprehensive_residual_diagnostics.png

│   │   └── fig15_actual_vs_predicted_six_panel.png

│   ├── metrics/

│   │   ├── model_comparison_metrics.csv

│   │   ├── train_val_test_metrics.csv

│   │   ├── online_recalibration_metrics.csv

│   │   ├── horizon_comparison.csv

│   │   ├── horizon_sensitivity_metrics.csv

│   │   ├── arima_order_selection.csv

│   │   ├── holt_parameter_tuning.csv

│   │   ├── rolling_origin_evaluation.csv

│   │   ├── rolling_origin_metrics.csv

│   │   ├── residual_diagnostics.csv

│   │   ├── stationarity_results.json

│   │   └── experiment_summary.json

│   └── forecasts/

│       └── test_forecasts_comparison.csv

│

├── research/

│   └── literature_review.md

│

├── reports/

│   ├── data_inspection.md

│   ├── methodology_audit.md

│   ├── project_report.md

│   └── technical_blog.md

│

├── tests/

│   └── test_preprocessing.py

│

├── requirements.txt

├── README.md

└── .gitignore

```



---



## 4. Reproducible Setup and Execution Instructions



### 4.1 Install Dependencies

Requires Python 3.11 or 3.12:

```powershell

pip install -r requirements.txt

```



### 4.2 Run the Complete End-to-End Pipeline

From the `before-the-battery-dies/` root directory, execute:

```powershell

$env:PYTHONPATH = (Get-Location).Path

python -m src.pipeline

```

This single command audits the raw Excel files in `data/raw/`, generates `data/processed/battery_capacity.csv`, runs stationarity tests, trains and evaluates all ARIMA and Holt models, saves all 15 figures to `results/figures/`, and exports all CSV/JSON tables to `results/metrics/` and `results/forecasts/`.



### 4.3 Run Automated Unit Tests

```powershell

$env:PYTHONPATH = (Get-Location).Path

pytest tests/ -v --rootdir=.

```



### 4.4 Launch the Interactive Streamlit Dashboard

From the `before-the-battery-dies/` root directory, execute:

```powershell

python -m streamlit run app/streamlit_app.py

```



### 4.5 Launch the Interactive Jupyter Notebook

```powershell

jupyter notebook notebooks/battery_degradation_analysis.ipynb

```



---



## 5. Interactive Streamlit Analytics Dashboard (`app/`)



The repository includes an industrial-grade, dark-themed battery analytics dashboard built with **Streamlit** and **Plotly** (`app/streamlit_app.py`). It wraps the validated statistical forecasting engine (`src/`) without modifying any core estimation or cross-validation procedures and strictly preserves causal, leakage-free evaluation.



### Dashboard Architecture & Pages



1. **Page 1 — Overview (`app/pages/overview.py`):**

   - Executive KPI summary cards for NASA Ames PCoE Cell **B0006** (`168` discharge cycles, `2.0523 Ah` initial capacity, `1.1722 Ah` minimum / `1.2038 Ah` final cycle-168 capacity, `57.12%` minimum / `58.66%` final SOH, `42.88%` peak capacity fade).

   - Interactive Plotly capacity degradation trajectory ($Q_t$ in $\text{Ah}$) and State of Health ($\text{SOH}_t$ in $\\%$) charts with 80%/20% chronological split markers (`Cycle 134`) and NASA EOL threshold (`1.40 Ah` / `68.22% SOH_init`).

   - Primary and secondary research questions, dataset audit summary, and interactive CSV table inspection.



2. **Page 2 — Exploratory Data Analysis (`app/pages/eda.py`):**

   - Four interactive tabs:

     - *Capacity, SOH & Fade Dynamics:** Cycle-to-cycle capacity difference $\Delta Q_t$ highlighting rest-induced capacity regeneration spikes (`> +0.025 Ah`) versus regular electrochemical fade.

     - *Rolling Mean & Volatility:** 10-cycle rolling mean and rolling standard deviation ($\text{Ah}$).

     - *Capacity & $\Delta Q_t$ Distributions:** Regime-stratified histograms and box plots (`Cycles 1–60`, `61–120`, `121–168`).

     - *ACF & PACF Diagnostics & Publication Gallery:** Strictly causal training-set (`Cycles 1–134`) Autocorrelation and Partial Autocorrelation functions up to lag 25, plus an expandable gallery of publication figures (`fig01`–`fig08`).



3. **Page 3 — Stationarity Analysis (`app/pages/stationarity.py`):**

   - Side-by-side comparison of the non-stationary original capacity series ($Q_t, d=0$) and the stationary first-order differenced series ($\Delta Q_t = Q_t - Q_{t-1}, d=1$).

   - Exact Augmented Dickey–Fuller (ADF) test statistics, $p$-values, and $1\\% / 5\\% / 10\\%$ critical values across full-series and causal training-only (`Cycles 1–134`) splits.

   - Mathematical explanation of why $d=1$ differencing is required for Box–Jenkins ARIMA and why zero-intercept $\text{ARIMA}(p,1,q)$ flattens out over multi-step horizons unless a drift term ($c$) or explicit local slope ($b_t$) is included.



4. **Page 4 — Forecasting Lab (`app/pages/forecasting.py`):**

   - Interactive model selector supporting `ARIMA (0,1,1) — Baseline`, `ARIMA (0,1,1) with Drift`, `Holt Linear Trend (Val-Tuned)`, `Holt Linear Trend (Train-MLE)`, and `Damped Holt`.

   - Configurable **Forecast Origin Cycle** (`Cycles 50–158`, default `Cycle 134`) and **Forecast Horizon** ($h = 5 \dots 60$ cycles, default `34 cycles`), fitting and tuning all models strictly causally on `Cycles 1..Origin`.

   - Interactive overlay of actual capacity, causal forecast trajectories, and 95% prediction intervals, downloadable cycle-by-cycle forecast table (`CSV`), fitted parameter readout, and mathematical formulation cards.



5. **Page 5 — Model Evaluation (`app/pages/evaluation.py`):**

   - Five analytical tabs:

     - *Held-Out Test Comparison:** Full `MAE`, `RMSE`, `MAPE`, and `sMAPE` leaderboard across Training, Inner Validation, and Held-Out Multi-Step Test (`Cycles 135–168`), plus multi-metric bar charts.

     - *Multi-Horizon Decay (`5, 10, 20, 34` cycles):** Error growth curves and tables across forecast horizons (`h = 1` to `34`).

     - *Rolling-Origin Evaluation (`5 Origins`):** Leakage-free evaluation across origins `90, 105, 120, 134, 148` ($h=20$), In this five-origin experiment, causally tuned Holt Linear Trend has the lowest mean rolling-origin error (`RMSE = 0.0299 Ah`, `MAPE = 1.92%`). This is distinct from the single inner-validation comparison, where Damped Holt has the lowest validation RMSE (`0.01375 Ah`); model rankings depend on the evaluation design.

     - *Online 1-Step Recalibration:** Comparison of multi-step open-loop forecasting against causal 1-step-ahead BMS filtering (`RMSE \~ 0.0149–0.0156 Ah`).

     - *Residual Diagnostics (Ljung–Box):** Exact Ljung–Box $Q$-statistics and $p$-values at lags `5, 10, 15, 20` with statistically accurate white-noise hypothesis interpretation, residual scatter/ACF plots, and publication figures (`fig09`–`fig15`).



6. **Page 6 — Remaining Useful Life (`app/pages/rul.py`):**

   - Interactive **Forecast Origin Cycle** selector (default `Cycle 60`, before the actual `Cycle 112` threshold crossing) and **EOL Capacity Threshold** slider (default `1.40 Ah`, NASA PCoE B0006 stopping criterion).

   - Computes real-time causal RUL predictions for all models using only cycles up to the selected origin, displaying predicted EOL crossing cycles, predicted RUL, actual observed RUL, and error in cycles.

   - Includes a dedicated engineering interpretation and experimental disclaimer regarding statistical trend extrapolation versus physics-of-failure electrochemistry. RUL estimates are conditional statistical extrapolations to the selected capacity threshold, not guaranteed service-life predictions.



---



## 6. Data Citation & License



- **Mendeley Dataset (CC BY 4.0):** Veeramsetty, Venkataramana; Keerthi Sri, G.V. (2026), *"EV Lithium Ion Battery State of Health Dataset"**, Mendeley Data, V2, DOI: `10.17632/7ctkcfnsdw.2`.

- **NASA Original Source:** Saha, B., & Goebel, K. (2007), *"Battery Data Set"**, NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
