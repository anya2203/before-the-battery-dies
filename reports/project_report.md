# BEFORE THE BATTERY DIES: Forecasting Lithium-Ion Battery Degradation Using Statistical Time-Series Models

**Course:** Undergraduate Time Series Analysis (TSA) Lab — Alternative Assessment Tool (AAT)  
**Domain:** Industrial & Energy Systems (Battery Health Monitoring & Prognostics)  
**Primary Dataset:** EV Lithium Ion Battery State of Health Dataset (Mendeley Data V2, DOI: `10.17632/7ctkcfnsdw.2`) / NASA Ames PCoE Cell **B0006**  

---

## 1. Abstract

Lithium-ion batteries undergo irreversible electrochemical ageing across repeated charge–discharge cycles, manifesting as progressive discharge capacity fade. While traditional Battery Management Systems (BMS) monitor present State of Health ($\text{SOH}$), proactive maintenance and range-reliability planning require forecasting future capacity over upcoming operating cycles. This study investigates whether trend-aware statistical forecasting models (**$\text{ARIMA}(0,1,1)$ with Linear Drift**, **Holt's Linear Trend**, and **Damped Holt's Trend**) predict future cycle-indexed discharge capacity more accurately than a conventional Box–Jenkins **$\text{ARIMA}(0,1,1)$** baseline. Using the 168-cycle discharge sequence of NASA PCoE Cell B0006 curated in the Mendeley Data V2 repository (`10.17632/7ctkcfnsdw.2`), we perform a rigorous schema audit, cycle-index alignment, exploratory data analysis, and Augmented Dickey–Fuller (ADF) stationarity testing ($\text{ADF} = -1.3520, p = 0.6051$ in levels; $\text{ADF} = -8.5062, p = 1.20 \times 10^{-13}$ after first differencing, establishing $d = 1$). Under a strict chronological 80%/20% split (Cycles 1–134 training, Cycles 111–134 inner validation, Cycles 135–168 held-out testing, horizon $H = 34$ cycles), standard multi-step $\text{ARIMA}(0,1,1)$ ($\text{AIC} = -593.50$) converges to a horizontal level prediction and incurs $\text{MAE} = 0.10094\text{ Ah}$, $\text{RMSE} = 0.11153\text{ Ah}$, $\text{MAPE} = 8.203\%$, and $\text{sMAPE} = 7.805\%$. In contrast, validation-tuned **Holt's Linear Trend** ($\alpha = 0.10, \beta = 0.05$) explicitly tracks both local capacity level ($L_t$) and per-cycle degradation slope ($T_t$), reducing 34-cycle test error by **75.4% in RMSE** ($\text{MAE} = 0.02253\text{ Ah}$, $\text{RMSE} = 0.02739\text{ Ah}$, $\text{MAPE} = 1.841\%$, $\text{sMAPE} = 1.817\%$), while **$\text{ARIMA}(0,1,1)$ with Linear Drift** (`trend='t'`) and training-only MLE Holt Linear ($\alpha = 0.88, \beta \approx 0.00$) achieve $\text{MAPE} = 1.030\%$ ($\text{RMSE} = 0.01522\text{ Ah}$) and $\text{MAPE} = 1.022\%$ ($\text{RMSE} = 0.01514\text{ Ah}$), respectively. Leakage-free rolling-origin cross-validation across five chronological cutoffs, multi-horizon decay analysis ($h \in \{5, 10, 20, 34\}$), Ljung–Box residual diagnostics, and Remaining Useful Life ($\text{RUL}$) threshold-crossing experiments confirm that explicit trend modeling is essential for multi-step battery degradation forecasting.

---

## 2. Introduction

Lithium-ion (Li-ion) batteries form the energy-storage backbone of modern electric vehicles (EVs), grid-scale renewable storage installations, consumer electronics, and aerospace power systems. Although fresh Li-ion cells deliver a rated discharge capacity under nominal operating conditions (e.g., $2.0\text{ Ah}$ for standard 18650 cylindrical cells), repeated cycling induces solid-electrolyte interphase (SEI) film growth, lithium plating, loss of lithium inventory (LLI), and loss of active electrode material (LAM). Consequently, the maximum deliverable charge during a discharge cycle decreases over the battery's operational lifetime—a phenomenon known as **capacity fade** or **capacity degradation**.

Unanticipated capacity fade poses severe operational risks:
- **Electric Vehicles:** Progressive loss of usable driving range and degraded peak power delivery.
- **Grid & Industrial Storage:** Reduced dispatchable energy capacity and thermal runaway hazards in severely aged cells.
- **Lifecycle Management:** Difficulty scheduling timely cell replacement, warranty provisioning, and second-life repurposing.

While onboard sensors measure terminal voltage ($V$), current ($I$), and temperature ($T$) to estimate the battery's *present* condition, statistical time-series forecasting enables a forward-looking capability: predicting future discharge capacity and State of Health ($\text{SOH}$) dozens of cycles before the cell reaches End of Life ($\text{EOL}$).

---

## 3. Problem Statement and Research Questions

### 3.1 Methodological Formulation (Cycle-Indexed Time Series)
Unlike meteorological or financial series indexed by wall-clock timestamps, electrochemical battery degradation is fundamentally driven by **charge–discharge cycling**. Let
$$Y = \{Q_1, Q_2, Q_3, \dots, Q_N\}$$
denote the sequential time series of measured discharge capacities (in Ampere-hours, $\text{Ah}$), where the index $t \in \{1, 2, \dots, N\}$ represents the **successive discharge cycle number**. Given historical observations $\{Q_1, Q_2, \dots, Q_n\}$ up to cycle $n$, our forecasting task is to predict future discharge capacity over a horizon of $h$ subsequent discharge cycles:
$$\hat{Q}_{n+h} = f(Q_1, Q_2, \dots, Q_n), \quad h \in \{1, 2, \dots, H\}.$$

### 3.2 Central Research Question
> **Can trend-aware statistical forecasting models improve the prediction of lithium-ion battery capacity degradation compared with conventional ARIMA forecasting?**

### 3.3 Supporting Research Questions
1. **Stationarity & Integration Order:** Does the discharge-capacity series $Q_t$ exhibit stochastic non-stationarity, and is first-order differencing ($\Delta Q_t = Q_t - Q_{t-1}$, $d=1$) both necessary and sufficient to achieve stationarity without over-differencing?
2. **Role of Explicit Drift / Slope States:** Why does standard Box–Jenkins $\text{ARIMA}(0,1,1)$ plateau over multi-step horizons, and to what extent does incorporating a deterministic drift term (`trend='t'` in Model 2) or an adaptive exponential smoothing slope ($T_t$ in Model 3: Holt's Linear Trend and Model 4: Damped Holt) reduce multi-step forecast error?
3. **Horizon Error Scaling:** How do $\text{MAE}$, $\text{RMSE}$, $\text{MAPE}$, and $\text{sMAPE}$ evolve as the forecast horizon expands from short-term ($h = 5, 10$ cycles) to medium- and long-term ($h = 20, 34$ cycles)?
4. **Rolling-Origin Stability & Causal Tuning:** Do trend-aware models maintain their accuracy advantage across multiple chronological forecast origins (`Cycles 90, 105, 120, 134, 148`) when hyperparameters are tuned strictly causally using only past cycles?
5. **Residual Diagnostics & Smoothing Trade-Offs:** Do the training residuals of $\text{ARIMA}(0,1,1)$, $\text{ARIMA}(0,1,1)$ with Drift, and Holt variants satisfy the Ljung–Box white-noise condition, and how do discrete rest-induced capacity regeneration spikes affect residual normality and smoothing parameter selection?

---

## 4. Objectives

1. **Data Audit & Provenance Verification:** Inspect all raw spreadsheet schemas in the Mendeley V2 EV Battery SOH dataset (`10.17632/7ctkcfnsdw.2`), reconcile cycle indices against the underlying NASA PCoE experiment, and prevent multi-battery concatenation.
2. **Preprocessing & Health Indicator Derivation:** Construct a clean, strictly ordered single-battery cycle series ($t = 1, \dots, 168$) and compute derived State of Health ($\text{SOH}_t$) and capacity fade indicators.
3. **Exploratory Time-Series Analysis (EDA):** Characterize the degradation trend, non-stationarity, training-only autocorrelation structure, distributional properties, ambient temperature invariance, and rest-induced capacity regeneration spikes across 8 diagnostic visualizations.
4. **Formal Stationarity Testing:** Apply the Augmented Dickey–Fuller (ADF) unit-root test, rolling window statistics, and ACF/PACF analysis to justify the order of integration $d = 1$.
5. **Fair Multi-Model Comparison:** Compare four core multi-step models—**Model 1: $\text{ARIMA}(0,1,1)$ Baseline**, **Model 2: $\text{ARIMA}(0,1,1)$ with Drift**, **Model 3: Holt's Linear Trend**, and **Model 4: Damped Holt's Trend**—under a strict training/validation/test protocol with zero test leakage (`reports/methodology_audit.md`).
6. **Strengthened Forecast Evaluation:** Report $\text{MAE}$, $\text{RMSE}$, $\text{MAPE}$, and $\text{sMAPE}$ separately for training, validation, held-out multi-step testing, and online 1-step recalibration, supplemented by horizon analysis ($h = 5, 10, 20, 34$), leakage-free rolling-origin cross-validation (mean $\pm$ SD across 5 origins), Ljung–Box residual diagnostics, and causal $\text{RUL}$ estimation.

---

## 5. Domain Justification

Industrial energy storage and electric mobility depend on lightweight, interpretable prognostic algorithms that can execute reliably on embedded Battery Management System (BMS) microcontrollers. While deep neural networks (e.g., LSTMs and Transformers) require large multi-cell training corpora and heavy compute resources, statistical time-series models such as $\text{ARIMA}$ and Holt's Exponential Smoothing operate directly on a single cell's historical capacity trajectory with closed-form state updates ($L_t, T_t$) and transparent prediction intervals. Evaluating their empirical strengths and failure modes on real experimental ageing data directly informs practical BMS algorithm design.

---

## 6. Literature Review

We reviewed **seven peer-reviewed research papers** spanning experimental battery ageing, capacity fade modeling, $\text{SOH}$ estimation, $\text{ARIMA}$ forecasting, exponential smoothing, and $\text{RUL}$ prediction. Full details are maintained in `research/literature_review.md` and summarized in Table 1.

| # | Authors & Year | Title & Venue | DOI | Dataset | Method | Key Findings | Limitation / Gap |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Zhou & Huang (2016) | *Lithium-ion batteries remaining useful life prediction based on a mixture of empirical mode decomposition and ARIMA model*, Microelectronics Reliability (Vol. 65, pp. 265–273) | `10.1016/j.microrel.2016.07.151` | NASA PCoE Li-ion Dataset | EMD + $\text{ARIMA}$ | Standalone $\text{ARIMA}$ captures local autocorrelation; EMD+ARIMA reduces RUL error below 5%. | Un-drifted $\text{ARIMA}(p,1,q)$ flattens over multi-step horizons; EMD suffers boundary distortion. |
| **2** | Pan, Chen, Wang, & He (2019) | *Lithium-ion Battery Remaining Useful Life Prediction Based on Exponential Smoothing and Particle Filter*, Int. J. Electrochem. Sci. (Vol. 14, pp. 9537–9551) | `10.20964/2019.10.15` | NASA PCoE Li-ion Dataset | Exponential Smoothing + Particle Filter | Exponential smoothing adapts rapidly to local capacity fade slope changes across mid-to-late cycle life. | High Monte Carlo complexity; standalone double exponential smoothing (Holt) not isolated. |
| **3** | Chen, Xu, & Zhou (2018) | *Novel Approach for Lithium-Ion Battery On-Line Remaining Useful Life Prediction Based on Permutation Entropy*, Energies (Vol. 11, No. 4, 820) | `10.3390/en11040820` | NASA PCoE (B0005, **B0006**, B0007) | VMD + PE + $\text{ARIMA}$ + $\text{GM}(1,1)$ | Decoupling oscillatory regeneration transients ($\text{ARIMA}$) from monotonic trend ($\text{GM}$) improves B0006 RUL accuracy. | Multi-stage decomposition requires mode-count tuning sensitive to window boundaries. |
| **4** | Zhang, Xiong, He, & Pecht (2018) | *Long Short-Term Memory Recurrent Neural Network for Remaining Useful Life Prediction of Lithium-Ion Batteries*, IEEE Trans. Veh. Technol. (Vol. 67, No. 7, pp. 5695–5705) | `10.1109/TVT.2018.2805189` | NASA PCoE & CALCE Datasets | LSTM RNN | Captures nonlinear fade trajectories when trained on 50–70% of historical cycle life. | Black-box parameterization; high compute and data demands for single-cell BMS deployment. |
| **5** | Fauzi, Yudistira, & Mahmudy (2024) | *State-of-Health Prediction of Lithium-Ion Batteries Using Exponential Smoothing Transformer With Seasonal and Growth Embedding*, IEEE Access (Vol. 12, pp. 14659–14670) | `10.1109/ACCESS.2024.3357736` | NASA PCoE (B0005, **B0006**, B0007, B0018) | Exponential Smoothing Transformer | Embedding Holt-inspired level and growth smoothing equations into attention layers stabilizes SOH forecasts. | Does not isolate standalone classical Holt's Linear Trend against $\text{ARIMA}$ across horizons. |
| **6** | Luh & Blank (2024) | *Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell*, Nature Scientific Data (Vol. 11, 1004) | `10.1038/s41597-024-03831-x` | KIT NMC/C-SiO Aging Dataset (`10.35097/1947`) | Coulomb Counting, RPT & EIS | Characterizes physical LLI/LAM fade slopes and relaxation-induced capacity recovery after rest periods. | Experimental dataset paper; does not benchmark statistical forecasting models. |
| **7** | Liao, Lyu, Zou, & Zeng (2026) | *SDMnet: A seasonal-trend decomposition guided data augmentation and prediction framework for lithium-ion battery capacity forecasting*, J. Energy Storage | `10.1016/j.est.2026.121800` | NASA / CALCE Aging Datasets | Seasonal-Trend Decomposition + Augmentation | Explicitly separating degradation trend from local fluctuations substantially improves multi-step capacity forecasting. | Relies on synthetic data augmentation and deep networks rather than interpretable statistical models. |

---

## 7. Research Gap

While recent literature increasingly embeds trend decomposition inside complex neural architectures (Fauzi et al., 2024; Liao et al., 2026) or multi-stage signal processing pipelines (Zhou & Huang, 2016; Li et al., 2018), a fundamental statistical question remains under-documented: **how much multi-step forecasting accuracy on real battery degradation data can be achieved purely by transitioning from a conventional difference-stationary Box–Jenkins $\text{ARIMA}(0,1,1)$ baseline to explicitly trend-aware statistical models ($\text{ARIMA}(0,1,1)$ with Drift, Holt's Linear Trend, and Damped Holt)?** Specifically, standard $\text{ARIMA}(p,1,q)$ without drift models only zero-mean stationary increments $\Delta Q_t$ and lacks a persistent slope state, whereas drifted $\text{ARIMA}$ and Holt's Linear Trend explicitly propagate the per-cycle fade rate. Furthermore, existing studies rarely contrast validation-tuned smoothing parameters against maximum-likelihood 1-step estimators under a strict causal rolling-origin protocol. Our study fills this gap on NASA Cell B0006.

---

## 8. Dataset Description and Source

- **Primary Repository:** *EV Lithium Ion Battery State of Health Dataset*, Mendeley Data, Version 2 (Published August 7, 2026), authored by Venkataramana Veeramsetty and Keerthi Sri G.V., DOI: `10.17632/7ctkcfnsdw.2` (CC BY 4.0 license).
- **Underlying Experimental Origin:** NASA Ames Prognostics Center of Excellence (PCoE) Lithium-Ion Battery Aging Dataset, Cell **B0006** (18650 LiCoO$_2$/graphite cell, nominal rated capacity $2.0\text{ Ah}$).
- **Experimental Protocol:** Cycled in an isothermal chamber at a constant ambient temperature of **$24^\circ\text{C}$**. Charging was conducted in constant-current/constant-voltage (CC-CV) mode at $1.5\text{ A}$ up to $4.2\text{ V}$ until current dropped to $20\text{ mA}$. Discharging was performed at a constant current of $2.0\text{ A}$ until terminal voltage fell to $2.5\text{ V}$.
- **Files Inspected in `data/raw/`:**
  1. `Discharge AH.xlsx` (`50,285 rows × 9 columns`, $4.59\text{ MB}$): Intra-cycle discharge time-step telemetry (`Cycle_Number`, `Cycle_Type`, `Ambient_Temperature`, `Time`, `Current_measured`, `Voltage_measured`, `Current Mod`, `delta t`, `Discharge Ah`).
  2. `Discharge Summary.xlsx` (`168 rows × 2 columns`, $12.4\text{ KB}$): Cycle-level summary mapping experimental operation index `Dis-ch Cycle` ($2, 4, \dots, 614$) to integrated discharge capacity `Dis-chg Ah`.
  3. `SoH Data.xlsx` (`168 rows × 4 columns`, $15.6\text{ KB}$): Maps experimental operation index `Charge Cycle` ($2, 4, \dots, 614$) and sequential discharge index `Cycle #` ($1, 2, \dots, 168$) to `Dis-chg Ah` and `SoH` (%).

---

## 9. Data Preprocessing

Following our Phase 1 data and methodology audits (`reports/data_inspection.md` and `reports/methodology_audit.md`), preprocessing (`src/preprocessing.py`) enforces five strict data-integrity rules:
1. **Single-Battery Identity Verification (`NASA_B0006`):** All 168 discharge cycles belong to a single physical cell tested at $24^\circ\text{C}$, starting at $Q_1 = 2.051974\text{ Ah}$ and degrading to a minimum of $Q_{164} = 1.172108\text{ Ah}$ ($Q_{168} = 1.203769\text{ Ah}$). No multi-battery concatenation is performed.
2. **Cycle Index Disambiguation:** In the NASA PCoE test bed, charge, discharge, and Electrochemical Impedance Spectroscopy (EIS) operations share a global operation counter (`2` through `614`). We preserve this counter as `operation_cycle_id` (`2..614`) while indexing our time series strictly by the **sequential discharge cycle number** `cycle_number` $t \in \{1, 2, \dots, 168\}$.
3. **Handling Boundary Artifacts in Raw Telemetry:** In `Discharge AH.xlsx`, row-to-row differencing across cycle boundaries produced $167$ negative `delta t` values (e.g., $-3690.234\text{ s}$) and $1$ trailing missing value at index `7827`. Rather than using corrupted boundary differences, we use the verified cycle-level capacities in `SoH Data.xlsx` (`0` missing values, `0` duplicates) and aggregate valid intra-cycle telemetry (`discharge_duration_s`, `mean_voltage_v`, `min_voltage_v`, `ambient_temperature_c`) grouped cleanly within each `Cycle_Number`.
4. **Derived Health Indicators (Zero Test Leakage):**
   - **Primary State of Health ($\text{SOH}_t$):** Normalized strictly by Cycle 1 capacity ($Q_1 = 2.051974\text{ Ah}$), which lies at the start of the training set:
     $$\text{SOH}_t (\%) = \frac{Q_t}{Q_{\text{initial}}} \times 100, \quad Q_{\text{initial}} = Q_1 = 2.051974\text{ Ah}.$$
   - **Nominal Rated $\text{SOH}$ ($\text{SOH}_{\text{rated},t}$):** Computed relative to NASA's $2.0000\text{ Ah}$ nominal rating ($\text{SOH}_{\text{rated},1} = 102.60\%$, crossing $70\%$ rated capacity [$1.4000\text{ Ah}$] at Cycle 112).
   - **Cumulative Capacity Fade:** $\text{Fade}_t (\text{Ah}) = Q_{\text{initial}} - Q_t$ and $\text{Fade}_t (\%) = 100 - \text{SOH}_t$.
5. **Reproducible Export:** Saved to `data/processed/battery_capacity.csv` (`168 rows × 14 columns`).

---

## 10. Exploratory Data Analysis (EDA)

Eight diagnostic figures (`results/figures/fig01` through `fig08`) were generated by `src/eda.py` using consistent axis labels (`Discharge Cycle` and `Discharge Capacity (Ah)`):

1. **Figure 1 (`fig01_capacity_vs_cycle.png`) — Discharge Capacity vs. Discharge Cycle:**
   - Capacity declines from $2.0520\text{ Ah}$ (Cycle 1) to $1.1721\text{ Ah}$ (Cycle 164), ending at $1.2038\text{ Ah}$ (Cycle 168), with mean $1.5595\text{ Ah}$ ($\text{SD} = 0.2506\text{ Ah}$).
   - **Interpretation:** The decline is strongly trended and non-stationary, but punctuated by upward **capacity regeneration spikes** (self-recharge recovery following rest/EIS intervals), most notably at **Cycle 90** ($+0.1518\text{ Ah}$), **Cycle 20** ($+0.1220\text{ Ah}$), **Cycle 48** ($+0.1190\text{ Ah}$), and **Cycle 151** ($+0.0380\text{ Ah}$). Across the 167 cycle transitions, $31$ exhibit positive capacity jumps while $136$ exhibit capacity loss.
2. **Figure 2 (`fig02_soh_vs_cycle.png`) — State of Health ($\text{SOH}\%$) vs. Discharge Cycle:**
   - $\text{SOH}_t$ drops from $100.00\%$ (Cycle 1) to $57.12\%$ (Cycle 164), finishing at $58.66\%$ (Cycle 168).
   - **Interpretation:** The cell crosses the **80% Initial SOH benchmark ($1.6416\text{ Ah}$)** at **Cycle 61**, the **70% Initial SOH threshold ($1.4364\text{ Ah}$)** at **Cycle 103**, and **NASA's 30% rated-fade stopping threshold ($1.4000\text{ Ah}$)** at **Cycle 112**.
3. **Figure 3 (`fig03_capacity_fade_vs_cycle.png`) — Capacity Fade & Cycle-to-Cycle Differences $\Delta Q_t$:**
   - Cumulative fade reaches $0.8799\text{ Ah}$ ($42.88\%$ of initial capacity) at Cycle 164 ($0.8482\text{ Ah}$ or $41.34\%$ at Cycle 168).
   - **Interpretation:** The lower panel ($\Delta Q_t = Q_t - Q_{t-1}$) reveals a steady negative baseline drift (mean $\Delta Q = -0.005079\text{ Ah/cycle}$, median $-0.0067\text{ Ah/cycle}$) interrupted by discrete positive regeneration impulses. There is **no repeating periodic or seasonal cycle**—regeneration events occur at irregular operational rest intervals.
4. **Figure 4 (`fig04_rolling_statistics.png`) — Rolling Mean and Standard Deviation:**
   - The 10-cycle and 20-cycle rolling means fall monotonically from $\sim 1.98\text{ Ah}$ to $\sim 1.20\text{ Ah}$, while the rolling standard deviation fluctuates between $0.015\text{ Ah}$ (smooth late-life fade) and $0.055\text{ Ah}$ (around regeneration jumps at Cycles 20, 48, and 90).
   - **Interpretation:** Visual confirmation of **mean non-stationarity** and localized variance bursts during rest-recovery events.
5. **Figure 5 (`fig05_acf_plot.png`) & Figure 6 (`fig06_pacf_plot.png`) — Training-Set ACF and PACF Diagnostics (Cycles 1–134):**
   - Computed strictly on the training set (Cycles 1–134) to prevent test-set leakage into Box–Jenkins order identification. On the level series $Q_t$, the ACF decays very slowly and nearly linearly ($r_1 = 0.968$), while the PACF shows a single dominant spike near $+0.968$ at lag 1 and cuts off into the 95% confidence band thereafter.
   - **Interpretation:** This is the textbook signature of a non-stationary random walk with drift / integrated $I(1)$ process. After first differencing ($\Delta Q_t$ on Cycles 2–134), the ACF and PACF remain near zero across lags 1–35, confirming that first differencing removes the stochastic trend.
6. **Figure 7 (`fig07_capacity_distribution.png`) — Distributional Analysis:**
   - The level distribution $Q_t$ is broad and bimodal/right-skewed (mean $1.559\text{ Ah}$, median $1.499\text{ Ah}$) because the battery spends more cycles in the lower-slope mid-to-late degradation regime ($1.2\text{–}1.5\text{ Ah}$) than in the steep early regime ($1.8\text{–}2.05\text{ Ah}$). Conversely, $\Delta Q_t$ is tightly concentrated around $-0.0051\text{ Ah/cycle}$ with a positive right tail caused by regeneration spikes.
7. **Figure 8 (`fig08_temperature_and_telemetry_vs_capacity.png`) — Temperature & Discharge Duration Analysis:**
   - Inspection of `Ambient_Temperature` across all $50,285$ rows confirms a constant chamber setting of **$24.0^\circ\text{C}$ ($\text{SD} = 0.0^\circ\text{C}$)**. Thus, ambient temperature has zero variance and cannot be used as a dynamic exogenous predictor ($X_t$) in ARIMAX. Meanwhile, cycle discharge duration decreases linearly with capacity from $61.5\text{ min}$ to $46.5\text{ min}$.

---

## 11. Stationarity Analysis

To avoid test-set leakage when determining the integration order $d$, we executed the Augmented Dickey–Fuller (ADF) test (`autolag='AIC'`) on the chronological training sequence (Cycles 1–134) as well as the full 168-cycle series (`src/stationarity.py`). Table 2 reports the exact metrics.

| Series Tested | Integration Order ($d$) | ADF Test Statistic | $p$-Value | Lags Used | Obs ($N$) | Critical Value (1%) | Critical Value (5%) | Critical Value (10%) | Statistical Conclusion ($\alpha = 0.05$) |
| :--- | :---: | ---: | ---: | :---: | :---: | ---: | ---: | ---: | :--- |
| **Training Series $Q_t$ (Cycles 1–134)** | $d = 0$ | $-1.3520$ | $0.6051$ | $3$ | $130$ | $-3.4817$ | $-2.8840$ | $-2.5788$ | **Fail to Reject $H_0$** (Non-Stationary) |
| **Training First Diff $\Delta Q_t$ (Cycles 2–134)** | $d = 1$ | $-8.5062$ | $1.20 \times 10^{-13}$ | $2$ | $130$ | $-3.4817$ | $-2.8840$ | $-2.5788$ | **Reject $H_0$** (Stationary) |
| **Full Series $Q_t$ (Cycles 1–168)** | $d = 0$ | $-1.3659$ | $0.5985$ | $3$ | $164$ | $-3.4709$ | $-2.8793$ | $-2.5763$ | **Fail to Reject $H_0$** (Non-Stationary) |
| **Full First Diff $\Delta Q_t$ (Cycles 2–168)** | $d = 1$ | $-9.5790$ | $2.18 \times 10^{-16}$ | $2$ | $164$ | $-3.4709$ | $-2.8793$ | $-2.5763$ | **Reject $H_0$** (Stationary) |

**Justification of $d = 1$ (Answering Supporting Question 1):** On the training set, the level series $Q_t$ yields $\text{ADF} = -1.3520 > -2.8840$ ($p = 0.6051$), failing to reject the unit-root null hypothesis $H_0$. Furthermore, splitting the training sequence into three equal 44-cycle segments confirms a shifting mean: Segment 1 mean $= 1.9050\text{ Ah}$, Segment 2 mean $= 1.6005\text{ Ah}$, and Segment 3 mean $= 1.4152\text{ Ah}$. Applying first-order differencing $\Delta Q_t = Q_t - Q_{t-1}$ yields $\text{ADF} = -8.5062 \ll -3.4817$ ($p = 1.20 \times 10^{-13}$), decisively rejecting $H_0$ at the $1\%$ significance level. Consequently, **$d = 1$ is necessary and sufficient**; second-order differencing ($d = 2$) is unwarranted and would induce over-differencing noise.

---

## 12. ARIMA Baseline & Drift Methodology (Models 1 and 2)

### 12.1 Mathematical Formulation
- **Model 1 — Baseline $\text{ARIMA}(p, d, q)$ without Drift (`trend='n'`):**
  $$\left(1 - \sum_{i=1}^p \phi_i B^i\right) (1 - B)^d Q_t = \left(1 + \sum_{j=1}^q \theta_j B^j\right) \varepsilon_t, \quad \varepsilon_t \sim \text{WN}(0, \sigma^2),$$
  where $B Q_t = Q_{t-1}$. For $\text{ARIMA}(0, 1, 1)$, $\Delta Q_t = \varepsilon_t + \theta_1 \varepsilon_{t-1}$, and for $h \ge 2$ the multi-step forecast is constant: $\hat{Q}_{n+h|n} = \hat{Q}_{n+1|n}$.
- **Model 2 — $\text{ARIMA}(0, 1, 1)$ with Linear Drift (`trend='t'`):**
  $$\Delta Q_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1},$$
  where $c < 0$ is the deterministic per-cycle capacity drift rate. Over a multi-step horizon $h$, the forecast extrapolates linearly with slope $c$: $\hat{Q}_{n+h|n} = \hat{Q}_{n+1|n} + (h - 1)c$.

### 12.2 Training-Only Order Selection
With $d = 1$ fixed from the training ADF test, we evaluated eight candidate $(p, 1, q)$ configurations on the training set (Cycles 1–134), recording in-sample $\text{AIC}$, $\text{BIC}$, Ljung–Box residual white-noise $p$-values at lag 10, and inner chronological validation $\text{RMSE}$ (trained on Cycles 1–110, validated on Cycles 111–134). Table 3 summarizes the candidate search (`results/metrics/arima_order_selection.csv`).

| Candidate Order | Training AIC | Training BIC | Training HQIC | Inner Val RMSE (Ah) | Inner Val MAE (Ah) | Ljung–Box $p$ (Lag 10) | Converged |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **$\text{ARIMA}(0, 1, 1)$** | **$-593.5014$** | **$-587.7207$** | **$-591.1524$** | $0.04286$ | $0.03663$ | $0.6282$ | Yes |
| $\text{ARIMA}(1, 1, 0)$ | $-593.4628$ | $-587.6821$ | $-591.1138$ | $0.04286$ | $0.03663$ | $0.6246$ | Yes |
| $\text{ARIMA}(2, 1, 0)$ | $-592.7353$ | $-584.0643$ | $-589.2118$ | $0.04330$ | $0.03713$ | $0.7988$ | Yes |
| $\text{ARIMA}(1, 1, 1)$ | $-592.2636$ | $-583.5926$ | $-588.7400$ | $0.04422$ | $0.03806$ | $0.7566$ | Yes |
| $\text{ARIMA}(1, 1, 2)$ | $-590.9528$ | $-579.3914$ | $-586.2547$ | $0.04435$ | $0.03823$ | $0.8442$ | Yes |
| $\text{ARIMA}(3, 1, 1)$ | $-590.1467$ | $-575.6949$ | $-584.2740$ | $0.04433$ | $0.03824$ | $0.9260$ | Yes |
| **$\text{ARIMA}(2, 1, 1)$** | $-589.5255$ | $-577.9641$ | $-584.8273$ | **$0.02791$** | **$0.02181$** | $0.6598$ | Yes |
| $\text{ARIMA}(2, 1, 2)$ | $-588.8499$ | $-574.3982$ | $-582.9773$ | $0.04394$ | $0.03783$ | $0.8168$ | Yes |

- **Model 1 Selection:** $\text{ARIMA}(0, 1, 1)$ minimizes both $\text{AIC}$ ($-593.5014$) and $\text{BIC}$ ($-587.7207$) on the training set and fails to reject the Ljung–Box null hypothesis $H_0$ of zero residual autocorrelation ($p_{10} = 0.6282 > 0.05$).
- **Model 2 Fit:** Adding a linear drift term (`trend='t'`) to $\text{ARIMA}(0, 1, 1)$ on Cycles 1–134 improves training $\text{AIC}$ to $-598.2452$ ($\text{BIC} = -589.5741$) and estimates a drift rate of $c = -0.005251\text{ Ah/cycle}$ ($\theta_1 = -0.1178$).

---

## 13. Holt's Linear Trend & Damped Holt Methodology (Models 3 and 4)

### 13.1 Mathematical Formulation
**Model 3 — Holt's Linear Trend** (double exponential smoothing) decomposes the series into a smoothed capacity level $L_t$ and a per-cycle degradation slope $T_t$:
$$\begin{aligned}
\text{Level Equation:} \quad L_t &= \alpha Q_t + (1 - \alpha)(L_{t-1} + T_{t-1}), \quad 0 < \alpha \le 1, \\
\text{Trend Equation:} \quad T_t &= \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1}, \quad 0 \le \beta \le 1, \\
\text{$h$-Step Forecast:} \quad \hat{Q}_{t+h|t} &= L_t + h T_t.
\end{aligned}$$
**Model 4 — Damped Holt's Trend** introduces an autoregressive damping factor $0 < \phi < 1$ on the slope:
$$\begin{aligned}
L_t &= \alpha Q_t + (1 - \alpha)(L_{t-1} + \phi T_{t-1}), \\
T_t &= \beta (L_t - L_{t-1}) + (1 - \beta)\phi T_{t-1}, \\
\hat{Q}_{t+h|t} &= L_t + \left(\sum_{j=1}^h \phi^j\right) T_t.
\end{aligned}$$

### 13.2 Parameter Selection & Primary Model Justification (Zero Test Leakage)
Using only the inner chronological validation window (sub-train Cycles 1–110, validation Cycles 111–134, `results/metrics/holt_parameter_tuning.csv`):
- **Model 3 (Holt Linear Val-Tuned — Primary Enhanced Model):** The grid configuration $(\alpha = 0.10, \beta = 0.05)$ achieved the lowest 24-step validation $\text{RMSE} = 0.01732\text{ Ah}$ ($\text{MAE} = 0.01251\text{ Ah}$). Refit on the full 134-cycle training set, it estimates $L_{134} = 1.3318\text{ Ah}$ and local slope $T_{134} = -0.003275\text{ Ah/cycle}$ ($\text{AIC} = -887.95$, Figure 10).
- **Model 3b (Holt Linear Train-MLE — Diagnostic Variant):** Fitted strictly on the training set (Cycles 1–134) by minimizing 1-step-ahead Gaussian SSE ($\text{SSE} = 0.08284\text{ Ah}^2$, $\alpha = 0.8756, \beta = 0.0000, \text{AIC} = -982.08$), preserving the global training slope $T_0 = -0.005249\text{ Ah/cycle}$.
- **Why Their Parameters and Test Results Differ, and Why Model 3 (Val-Tuned) Is the Primary Enhanced Model:**
  1. **Different Optimization Objectives:** `Train-MLE` optimizes **1-step-ahead in-sample errors** on Cycles 1–134. Because battery capacity contains sharp rest-regeneration jumps (`Cycles 20, 48, 90`), 1-step SSE minimization selects a high level-smoothing weight ($\alpha = 0.8756$) so $\hat{Q}_{t+1|t}$ snaps immediately to the previous cycle's level, while setting $\beta = 0.0000$ to hold the average slope constant ($T_t \equiv T_0 = -0.005249\text{ Ah/cycle}$). Conversely, **validation tuning** directly minimizes **24-step-ahead out-of-sample forecast RMSE** on Cycles 111–134 (`val_rmse = 0.01732 Ah` for $\alpha=0.10, \beta=0.05$ vs. `0.04683 Ah` [rank 39 of 62] for `MLE_Linear`). Because the degradation rate decelerated between Cycles 91–134 ($-0.0033\text{ Ah/cycle}$) compared to Cycles 1–90 ($-0.0061\text{ Ah/cycle}$), allowing $\beta = 0.05$ to adapt to the local slope while smoothing out regeneration spikes ($\alpha = 0.10$) dramatically improved multi-step validation accuracy.
  2. **Why Test Results Differ Across Horizons:** On the held-out test set (Cycles 135–168), `Model 3 (Val-Tuned)` achieves the best accuracy across short and medium horizons ($h = 5$: $\text{RMSE} = 0.00472\text{ Ah}$ vs. $0.01467\text{ Ah}$ for `Train-MLE`; $h = 10$: $0.00910\text{ Ah}$ vs. $0.01572\text{ Ah}$; $h = 20$: $0.01477\text{ Ah}$ vs. $0.01682\text{ Ah}$). Only at the full 34-cycle horizon ($h = 34$) does `Train-MLE` achieve a lower RMSE ($0.01514\text{ Ah}$ vs. $0.02739\text{ Ah}$) because late-life fade accelerated again between Cycles 152 and 164 ($-0.0057\text{ Ah/cycle}$), matching the global training average slope ($-0.00525\text{ Ah/cycle}$).
  3. **Correct Model-Selection Procedure:** Because our research task is multi-step forecasting ($H = 34$) and model selection must be performed strictly *prior* to inspecting the held-out test set, choosing the model that minimizes multi-step chronological validation RMSE (`Model 3: Holt Linear Val-Tuned`, $\alpha = 0.10, \beta = 0.05$) is the only methodologically valid selection rule. Selecting `Train-MLE` based on its 34-cycle test RMSE would constitute post-hoc test selection. Therefore, **Model 3 (Val-Tuned)** is designated as our **Primary Enhanced Model**, while **Model 3b (Train-MLE)** is reported transparently for theoretical comparison.
- **Model 4 (Damped Holt Val-Tuned):** Selected via inner validation grid search ($\alpha = 0.60, \beta = 0.05, \phi = 0.98$, validation $\text{RMSE} = 0.01375\text{ Ah}$, refit training $\text{AIC} = -968.77$).

---

## 14. Experimental Setup and Evaluation Metrics

- **Chronological Split Protocol:** Strictly ordered 80% / 20% split with zero shuffling:
  - **Outer Training Set:** Cycles $1\text{–}134$ ($n_{\text{train}} = 134$ cycles, $2.0520\text{ Ah} \to 1.3547\text{ Ah}$).
  - **Inner Validation Split (for hyperparameter/order tuning):** Sub-train Cycles $1\text{–}110$, Validation Cycles $111\text{–}134$ ($n_{\text{val}} = 24$ cycles).
  - **Untouched Held-Out Test Set:** Cycles $135\text{–}168$ ($n_{\text{test}} = 34$ cycles, $1.3369\text{ Ah} \to 1.1721\text{ Ah}$, ending at $1.2038\text{ Ah}$).
- **Metric Definitions and Physical Interpretation:**
  $$\text{MAE} = \frac{1}{H}\sum_{h=1}^H |Q_{n+h} - \hat{Q}_{n+h}|, \quad \text{RMSE} = \sqrt{\frac{1}{H}\sum_{h=1}^H (Q_{n+h} - \hat{Q}_{n+h})^2},$$
  $$\text{MAPE} = \frac{100}{H}\sum_{h=1}^H \left|\frac{Q_{n+h} - \hat{Q}_{n+h}}{Q_{n+h}}\right|, \quad \text{sMAPE} = \frac{100}{H}\sum_{h=1}^H \frac{2|Q_{n+h} - \hat{Q}_{n+h}|}{|Q_{n+h}| + |\hat{Q}_{n+h}|}.$$
  - **Physical meaning in Ah:** $\text{MAE}$ measures the average absolute charge prediction error in Ampere-hours (where $0.02\text{ Ah}$ corresponds to $1.0\%$ of a $2.0\text{ Ah}$ nominal cell). $\text{RMSE}$ penalizes large misses quadratically, making it sensitive to late-horizon divergence and unmodeled regeneration spikes.
  - **Limitations of MAPE and why sMAPE is included:** Because $\text{MAPE}$ divides by actual capacity $Q_{n+h}$, the same $0.02\text{ Ah}$ error looks $1.67\times$ larger at late life ($Q = 1.20\text{ Ah}$) than at early life ($Q = 2.00\text{ Ah}$), and $\text{MAPE}$ asymmetrically penalizes over-forecasts ($\hat{Q} > Q$) when actual $Q$ drops. Including **symmetric MAPE ($\text{sMAPE}$)** normalizes by the average of actual and predicted capacity, mitigating over-prediction asymmetry.

---

## 15. Results and Model Comparison

### 15.1 Separation of Training, Validation, and Held-Out Multi-Step Test Performance
To ensure complete transparency without mixing evaluation regimes (`results/metrics/train_val_test_metrics.csv` and `results/metrics/model_comparison_metrics.csv`), Table 4 reports the Training (1-step in-sample, Cycles 1–134), Inner Validation (24-step out-of-sample, Cycles 111–134), and Held-Out Test (34-step out-of-sample, Cycles 135–168) metrics for every multi-step model.

| Model | Train RMSE (Ah) [1-Step, C1–134] | Val RMSE (Ah) [24-Step, C111–134] | Test MAE (Ah) [34-Step, C135–168] | Test RMSE (Ah) [34-Step, C135–168] | Test MAPE (%) | Test sMAPE (%) | Test RMSE vs. Model 1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Model 1: $\text{ARIMA}(0, 1, 1)$ — Baseline** | $0.02560$ | $0.04286$ | $0.10094$ | $0.11153$ | $8.203\%$ | $7.805\%$ | — |
| **Model 2: $\text{ARIMA}(0, 1, 1)$ with Drift** | $0.02496$ | $0.04689$ | **$0.01298$** | **$0.01522$** | **$1.030\%$** | **$1.028\%$** | **$-86.4\%$** |
| **Model 3: Holt's Linear Trend (Val-Tuned: $\alpha=0.10, \beta=0.05$)** | $0.03533$ | $0.01732$ | **$0.02253$** | **$0.02739$** | **$1.841\%$** | **$1.817\%$** | **$-75.4\%$** |
| **Model 3b: Holt's Linear Trend (Train-MLE: $\alpha=0.88, \beta=0.00$)** | $0.02486$ | $0.04683$ | **$0.01288$** | **$0.01514$** | **$1.022\%$** | **$1.020\%$** | **$-86.4\%$** |
| **Model 4: Damped Holt's Trend (Val-Tuned: $\alpha=0.60, \beta=0.05, \phi=0.98$)** | $0.02594$ | $0.01375$ | $0.06616$ | $0.07513$ | $5.390\%$ | $5.205\%$ | $-32.6\%$ |
| Candidate $\text{ARIMA}(1, 1, 1)$ (Multi-Step) | $0.02553$ | $0.04422$ | $0.09890$ | $0.10962$ | $8.040\%$ | $7.655\%$ | $-1.7\%$ |
| Candidate $\text{ARIMA}(2, 1, 1)$ (Val-Optimal, Multi-Step) | $0.02560$ | $0.02791$ | $0.10096$ | $0.11151$ | $8.205\%$ | $7.807\%$ | $-0.02\%$ |

### 15.2 Online 1-Step-Ahead Recalibration (Separate Evaluation Regime)
When a BMS receives a fresh capacity measurement after every single discharge cycle $t$ and forecasts only 1 cycle ahead ($h = 1$), performance is governed by 1-step local tracking rather than 34-cycle trend extrapolation (`results/metrics/online_recalibration_metrics.csv`, Table 5).

| Online Model (Expanding Window on Cycles 135–168) | Horizon ($h$) | Steps | MAE (Ah) | RMSE (Ah) | MAPE (%) | sMAPE (%) |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: |
| **Online 1-Step $\text{ARIMA}(0, 1, 1)$** | $1$ | $34$ | $0.00950$ | $0.01218$ | $0.757\%$ | $0.756\%$ |
| **Online 1-Step $\text{ARIMA}(0, 1, 1)$ with Drift** | $1$ | $34$ | **$0.00723$** | **$0.01125$** | **$0.577\%$** | **$0.579\%$** |
| **Online 1-Step Holt's Linear Trend ($\alpha=0.10, \beta=0.05$)** | $1$ | $34$ | $0.01236$ | $0.01482$ | $0.998\%$ | $0.996\%$ |

### 15.3 Forecast Horizon Analysis ($h = 5, 10, 20, 34$ Cycles)
Table 6 tracks multi-step forecast accuracy from Origin Cycle 134 across increasing forecast horizons (`results/metrics/horizon_comparison.csv`, Figure 12B).

| Horizon ($h$) | Test Window | Model 1: $\text{ARIMA}(0,1,1)$ RMSE (MAPE / sMAPE) | Model 2: $\text{ARIMA}(0,1,1)$+Drift RMSE (MAPE / sMAPE) | Model 3: Holt Val-Tuned RMSE (MAPE / sMAPE) | Model 3b: Holt Train-MLE RMSE (MAPE / sMAPE) | Model 4: Damped Holt RMSE (MAPE / sMAPE) |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| **$h = 5$** | C135–139 | $0.03377\text{ Ah}$ ($2.469\% / 2.437\%$) | $0.01485\text{ Ah}$ ($1.100\% / 1.094\%$) | **$0.00472\text{ Ah}$ ($0.306\% / 0.306\%$)** | $0.01467\text{ Ah}$ ($1.086\% / 1.080\%$) | $0.01898\text{ Ah}$ ($1.349\% / 1.339\%$) |
| **$h = 10$** | C135–144 | $0.04952\text{ Ah}$ ($3.588\% / 3.517\%$) | $0.01589\text{ Ah}$ ($1.190\% / 1.182\%$) | **$0.00910\text{ Ah}$ ($0.591\% / 0.589\%$)** | $0.01572\text{ Ah}$ ($1.176\% / 1.168\%$) | $0.03035\text{ Ah}$ ($2.146\% / 2.119\%$) |
| **$h = 20$** | C135–154 | $0.07092\text{ Ah}$ ($5.197\% / 5.047\%$) | $0.01692\text{ Ah}$ ($1.173\% / 1.170\%$) | **$0.01477\text{ Ah}$ ($0.983\% / 0.979\%$)** | $0.01682\text{ Ah}$ ($1.165\% / 1.162\%$) | $0.04407\text{ Ah}$ ($3.163\% / 3.104\%$) |
| **$h = 34$** | C135–168 | $0.11153\text{ Ah}$ ($8.203\% / 7.805\%$) | $0.01522\text{ Ah}$ ($1.030\% / 1.028\%$) | $0.02739\text{ Ah}$ ($1.841\% / 1.817\%$) | **$0.01514\text{ Ah}$ ($1.022\% / 1.020\%$)** | $0.07513\text{ Ah}$ ($5.390\% / 5.205\%$) |

### 15.4 Leakage-Free Rolling-Origin Evaluation Across Five Chronological Cutoffs
To eliminate lookahead bias, at each origin $t_{\text{origin}} \in \{90, 105, 120, 134, 148\}$, Holt and Damped Holt smoothing parameters were tuned strictly on an inner validation split of $\{1, \dots, t_{\text{origin}}\}$ before forecasting the next $h = 20$ cycles (`results/metrics/rolling_origin_evaluation.csv` and `results/metrics/rolling_origin_metrics.csv`, Table 7 and Figure 13A).

| Origin Cycle ($t_{\text{origin}}$) | Test Window ($h=20$) | Model 1: $\text{ARIMA}(0,1,1)$ RMSE (MAPE) | Model 2: $\text{ARIMA}(0,1,1)$+Drift RMSE (MAPE) | Model 3: Causal Holt Linear RMSE (MAPE) | Model 4: Causal Damped Holt RMSE (MAPE) |
| :---: | :---: | ---: | ---: | ---: | ---: |
| **Origin 90** | Cycles 91–110 | $0.15364\text{ Ah}$ ($10.263\%$) | $0.08350\text{ Ah}$ ($5.638\%$) | **$0.08242\text{ Ah}$ ($5.552\%$)** | $0.14087\text{ Ah}$ ($8.960\%$) |
| **Origin 105** | Cycles 106–125 | $0.05412\text{ Ah}$ ($3.672\%$) | $0.02601\text{ Ah}$ ($1.313\%$) | **$0.01851\text{ Ah}$ ($1.183\%$)** | $0.04424\text{ Ah}$ ($3.040\%$) |
| **Origin 120** | Cycles 121–140 | $0.05097\text{ Ah}$ ($3.450\%$) | $0.02256\text{ Ah}$ ($1.371\%$) | $0.01872\text{ Ah}$ ($1.083\%$) | **$0.01750\text{ Ah}$ ($1.090\%$)** |
| **Origin 134** | Cycles 135–154 | $0.07092\text{ Ah}$ ($5.197\%$) | $0.01692\text{ Ah}$ ($1.173\%$) | **$0.01327\text{ Ah}$ ($0.864\%$)** | $0.02605\text{ Ah}$ ($1.789\%$) |
| **Origin 148** | Cycles 149–168 | $0.05802\text{ Ah}$ ($4.122\%$) | $0.01787\text{ Ah}$ ($0.991\%$) | **$0.01668\text{ Ah}$ ($0.940\%$)** | $0.02281\text{ Ah}$ ($1.493\%$) |
| **Mean $\pm$ SD (5 Origins)** | — | **$0.07753 \pm 0.04322\text{ Ah}$** ($5.341\% \pm 2.833\%$) | **$0.03337 \pm 0.02826\text{ Ah}$** ($2.097\% \pm 1.985\%$) | **$0.02992 \pm 0.02943\text{ Ah}$** ($1.924\% \pm 2.032\%$) | **$0.05030 \pm 0.05162\text{ Ah}$** ($3.275\% \pm 3.261\%$) |

### 15.5 Residual Diagnostics and Ljung–Box Autocorrelation Testing
Table 8 summarizes the training-set residual diagnostics across all five fitted models, including exact Ljung–Box $Q(m)$ statistics and $p$-values at all four tested lags (`Lag 5`, `Lag 10`, `Lag 15`, `Lag 20`) and Shapiro–Wilk normality tests (`results/metrics/residual_diagnostics.csv`, illustrated in Figure 9 and Figure 14 `fig14_comprehensive_residual_diagnostics.png`).

| Model | $N$ | Mean (Ah) | SD (Ah) | Skew / Kurt | Ljung–Box Lag 5 $Q$ ($p$) | Ljung–Box Lag 10 $Q$ ($p$) | Ljung–Box Lag 15 $Q$ ($p$) | Ljung–Box Lag 20 $Q$ ($p$) | Shapiro–Wilk $W$ ($p$) | Ljung–Box $H_0$ Decision ($\alpha=0.05$) |
| :--- | :---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Model 1: $\text{ARIMA}(0, 1, 1)$** | $133$ | $-0.00547$ | $0.02510$ | $3.86 / 19.04$ | $6.357$ ($0.2730$) | $8.006$ ($0.6282$) | $9.435$ ($0.8537$) | $12.363$ ($0.9030$) | $0.604$ ($2.29 \times 10^{-17}$) | **Fail to Reject $H_0$** |
| **Model 2: $\text{ARIMA}(0, 1, 1)$ + Drift** | $133$ | $-0.00001$ | $0.02505$ | $3.91 / 19.13$ | $6.710$ ($0.2431$) | $8.437$ ($0.5862$) | $9.857$ ($0.8287$) | $12.661$ ($0.8915$) | $0.602$ ($2.09 \times 10^{-17}$) | **Fail to Reject $H_0$** |
| **Model 3: Holt Linear ($\alpha=0.10, \beta=0.05$)** | $134$ | $+0.00788$ | $0.03457$ | $1.90 / 4.90$ | $125.45$ ($2.19 \times 10^{-25}$) | $127.47$ ($1.53 \times 10^{-22}$) | $129.10$ ($3.19 \times 10^{-20}$) | $144.05$ ($8.60 \times 10^{-21}$) | $0.838$ ($7.87 \times 10^{-11}$) | **Reject $H_0$** (Autocorrelated) |
| **Model 3b: Holt Linear MLE ($\alpha=0.88, \beta=0.00$)** | $134$ | $-0.00000$ | $0.02496$ | $3.92 / 19.29$ | $6.826$ ($0.2339$) | $8.575$ ($0.5728$) | $10.002$ ($0.8196$) | $12.781$ ($0.8866$) | $0.601$ ($1.73 \times 10^{-17}$) | **Fail to Reject $H_0$** |
| **Model 4: Damped Holt ($\alpha=0.60, \beta=0.05, \phi=0.98$)** | $134$ | $-0.00138$ | $0.02600$ | $3.59 / 16.51$ | $15.904$ ($0.0071$) | $19.589$ ($0.0334$) | $20.801$ ($0.1433$) | $23.301$ ($0.2743$) | $0.645$ ($1.54 \times 10^{-16}$) | **Reject $H_0$** (Lags 5, 10) |

**Statistical Interpretation of Residual Diagnostics (Answering Supporting Question 5):**
1. **Failing to Reject $H_0$ Does Not "Prove" White Noise:** Under the Ljung–Box portmanteau test, the null hypothesis $H_0$ states that residual autocorrelations $\rho_1, \dots, \rho_m$ are jointly zero. Models 1, 2, and 3b yield $p > 0.23$ across all tested lags ($m \in \{5, 10, 15, 20\}$), meaning we **fail to reject $H_0$ at $\alpha = 0.05$**. Statistically, failing to reject $H_0$ indicates no significant linear autocorrelation is detected; it does *not* prove the residuals are strict white noise. Moreover, adding drift in Model 2 eliminates the negative residual mean bias of Model 1 (reducing $|\bar{e}|$ from $0.00547\text{ Ah}$ to $0.000006\text{ Ah}$).
2. **Why Validation-Tuned Holt ($\alpha=0.10, \beta=0.05$) Rejects $H_0$ on 1-Step Residuals:** Because $\alpha = 0.10$ applies strong low-pass smoothing to filter out discrete rest-regeneration spikes (Cycles 20, 48, 90) and stabilize multi-step slope extrapolation over 24–34 cycles, its 1-step in-sample level $L_t$ intentionally lags abrupt jumps, producing positive 1-step residual autocorrelation ($p \le 3.19 \times 10^{-20}$ across lags 5–20). By contrast, `Train-MLE Holt` ($\alpha = 0.88$) optimizes 1-step likelihood and fails to reject $H_0$ at all tested lags ($p_{5} = 0.2339, p_{10} = 0.5728, p_{15} = 0.8196, p_{20} = 0.8866$). Meanwhile, `Model 4 (Damped Holt)` rejects $H_0$ at short lags ($p_5 = 0.0071, p_{10} = 0.0334$) while failing to reject at longer lags ($p_{15} = 0.1433, p_{20} = 0.2743$).
3. **Non-Normality from Regeneration Spikes (Shapiro–Wilk $p < 10^{-10}$):** All five models decisively reject Gaussianity under the Shapiro–Wilk test ($p \in [1.73 \times 10^{-17}, 7.87 \times 10^{-11}]$) with positive skewness ($1.90\text{–}3.92$) and leptokurtic tails ($\text{kurtosis} = 4.90\text{–}19.29$) caused by discrete positive regeneration impulses (`+0.12` to `+0.15 Ah`) following experimental rest pauses.

### 15.6 Causal Remaining Useful Life (RUL) Estimation at Origin Cycle 60
Using NASA's experimental End-of-Life criterion ($30\%$ fade from $2.0\text{ Ah}$ nominal rating, i.e., $Q_{\text{EOL}} = 1.4000\text{ Ah}$), the battery crosses below $1.4000\text{ Ah}$ at **Cycle 112**. Launching a strictly causal long-range forecast from **Origin Cycle 60** ($Q_{60} = 1.6447\text{ Ah}$, actual $\text{RUL} = 112 - 60 = 52$ cycles, tuned solely on Cycles 1–60 with $\alpha=0.10, \beta=0.10$, Figure 13B):
- **Model 1 ($\text{ARIMA}(0,1,1)$ Baseline):** Predicts a flat horizontal line at $1.643\text{ Ah}$ and **never crosses $1.4000\text{ Ah}$** ($\text{Predicted RUL} = \text{None}$), failing to warn of impending EOL.
- **Model 2 ($\text{ARIMA}(0,1,1)$ with Drift):** Predicts EOL crossing at **Cycle 96** ($\text{Predicted RUL} = 36$ cycles, error $= -16$ cycles).
- **Model 3 (Causally Tuned Holt's Linear Trend):** Projects EOL crossing at **Cycle 98** ($\text{Predicted RUL} = 38$ cycles vs. actual $52$ cycles, an early/conservative margin of $-14$ cycles caused by the large unexpected $+0.1518\text{ Ah}$ rest-regeneration spike at Cycle 90 that delayed actual EOL by roughly a dozen cycles).

---

## 16. Discussion

1. **Why Un-Drifted $\text{ARIMA}(p,1,q)$ Fails on Multi-Step Battery Forecasting:** In standard Box–Jenkins $\text{ARIMA}(p,1,q)$ without a deterministic drift term (`trend='n'`), the differenced process $\Delta Q_t$ has zero unconditional mean. Consequently, multi-step forecasts $\hat{Q}_{n+h|n}$ rapidly converge to a horizontal asymptote ($1.3552\text{ Ah}$ for $\text{ARIMA}(0,1,1)$, Figure 15A). Because actual battery capacity continues to fade from $1.3369\text{ Ah}$ down to $1.1721\text{ Ah}$, $\text{ARIMA}$'s forecast error grows monotonically with horizon $h$ (from $2.469\%$ MAPE at $h=5$ to $8.203\%$ MAPE at $h=34$).
2. **Equivalence of Drifted $\text{ARIMA}(0,1,1)$ and MLE Holt's Linear Trend:** Adding a linear drift term (`trend='t'`) in Model 2 drops 34-cycle test $\text{MAPE}$ from $8.203\%$ to $1.030\%$ ($\text{RMSE} = 0.01522\text{ Ah}$), nearly identical to Training-MLE Holt Linear ($\text{MAPE} = 1.022\%, \text{RMSE} = 0.01514\text{ Ah}$, Figures 15B and 15D). Mathematically, $\text{ARIMA}(0,1,1)$ with constant drift is the state-space equivalent of Holt's Linear Trend when $\beta \to 0$ and $\alpha = 1 + \theta_1 = 1 - 0.1178 = 0.8822$ (matching our empirical MLE estimate $\alpha = 0.8756, \beta = 0.0000$). Meanwhile, validation-tuned Holt ($\alpha = 0.10, \beta = 0.05$, Figure 15C) achieves the lowest short-to-medium horizon error ($0.306\%$ MAPE at $h=5$, $0.591\%$ at $h=10$, $0.983\%$ at $h=20$) and the best 5-origin rolling RMSE ($0.02992\text{ Ah}$).
3. **Why Damped Holt Underperformed Un-Damped Linear Holt Here:** Damped Holt ($\phi = 0.98$, Figure 15E) achieved $\text{MAPE} = 5.390\%$, outperforming un-drifted $\text{ARIMA}(0,1,1)$ ($8.203\%$) by $32.6\%$ in RMSE but trailing un-damped Holt Linear ($1.841\%$). Between Cycles 135 and 164, Cell B0006 did *not* plateau; it continued losing capacity at $-0.0057\text{ Ah/cycle}$. Damping the negative slope caused the forecast to flatten prematurely above the actual trajectory.
4. **Impact of Capacity Regeneration Spikes:** All models experience localized positive residuals at Cycle 151 (where capacity briefly jumped from $1.2638\text{ Ah}$ to $1.3018\text{ Ah}$ after a rest pause) and Cycles 167–168 ($1.1744 \to 1.2038\text{ Ah}$). Pure univariate cycle-indexed models cannot anticipate when an operator pauses the test bench unless rest duration $\Delta t_{\text{rest}}$ is provided as an exogenous variable.

---

## 17. Dataset and Methodological Limitations

1. **Single-Battery Scope (`NASA_B0006`) and Sample Size ($N = 168$ Discharge Observations):** The Mendeley V2 / NASA B0006 dataset comprises $168$ sequential discharge-capacity observations from a single 18650 $\text{LiCoO}_2$/graphite cell ($n_{\text{train}} = 134, n_{\text{test}} = 34$). Because manufacturing tolerances and electrode microstructure vary across cells, parameters tuned on B0006 cannot be generalized directly to other cells or chemistries (e.g., LFP, NMC, NCA) without per-cell recalibration or multi-cell hierarchical validation.
2. **Constant Ambient Temperature ($24^\circ\text{C}$) and Fixed Discharge Protocol ($2.0\text{ A}$ CC):** All $50,285$ intra-cycle measurements were recorded in an isothermal chamber at $24.0^\circ\text{C}$ ($\text{SD} = 0.0^\circ\text{C}$) under a constant $2.0\text{ A}$ discharge current. Real-world electric vehicle packs experience dynamic C-rates, regenerative braking, and wide seasonal temperature swings ($-10^\circ\text{C}$ to $+45^\circ\text{C}$) that alter internal resistance and Arrhenius degradation kinetics.
3. **Unmodeled Rest-Induced Capacity Regeneration:** Because cycle-level summaries omit the calendar rest duration ($\Delta t_{\text{rest}}$) between consecutive cycles, univariate cycle-indexed models treat self-recharge recovery spikes (e.g., $+0.1518\text{ Ah}$ at Cycle 90, $+0.0380\text{ Ah}$ at Cycle 151) as heavy-tailed stochastic innovations rather than deterministic relaxation recovery.
4. **Structural Limitations of Linear Statistical Forecasting Near Aging Knees:** Classical univariate models ($\text{ARIMA}$ with drift and Holt's Linear Trend) extrapolate locally linear trends and lack electrochemical state physics. While linear slope extrapolation excels across mid-to-late life on Cell B0006, batteries undergoing a sudden late-life "aging knee" (accelerated nonlinear collapse due to lithium plating) require online rolling recalibration, covariates, or nonlinear state-space models.

---

## 18. Conclusion and Future Scope

This project demonstrated a complete, audited, and reproducible statistical time-series forecasting framework for lithium-ion battery capacity degradation. Answering our central research question, **trend-aware statistical models—Holt's Linear Trend ($\text{RMSE} = 0.02739\text{ Ah}, \text{MAPE} = 1.841\%$ Val-Tuned; $\text{RMSE} = 0.01514\text{ Ah}, \text{MAPE} = 1.022\%$ Train-MLE) and $\text{ARIMA}(0,1,1)$ with Linear Drift ($\text{RMSE} = 0.01522\text{ Ah}, \text{MAPE} = 1.030\%$)—substantially outperform the conventional un-drifted $\text{ARIMA}(0,1,1)$ baseline ($\text{RMSE} = 0.11153\text{ Ah}, \text{MAPE} = 8.203\%$)** across a 34-cycle held-out test horizon, cutting RMSE by **75.4% to 86.4%** and consistently outperforming un-drifted $\text{ARIMA}$ across all forecast horizons ($h = 5, 10, 20, 34$) and all five leakage-free rolling-origin cutoffs. Furthermore, when single-cycle online updates are available in a BMS, rolling 1-step recalibration reduces forecast error below **1.0% MAPE** ($0.577\%$ for drifted $\text{ARIMA}$).

**Future Scope:**
- Incorporate inter-cycle rest duration ($\Delta t_{\text{rest}}$) and multi-temperature chamber datasets (e.g., Mendeley Temperature-Enriched SOH dataset, `10.17632/sxtrrxkbmm.1`) using **ARIMAX / State-Space Exponential Smoothing with exogenous covariates**.
- Combine Holt's linear level/slope tracking with regime-switching models to capture sudden "knee-point" degradation transitions.

---

## 19. References

1. **Veeramsetty, V., & Keerthi Sri, G. V.** (2026). *EV Lithium Ion Battery State of Health Dataset* (Version 2). Mendeley Data. DOI: [10.17632/7ctkcfnsdw.2](https://doi.org/10.17632/7ctkcfnsdw.2).
2. **Saha, B., & Goebel, K.** (2007). *Battery Data Set*. NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA. [https://data.nasa.gov/dataset/li-ion-battery-aging-datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets).
3. **Zhou, Y., & Huang, M.** (2016). Lithium-ion batteries remaining useful life prediction based on a mixture of empirical mode decomposition and ARIMA model. *Microelectronics Reliability*, 65, 265–273. DOI: [10.1016/j.microrel.2016.07.151](https://doi.org/10.1016/j.microrel.2016.07.151).
4. **Pan, C., Chen, Y., Wang, L., & He, Z.** (2019). Lithium-ion Battery Remaining Useful Life Prediction Based on Exponential Smoothing and Particle Filter. *International Journal of Electrochemical Science*, 14(10), 9537–9551. DOI: [10.20964/2019.10.15](https://doi.org/10.20964/2019.10.15).
5. **Chen, L., Xu, L., & Zhou, Y.** (2018). Novel Approach for Lithium-Ion Battery On-Line Remaining Useful Life Prediction Based on Permutation Entropy. *Energies*, 11(4), 820. DOI: [10.3390/en11040820](https://doi.org/10.3390/en11040820).
6. **Zhang, Y., Xiong, R., He, H., & Pecht, M. G.** (2018). Long Short-Term Memory Recurrent Neural Network for Remaining Useful Life Prediction of Lithium-Ion Batteries. *IEEE Transactions on Vehicular Technology*, 67(7), 5695–5705. DOI: [10.1109/TVT.2018.2805189](https://doi.org/10.1109/TVT.2018.2805189).
7. **Fauzi, M. R., Yudistira, N., & Mahmudy, W. F.** (2024). State-of-Health Prediction of Lithium-Ion Batteries Using Exponential Smoothing Transformer With Seasonal and Growth Embedding. *IEEE Access*, 12, 14659–14670. DOI: [10.1109/ACCESS.2024.3357736](https://doi.org/10.1109/ACCESS.2024.3357736).
8. **Luh, M., & Blank, T.** (2024). Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell. *Scientific Data*, 11, 1004. DOI: [10.1038/s41597-024-03831-x](https://doi.org/10.1038/s41597-024-03831-x).
9. **Liao, X., Lyu, P., Zou, H., & Zeng, Z.** (2026). SDMnet: A seasonal-trend decomposition guided data augmentation and prediction framework for lithium-ion battery capacity forecasting. *Journal of Energy Storage*, Article 121800 (`S2352-152X(26)01464-7`). DOI: [10.1016/j.est.2026.121800](https://doi.org/10.1016/j.est.2026.121800).
