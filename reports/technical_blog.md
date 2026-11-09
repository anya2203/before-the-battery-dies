# Before the Battery Dies: Why Trend-Aware Statistical Models Beat Standard ARIMA at Forecasting Lithium-Ion Capacity Fade

Every electric vehicle driver, grid storage operator, and smartphone owner faces the same physical reality: **lithium-ion batteries age with every charge–discharge cycle**. A fresh cylindrical 18650 lithium-ion cell rated for $2.0\text{ Ampere-hours (Ah)}$ might initially deliver over $2.05\text{ Ah}$ on the test bench. Yet as solid-electrolyte interphase (SEI) layers thicken and active lithium ions become trapped through repeated cycling, the battery's maximum deliverable charge steadily declines. This phenomenon—known as **capacity fade**—quietly erodes EV driving range and narrows the safety margins of energy storage systems.

Traditional Battery Management Systems (BMS) excel at diagnostic monitoring: measuring present voltage, current, and temperature to estimate current **State of Health ($\text{SOH}$)**:

$$\text{SOH}_t (\%) = \frac{Q_t}{Q_{\text{initial}}} \times 100$$

where $Q_t$ is the measured discharge capacity at cycle $t$ and $Q_{\text{initial}}$ is the initial reference capacity. However, preventive maintenance requires shifting from *diagnostics* ("What is the battery's health today?") to *prognostics* ("What will the battery's capacity be 30 cycles from now, and when will it cross its End-of-Life threshold?").

In our undergraduate Time Series Analysis (TSA) research project, **Before the Battery Dies**, we investigated a central question: **Can trend-aware statistical forecasting models ($\text{ARIMA}$ with Linear Drift, Holt's Linear Trend, and Damped Holt) predict future lithium-ion battery capacity degradation more accurately than a conventional Box–Jenkins $\text{ARIMA}(0,1,1)$ baseline?**

---

## 1. The Dataset: Cycle-Indexed Time Series vs. Wall-Clock Time

To ground our experiment in real electrochemistry, we analyzed the **EV Lithium Ion Battery State of Health Dataset** (Mendeley Data V2, DOI: `10.17632/7ctkcfnsdw.2`, by V. Veeramsetty and K. S. G.V.), derived from NASA Ames Prognostics Center of Excellence (PCoE) aging experiments on **Cell B0006** at a constant $24^\circ\text{C}$ ambient temperature.

Before writing a single line of modeling code, our data audit uncovered two vital domain truths:
1. **Battery Degradation is Indexed by Cycle Number, Not Calendar Dates:** While raw telemetry (`Discharge AH.xlsx`, $50,285$ rows) logs intra-cycle elapsed seconds, capacity fade occurs across sequential discharge cycles $t \in \{1, 2, \dots, 168\}$. Furthermore, NASA's raw operation counter (`2` through `614`) interleaves charge, discharge, and impedance tests; mapping these to sequential discharge cycle indices $t = 1, \dots, 168$ is essential for valid time-series modeling.
2. **Capacity Fade is Interrupted by "Self-Recharge" Regeneration Spikes:** Over 168 cycles, Cell B0006 degrades from $Q_1 = 2.0520\text{ Ah}$ ($100.0\%\text{ SOH}$) down to a minimum of $Q_{164} = 1.1721\text{ Ah}$ ($57.12\%\text{ SOH}$). However, whenever the battery rests between test blocks, electrochemical relaxation briefly restores deliverable charge—producing sudden upward spikes at **Cycle 20 ($+0.122\text{ Ah}$)**, **Cycle 48 ($+0.119\text{ Ah}$)**, **Cycle 90 ($+0.152\text{ Ah}$)**, and **Cycle 151 ($+0.038\text{ Ah}$)**.

---

## 2. Stationarity Diagnostics: Why Differencing is Necessary

A time series is stationary only if its mean, variance, and autocorrelation structure remain invariant over time. Visually, battery capacity violates stationarity due to its relentless downward trajectory. Splitting our chronological training set (Cycles 1–134) into three equal segments confirmed a rapidly falling mean: $1.905\text{ Ah}$ (first third), $1.601\text{ Ah}$ (middle third), and $1.415\text{ Ah}$ (final third).

Running the **Augmented Dickey–Fuller (ADF)** unit-root test strictly on the training sequence confirmed this formally:
- **Level Series $Q_t$ ($d = 0$):** $\text{ADF Statistic} = -1.3520$, $p = 0.6051$ ($> 0.05$ critical value of $-2.8840$). We fail to reject the null hypothesis of a unit root; the raw series is non-stationary.
- **First-Differenced Series $\Delta Q_t = Q_t - Q_{t-1}$ ($d = 1$):** $\text{ADF Statistic} = -8.5062$, $p = 1.20 \times 10^{-13}$ ($\ll 0.01$ critical value of $-3.4817$). First-order differencing renders the series stationary, establishing $d = 1$.

---

## 3. Fair Comparison of Four Statistical Forecasting Models

To prevent data leakage, we split the 168-cycle trajectory chronologically (80% training: Cycles 1–134; 20% untouched held-out test set: Cycles 135–168, horizon $H = 34$ cycles) and used an inner validation window (Cycles 111–134) for hyperparameter selection.

### Model 1 (Baseline): $\text{ARIMA}(0, 1, 1)$ without Drift
Among eight candidate orders with $d=1$, **$\text{ARIMA}(0, 1, 1)$** achieved the optimal Akaike Information Criterion ($\text{AIC} = -593.50$) and failed to reject the Ljung–Box null hypothesis of zero residual autocorrelation ($p_{10} = 0.628$). However, a standard un-drifted $\text{ARIMA}(p, 1, q)$ model assumes the differenced series $\Delta Q_t$ fluctuates around zero mean. Consequently, when asked to forecast $h = 1, \dots, 34$ cycles into the future without seeing test labels, its multi-step prediction plateaus horizontally at $1.3552\text{ Ah}$—missing the continued degradation down to $1.1721\text{ Ah}$.

### Model 2: $\text{ARIMA}(0, 1, 1)$ with Linear Drift
Augmenting $\text{ARIMA}(0, 1, 1)$ with a deterministic drift constant $c = -0.00525\text{ Ah/cycle}$ (`trend='t'`, $\text{AIC} = -598.25$, Ljung–Box $p_{10} = 0.586$) propagates the average per-cycle capacity loss across the multi-step forecast horizon.

### Models 3 & 4: Holt's Linear Trend and Damped Holt's Trend
**Holt's Linear Trend** (double exponential smoothing) explicitly tracks two evolving states—the current capacity **level ($L_t$)** and the per-cycle **degradation rate ($T_t$)**:

$$L_t = \alpha Q_t + (1 - \alpha)(L_{t-1} + T_{t-1}), \quad T_t = \beta(L_t - L_{t-1}) + (1 - \beta)T_{t-1}, \quad \hat{Q}_{t+h|t} = L_t + h T_t$$

Tuned on our inner validation window ($\alpha = 0.10, \beta = 0.05$), Model 3 estimated a terminal training level $L_{134} = 1.3318\text{ Ah}$ and a local degradation slope $T_{134} = -0.00328\text{ Ah/cycle}$, while training-only Maximum Likelihood Estimation (Model 3b: $\alpha = 0.88, \beta = 0.00$) preserved the global slope ($-0.00525\text{ Ah/cycle}$) with uncorrelated 1-step residuals ($p_{10} = 0.573$). Model 4 (**Damped Holt**, $\phi = 0.98$) geometrically damps the projected slope over long horizons.

---

## 4. Quantitative Results & Key Takeaways

Evaluating all models on the identical 34-cycle held-out test period (Cycles 135–168) revealed a clear hierarchy:

| Forecasting Model (Held-Out Test: Cycles 135–168) | Horizon ($H$) | MAE (Ah) | RMSE (Ah) | MAPE (%) | sMAPE (%) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| **Model 1: $\text{ARIMA}(0, 1, 1)$ — Baseline** | 34 | 0.1009 | 0.1115 | 8.20% | 7.81% |
| **Model 2: $\text{ARIMA}(0, 1, 1)$ with Linear Drift** | 34 | **0.0130** | **0.0152** | **1.03%** | **1.03%** |
| **Model 3: Holt's Linear Trend (Val-Tuned: $\alpha=0.10, \beta=0.05$)** | 34 | **0.0225** | **0.0274** | **1.84%** | **1.82%** |
| **Model 3b: Holt's Linear Trend (Train-MLE: $\alpha=0.88, \beta=0.00$)** | 34 | **0.0129** | **0.0151** | **1.02%** | **1.02%** |
| **Model 4: Damped Holt's Trend ($\alpha=0.60, \beta=0.05, \phi = 0.98$)** | 34 | 0.0662 | 0.0751 | 5.39% | 5.21% |
| *Online 1-Step Recalibrated $\text{ARIMA}(0, 1, 1)$ with Drift* | 1 (×34) | *0.0072* | *0.0112* | *0.58%* | *0.58%* |

Three practical engineering lessons stand out from our experiments:
1. **Explicit Slope Modeling Cuts Multi-Step Forecast Error by 75%–86%:** Validation-tuned Holt's Linear Trend reduced 34-cycle test RMSE from $0.1115\text{ Ah}$ to $0.0274\text{ Ah}$ ($\text{MAPE}$ from $8.20\%$ down to $1.84\%$), achieved sub-1% MAPE across horizons $h = 5, 10, 20$ cycles, and achieved the lowest 5-origin rolling RMSE ($0.0299 \pm 0.0294\text{ Ah}$ vs. $0.0775 \pm 0.0432\text{ Ah}$ for Model 1) across Cycles 90, 105, 120, 134, and 148.
2. **Adding Drift Reconciles ARIMA and Exponential Smoothing:** Including linear drift in $\text{ARIMA}(0,1,1)$ cut test $\text{MAPE}$ from $8.20\%$ to $1.03\%$, matching Training-MLE Holt Linear ($1.02\%$). Mathematically, $\text{ARIMA}(0,1,1)$ with drift is the state-space dual of Holt's Linear Trend when $\beta \to 0$.
3. **Remaining Useful Life (RUL) Needs Trend Projection:** When forecasting causally from Cycle 60 ($1.645\text{ Ah}$) toward NASA's $1.400\text{ Ah}$ End-of-Life threshold (actual crossing at Cycle 112, $\text{RUL} = 52$ cycles), un-drifted $\text{ARIMA}$ plateaued and never predicted failure, whereas causally tuned Holt's Linear Trend and drifted $\text{ARIMA}$ predicted threshold crossing at Cycles 98 and 96 ($\text{RUL} = 38$ and $36$ cycles), providing a conservative maintenance buffer ahead of End of Life.

Before a lithium-ion battery dies, its historical cycle curve leaves a clear statistical signature. By pairing rigorous stationarity diagnostics with trend-aware statistical models, engineers can forecast capacity fade accurately with transparent, lightweight algorithms suitable for real-world Battery Management Systems.
