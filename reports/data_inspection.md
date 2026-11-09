# Phase 1 — Raw Dataset Inspection & Data Quality Audit

## 1. Dataset Provenance
- **Primary Source:** EV Lithium Ion Battery State of Health Dataset (Mendeley Data, V2, DOI: `10.17632/7ctkcfnsdw.2`)
- **Authors:** Venkataramana Veeramsetty and Keerthi Sri G.V.
- **Underlying Experimental Source:** NASA Ames Prognostics Center of Excellence (PCoE) Lithium-Ion Battery Aging Dataset (Cell **B0006**, cycled at constant $24^\circ\text{C}$ ambient temperature).

## 2. File-by-File Schema Audit

### `Discharge AH.xlsx` — Sheet `Sheet1`
- **Shape:** `50285 rows × 9 columns`
- **Duplicate Rows:** `0`
- **Total Missing Cells:** `1`

| Column Name | Data Type | Missing | Unique | Min / First | Max / Last |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `Cycle_Number` | `int64` | 0 | 168 | 2 | 614 |
| `Cycle_Type` | `object` | 0 | 1 | discharge | discharge |
| `Ambient_Temperature` | `int64` | 0 | 1 | 24 | 24 |
| `Time` | `float64` | 0 | 32282 | 0.000000 | 3690.234000 |
| `Current_measured` | `float64` | 0 | 50285 | -2.027018 | 0.008809 |
| `Voltage_measured` | `float64` | 0 | 50285 | 2.120698 | 4.222516 |
| `Current Mod` | `float64` | 0 | 50285 | -0.008809 | 2.027018 |
| `delta t` | `float64` | 0 | 1223 | -3690.234000 | 20.500000 |
| `Discharge Ah` | `float64` | 1 | 50284 | -0.006986 | 0.011437 |

**First 5 Sample Rows:**
```text
   Cycle_Number Cycle_Type  Ambient_Temperature    Time  Current_measured  Voltage_measured  Current Mod  delta t  Discharge Ah
0             2  discharge                   24   0.000         -0.002366          4.179800     0.002366    0.000      0.000000
1             2  discharge                   24  16.781         -0.000434          4.179823     0.000434   16.781      0.000002
2             2  discharge                   24  35.703         -2.014242          3.966528     2.014242   18.922      0.010587
3             2  discharge                   24  53.781         -2.008730          3.945886     2.008730   18.078      0.010087
4             2  discharge                   24  71.922         -2.013381          3.930354     2.013381   18.141      0.010146
```

### `Discharge Summary.xlsx` — Sheet `Sheet1`
- **Shape:** `168 rows × 2 columns`
- **Duplicate Rows:** `0`
- **Total Missing Cells:** `0`

| Column Name | Data Type | Missing | Unique | Min / First | Max / Last |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `Dis-ch Cycle` | `int64` | 0 | 168 | 2 | 614 |
| `Dis-chg Ah` | `float64` | 0 | 168 | 1.172108 | 2.051978 |

**First 5 Sample Rows:**
```text
   Dis-ch Cycle  Dis-chg Ah
0             2    2.051978
1             4    2.038456
2             6    2.028602
3             8    2.016546
4            10    2.016070
```

### `SoH Data.xlsx` — Sheet `Sheet1`
- **Shape:** `168 rows × 4 columns`
- **Duplicate Rows:** `0`
- **Total Missing Cells:** `0`

| Column Name | Data Type | Missing | Unique | Min / First | Max / Last |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `Charge Cycle` | `int64` | 0 | 168 | 2 | 614 |
| `Cycle #` | `int64` | 0 | 168 | 1 | 168 |
| `Dis-chg Ah` | `float64` | 0 | 168 | 1.172108 | 2.051974 |
| `SoH` | `float64` | 0 | 168 | 57.120975 | 100.000000 |

**First 5 Sample Rows:**
```text
   Charge Cycle  Cycle #  Dis-chg Ah         SoH
0             2        1    2.051974  100.000000
1             4        2    2.038456   99.341209
2             6        3    2.028602   98.860978
3             8        4    2.016546   98.273463
4            10        5    2.016070   98.250243
```

## 3. Cross-File Verification & Anomaly Findings

1. **Single Consistent Battery Identity (`NASA_B0006`):** All `50,285` discharge time-step records in `Discharge AH.xlsx` belong to a single battery cell tested at a constant `Ambient_Temperature = 24` °C across `168` discharge cycles (spanning NASA experimental operation indices `2` to `614`). No multi-battery concatenation exists.
2. **Cycle Indexing Distinction:**
   - In `Discharge AH.xlsx` (`Cycle_Number`), `Discharge Summary.xlsx` (`Dis-ch Cycle`), and `SoH Data.xlsx` (`Charge Cycle`), the integer index (`2, 4, 6, ..., 614`) is the **experimental operation number** (which increments across charge, discharge, and impedance operations in the NASA test bed).
   - In `SoH Data.xlsx`, `Cycle #` (`1, 2, 3, ..., 168`) is the **sequential discharge cycle index** $t \in \{1, 2, \dots, 168\}$. This sequential index $t$ is the proper time-series index for cycle-based degradation forecasting.
3. **Cross-File Capacity Consistency:** `Discharge Summary.xlsx` (`Dis-chg Ah`) and `SoH Data.xlsx` (`Dis-chg Ah`) contain the exact same 168 discharge cycles (`operation_id_exact_match = True`) with a maximum absolute discrepancy of `7.982197e-04 Ah` (attributable solely to rounding in the first row: `2.051978` vs `2.051974` Ah).
4. **Boundary Artifacts in `Discharge AH.xlsx`:**
   - `Discharge AH.xlsx` computes `delta t` as a simple row-to-row difference without resetting at cycle boundaries (`167` negative `delta t` rows and `613` negative `Discharge Ah` rows, plus `1` missing value at index `[7827]`).
   - The cycle-level capacities in `SoH Data.xlsx` (`Dis-chg Ah`) are clean, positive, and free of missing values (`0` missing, range `1.1721 Ah` to `2.0520 Ah`).
