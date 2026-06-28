# Philippine Food Security Forecasting

A replication and extension of an FAO-based food production forecasting study for the Philippines. Covers **15 crops** from **1961–2024** and projects production, consumption, and surplus/deficit through **2044** using three time-series modelling approaches.

**Live dashboard →** [aggracultural-ph.streamlit.app](https://aggracultural-ph.streamlit.app)

---

## Overview

This project replicates a published study that used FAO FAOSTAT data to forecast Philippine agricultural production and assess food security. The pipeline extracts raw FAO data, runs exploratory analysis, trains and compares forecasting models, then computes per-crop surplus/deficit against estimated population consumption.

### Crops Analysed

| Category | Crops |
|----------|-------|
| Staples | Rice, Corn, Cassava, Sweet Potato |
| Export | Coconut, Sugarcane, Banana, Pineapple |
| Beverages | Coffee, Cacao |
| Vegetables | Onion, Garlic, Eggplant, Cabbage |
| Legumes | Groundnut |

> Garlic, Eggplant, and Cabbage have no FAO Food Balance Sheet per-capita supply data, so surplus/deficit cannot be computed for them.

---

## Key Findings

| Result | Paper | This Replication |
|--------|-------|-----------------|
| Confirmed deficit crops | Garlic, Groundnuts, Onions, Sweet Potatoes | Groundnuts ✅, Onions ✅, Sweet Potatoes ✅ |
| Additional deficits found | — | Rice, Coffee, Cacao |
| Garlic | Deficit | No FBS data — unverifiable |

Rice, Coffee, and Cacao appear in deficit in this replication because FAO Food Balance Sheet per-capita figures include imports, which inflates apparent consumption beyond domestic production.

---

## Project Structure

```
.
├── src/
│   ├── 01_extract.py       # Step 1: Extract & clean FAO zip files
│   ├── 02_eda.py           # Step 2: Decomposition, ACF/PACF, ADF test
│   ├── 03_model.py         # Step 3: PyCaret + Auto ARIMA + Prophet
│   └── 04_deficit.py       # Step 4: Surplus/deficit computation
├── data/
│   ├── raw/                # Original FAO zip files (FAOSTAT)
│   ├── interim/            # Per-crop production & per-capita CSVs
│   └── processed/          # Forecast & deficit CSVs per crop
├── outputs/
│   ├── eda/                # HTML plots (decomp, diff, ACF, PACF)
│   ├── model_summary.csv   # Best model + MAPE per crop
│   ├── deficit_summary.csv # Surplus/deficit status per crop
│   └── *_comparison.csv    # Full model ranking per crop
├── dashboard.py            # Streamlit dashboard (6 pages)
├── requirements.txt        # Dashboard dependencies
└── replication_log.md      # Detailed methodology notes
```

---

## Pipeline

Run the four scripts in order from the project root:

```bash
python src/01_extract.py   # ~10 seconds
python src/02_eda.py       # ~2 minutes
python src/03_model.py     # ~30–60 minutes (PyCaret trains many models)
python src/04_deficit.py   # ~10 seconds
```

> **Local setup requires the full ML environment** (PyCaret, Prophet, pmdarima). See [Local Development](#local-development) below.

### What each script does

| Script | Input | Output |
|--------|-------|--------|
| `01_extract.py` | `data/raw/*.zip` | `data/interim/prod_*.csv`, `percapita_*.csv`, `population.csv` |
| `02_eda.py` | `data/interim/prod_*.csv` | `outputs/eda/*.html` (4 plots × 15 crops) |
| `03_model.py` | `data/interim/prod_*.csv` | `data/processed/*_forecast.csv`, `*_arima_forecast.csv`, `*_prophet_forecast.csv`, `outputs/model_summary.csv`, `outputs/*_comparison.csv` |
| `04_deficit.py` | processed forecasts + interim data | `data/processed/*_deficit.csv`, `outputs/deficit_summary.csv` |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Overview | KPI cards, surplus/deficit bar chart, status table |
| Production History | Historical trends 1961–2024, multi-crop comparison |
| EDA | Decomposition, differencing, ACF, PACF per crop |
| Model Comparison | MAPE heatmap across all models and crops |
| Forecast Explorer | Interactive 20-year forecast — PyCaret vs ARIMA vs Prophet |
| Food Security | Surplus/deficit treemap, 20-year trajectories, per-capita trends |

---

## Data Sources

All data is from [FAO FAOSTAT](https://www.fao.org/faostat/en/#data):

| File | Content |
|------|---------|
| `Production_Crops_Livestock_E_All_Data_(Normalized).zip` | Annual crop production in tonnes (1961–2024) |
| `FoodBalanceSheets_E_All_Data_(Normalized).zip` | Per-capita food supply kg/capita/yr (2010–2023) |
| `FoodBalanceSheetsHistoric_E_All_Data_(Normalized).zip` | Per-capita food supply kg/capita/yr (1961–2013) |
| `Population_E_All_Data_(Normalized).zip` | Philippine population (1961–2044 projection) |

The two Food Balance Sheet files are stitched at 2009→2010 (historic ≤ 2009, new ≥ 2010).

---

## Modelling Approach

### Track A — PyCaret `compare_models()`
Trains and ranks ~20 statistical and ML models using cross-validation. Settings: `fold=3`, `fh=20`, `seasonal_period=1` (annual data). The best model by MAPE is used for the primary 20-year forecast.

### Track B — Auto ARIMA
Uses `pmdarima.auto_arima()` with a 3-pass MAPE escalation:
- Pass 1: `(max_p=5, max_q=5, max_order=10)`
- Pass 2: `(max_p=8, max_q=8, max_order=15)` if MAPE > 10%
- Pass 3: `(max_p=10, max_q=10, max_order=20)` if still > 10%

Outputs 20-year forecast with 95% confidence intervals.

### Track C — Facebook Prophet
Fits Prophet on annual data with `yearly_seasonality=False`. Outputs 20-year forecast with 95% uncertainty intervals.

### Food Security Computation
```
Estimated Consumption (t) = kg_per_capita × population / 1000
Surplus/Deficit (t)       = Forecasted Production − Estimated Consumption
```
Per-capita figures are taken as the 5-year average (2019–2023) from the FAO Food Balance Sheet and held constant over the forecast horizon.

---

## Local Development

### Prerequisites
- Python 3.11 (PyCaret 3.3.x requires Python ≤ 3.11)
- The four FAO zip files placed in `data/raw/`

### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt --prefer-binary

# Additional packages needed to run the pipeline scripts (not the dashboard):
pip install pycaret[time_series]>=3.3,<3.4 pmdarima>=2.0 prophet>=1.1
```

### Run the dashboard locally

```bash
source .venv/bin/activate
streamlit run dashboard.py
```

---

## Replication Notes

See [replication_log.md](replication_log.md) for full methodology details including:
- Exact FAO item label strings used per crop
- Unit confirmations (population ×1000, production in tonnes)
- Modelling decisions not documented in the original paper
- FBS data seam handling at 2009→2010
