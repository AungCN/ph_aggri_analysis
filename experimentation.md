# Experimentation Guide — Replicating the Philippine Food-Security Forecasting Paper

This file is your working manual for VS Code. It rebuilds the paper's pipeline step by step: data processing → EDA → modeling → results.

**Legend used throughout:**
- ✅ = the paper states this clearly. Copy it.
- ⚠️ = the paper is silent or unclear. You must decide and **write your choice in the replication log** (Section 9).

**Goal of replication:** match the paper's two anchor results.
1. Rice production, best model **Auto ARIMA**, MAPE ≈ **0.0543**, MAE ≈ **590,365**, RMSE ≈ **745,660**.
2. Four deficit crops: **garlic, groundnuts, onions, sweet potatoes**, with onions ≈ **−150,000 tonnes in 2024**.

If you hit those two, your replication is on track.

---

## 0. Project setup

### 0.1 Folder structure
Make this layout in VS Code:

```
food-security-forecast/
├── data/
│   ├── raw/          <- put the 4 FAO zip files here
│   ├── interim/      <- per-crop cleaned CSVs land here
│   └── processed/    <- final forecast + deficit CSVs land here
├── notebooks/        <- optional Jupyter exploration
├── src/
│   ├── 01_extract.py
│   ├── 02_eda.py
│   ├── 03_model.py
│   └── 04_deficit.py
├── outputs/          <- plots and metric tables
├── replication_log.md
└── requirements.txt
```

### 0.2 The four input files (put in `data/raw/`)
- `Production_Crops_Livestock_E_All_Data_(Normalized).zip` → production (tonnes)
- `Population_E_All_Data_(Normalized).zip` → population
- `FoodBalanceSheets_E_All_Data_(Normalized).zip` → per-capita supply, 2010–2023
- `FoodBalanceSheetsHistoric_E_All_Data_(Normalized).zip` → per-capita supply, 1961–2013

### 0.3 Environment
Use **Python 3.10 or 3.11** (PyCaret is picky with very new Python). Create a clean virtual environment so versions stay fixed.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install "pycaret[full]" prophet pmdarima pandas numpy matplotlib statsmodels
pip freeze > requirements.txt
```

⚠️ **The single biggest replication risk is library versions.** PyCaret and Prophet change their defaults between releases, which shifts the numbers. After install, record your versions:

```python
import pycaret, prophet, pmdarima, pandas, statsmodels
print(pycaret.__version__, prophet.__version__, pmdarima.__version__,
      pandas.__version__, statsmodels.__version__)
```
Paste the output into `replication_log.md`.

---

## 1. The 15 crops and their FAO item names

This is the **most common place replication breaks.** FAO labels are specific and they changed naming around 2022. Match every crop by hand.

| Paper crop | Likely FAO item name (newer files) | Older alias to watch for |
|---|---|---|
| Rice | `Rice` | `Rice, paddy` |
| Corn | `Maize (corn)` | `Maize` |
| Coconut | `Coconuts, in shell` | `Coconuts` |
| Sugarcane | `Sugar cane` | `Sugar cane` |
| Banana | `Bananas` | `Bananas` |
| Pineapple | `Pineapples` | `Pineapples` |
| Coffee | `Coffee, green` | `Coffee, green` |
| Cacao | `Cocoa beans` | `Cocoa, beans` |
| Cassava | `Cassava, fresh` | `Cassava` |
| Sweet potato | `Sweet potatoes` | `Sweet potatoes` |
| Peanut (groundnut) | `Groundnuts, excluding shelled` | `Groundnuts, with shell` |
| Onion | `Onions and shallots, dry (excluding dehydrated)` | `Onions, dry` |
| Garlic | `Garlic` | `Garlic` |
| Eggplant | `Eggplants (aubergines)` | `Eggplants` |
| Cabbage | `Cabbages` | `Cabbages and other brassicas` |

⚠️ **Do not trust this table blindly.** Run the check in Section 2.3 to print the real labels in *your* download, then lock the exact strings into a dictionary.

---

## 2. Data extraction and cleaning (`01_extract.py`)

### 2.1 What each normalized file looks like
Every `(Normalized)` zip holds one big CSV covering **all countries**. The columns you care about are:

`Area`, `Item`, `Element`, `Year`, `Unit`, `Value`

Your job: keep only **Philippines**, the **right element**, and the **15 crops**.

### 2.2 Load a zip into pandas

```python
import pandas as pd, zipfile

def load_fao_zip(path):
    z = zipfile.ZipFile(path)
    # the main CSV is usually the largest file in the zip
    csv_name = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)
    # FAO files use latin-1 encoding, not utf-8
    return pd.read_csv(z.open(csv_name), encoding="latin-1")
```

### 2.3 Inspect before filtering (do this first!)

```python
prod = load_fao_zip("data/raw/Production_Crops_Livestock_E_All_Data_(Normalized).zip")

ph = prod[prod["Area"] == "Philippines"]
print(sorted(ph["Element"].unique()))   # find the exact "Production" label
print(sorted(ph["Item"].unique()))      # find the exact 15 crop labels
```

Copy the real strings into this dictionary (edit to match what you printed):

```python
CROP_ITEMS = {
    "rice":        "Rice",
    "corn":        "Maize (corn)",
    "coconut":     "Coconuts, in shell",
    "sugarcane":   "Sugar cane",
    "banana":      "Bananas",
    "pineapple":   "Pineapples",
    "coffee":      "Coffee, green",
    "cacao":       "Cocoa beans",
    "cassava":     "Cassava, fresh",
    "sweet_potato":"Sweet potatoes",
    "groundnut":   "Groundnuts, excluding shelled",
    "onion":       "Onions and shallots, dry (excluding dehydrated)",
    "garlic":      "Garlic",
    "eggplant":    "Eggplants (aubergines)",
    "cabbage":     "Cabbages",
}
```

### 2.4 Build clean per-crop production series

The paper's prep is exact (✅): keep only Year and Value, convert Year to datetime.

```python
import os
os.makedirs("data/interim", exist_ok=True)

prod_ph = prod[(prod["Area"] == "Philippines") &
               (prod["Element"] == "Production")]      # unit = tonnes

for key, item in CROP_ITEMS.items():
    s = prod_ph[prod_ph["Item"] == item][["Year", "Value"]].copy()
    if s.empty:
        print(f"⚠️  NO ROWS for {key} ({item}) — fix the item name!")
        continue
    s = s.sort_values("Year")
    s["Year"] = pd.to_datetime(s["Year"], format="%Y")
    s = s.set_index("Year")
    s.to_csv(f"data/interim/prod_{key}.csv")
    print(f"{key}: {s.index.min().year}-{s.index.max().year}, {len(s)} rows")
```

⚠️ **Row-count check.** 1961–2023 = **63 rows**. The paper variously says "60 instances" and "univariate / 7 features," which contradict each other. Use the real count (likely 63) and note the discrepancy in your log. Do not trim to 60 unless you discover a documented reason.

### 2.5 Population series

```python
pop = load_fao_zip("data/raw/Population_E_All_Data_(Normalized).zip")
pop_ph = pop[(pop["Area"] == "Philippines") &
             (pop["Element"] == "Total Population - Both sexes")][["Year","Value"]]
# ⚠️ FAO population Value is in 1000 persons. Multiply by 1000 for real headcount.
pop_ph = pop_ph.sort_values("Year")
pop_ph["population"] = pop_ph["Value"] * 1000
pop_ph[["Year","population"]].to_csv("data/interim/population.csv", index=False)
```

⚠️ **Unit trap:** FAO population is usually in **thousands**. If you forget the ×1000, your consumption and deficit numbers will be 1000× too small. Verify: Philippine population in 2020 should be roughly **110 million**, not 110,000.

### 2.6 Per-capita supply (the tricky one — two files, one methodology break)

```python
fbs_new = load_fao_zip("data/raw/FoodBalanceSheets_E_All_Data_(Normalized).zip")
fbs_old = load_fao_zip("data/raw/FoodBalanceSheetsHistoric_E_All_Data_(Normalized).zip")

ELEMENT = "Food supply quantity (kg/capita/yr)"

def fbs_series(df, item):
    out = df[(df["Area"] == "Philippines") &
             (df["Element"] == ELEMENT) &
             (df["Item"] == item)][["Year","Value"]]
    return out.sort_values("Year")
```

⚠️ **Methodology break at 2010.** The historic file runs ~1961–2013; the new file runs 2010–2023. Stitch them: use **historic for years ≤ 2009**, **new for years ≥ 2010**. Expect a small jump at the seam — that is FAO's artifact, not your bug. Note it in the log.

⚠️ **FBS item names differ from production item names.** In FBS, rice may appear as `Rice and products`, maize as `Maize and products`, etc. Re-inspect `fbs_new[...]["Item"].unique()` and build a second dictionary `FBS_ITEMS`. Not every one of the 15 crops exists in FBS (e.g. cut flowers or minor crops may be missing) — that limits which crops you can compute consumption for. The paper only reports deficits for 4 crops, so full FBS coverage is not required for the headline result.

```python
def percapita(item_new, item_old):
    a = fbs_series(fbs_old, item_old)
    a = a[a["Year"] <= 2009]
    b = fbs_series(fbs_new, item_new)
    b = b[b["Year"] >= 2010]
    s = pd.concat([a, b]).sort_values("Year")
    s = s.rename(columns={"Value":"kg_per_capita"})
    return s
```

---

## 3. Exploratory Data Analysis (`02_eda.py`) ✅

The paper does three EDA steps per crop: **decomposition**, **differencing**, **ADF stationarity test**.

```python
from pycaret.time_series import TSForecastingExperiment
import pandas as pd

df = pd.read_csv("data/interim/prod_rice.csv",
                 index_col="Year", parse_dates=True)
df.index = pd.PeriodIndex(df.index, freq="Y")   # PyCaret prefers PeriodIndex

exp = TSForecastingExperiment()
exp.setup(
    data = df,
    target = "Value",
    fh = 20,            # ✅ 20-year forecast horizon
    fold = 3,           # ⚠️ paper does not state folds — pick 3, record it
    seasonal_period = 1,# ⚠️ annual data → no within-year season
    session_id = 123,   # ⚠️ paper gives no seed — pick one, record it
)

exp.check_stats()                       # includes the ADF test
exp.plot_model(plot="decomp")           # trend / seasonal / residual
exp.plot_model(plot="diff",
               data_kwargs={"order_list":[1]})  # first differencing
exp.plot_model(plot="acf")
exp.plot_model(plot="pacf")
```

**What you should see (matches the paper):**
- Decomposition shows a clear **uptrend**, **no real seasonality** (annual data can't have it).
- ADF on the raw series → likely **non-stationary**; after one difference → closer to stationary.

⚠️ Because the data is annual, set `seasonal_period = 1`. Do not force a 12-month season; the paper itself finds no seasonality.

---

## 4. Train/test split ✅

The paper uses **80% train / 20% test**, taken from the **end** of the series (standard for time series — you never shuffle time).

With 63 rows: roughly **50 train / 13 test**. PyCaret handles this through `fh` and folds automatically. Ignore the stray "70%" line in the paper; every concrete step uses 80/20.

---

## 5. Modeling (`03_model.py`)

The paper runs **two tracks in parallel.** Replicate both.

### 5.1 Track A — PyCaret "compare everything" (gives the per-crop winner table)

```python
best = exp.compare_models(sort="MAPE")     # ranks ALL models by MAPE
results_table = exp.pull()                 # the metrics dataframe
results_table.to_csv("outputs/rice_model_comparison.csv")

final = exp.finalize_model(best)           # refit on all data
fcst  = exp.predict_model(final, fh=20)    # 20-year forecast
fcst.to_csv("data/processed/rice_forecast.csv")
```

This is where **Naive, Huber, AdaBoost, Extra Trees, Gradient Boosting, Theta, KNN** etc. all come from — PyCaret includes them by default, which explains the models the paper mentions in Results but never lists in Methods.

### 5.2 Track B — manual Auto ARIMA + Prophet (the paper's explicit recipes)

**Auto ARIMA** (9-step recipe ✅):

```python
from pmdarima import auto_arima
import numpy as np

y = df["Value"].astype(float).values

model = auto_arima(
    y,
    seasonal=False,        # annual data
    max_p=5, max_q=5,      # ✅ paper tunes these if MAPE > 10%
    max_order=10,
    stepwise=True,
    error_action="ignore",
    suppress_warnings=True,
)
print(model.summary())     # gives the chosen (p,d,q)
```

**The paper's decision rule (✅):** if test MAPE > 10% (0.1), raise `max_p`, `max_q`, `max_order` and re-run. Apply it.

**Prophet** (4-step recipe ✅):

```python
from prophet import Prophet

pdf = (pd.read_csv("data/interim/prod_rice.csv")
         .rename(columns={"Year":"ds","Value":"y"}))
pdf["ds"] = pd.to_datetime(pdf["ds"])

m = Prophet()              # ✅ paper notes Prophet needs little tuning
m.fit(pdf)
future = m.make_future_dataframe(periods=20, freq="YS")
prophet_fcst = m.predict(future)
```

### 5.3 Loop over all 15 crops
Wrap Section 3–5 in a function and loop `CROP_ITEMS.keys()`. Save one comparison CSV and one forecast CSV per crop. Keep a master dictionary `{crop: best_model_name, best_mape}`.

---

## 6. Evaluation metrics ✅

Use the metrics the paper quotes: **MASE, RMSSE, MAE, RMSE, MAPE, SMAPE** (R² is in the abstract too). PyCaret reports these automatically in `exp.pull()`.

**Direction rule (the paper words this confusingly — get it right):**
- **Lower is better:** MASE, RMSSE, MAE, RMSE, MAPE, SMAPE.
- **Higher is better:** R².
- So "most accurate" = **lowest MAPE**, never "highest MAPE."

**Anchor check — rice:** your best model should be **Auto ARIMA** with MAPE ≈ **0.054**, MAE ≈ **590,365**, RMSE ≈ **745,660**. If you are far off, the cause is almost always (a) wrong FAO element, (b) different library version, or (c) wrong item label.

---

## 7. Deficit / surplus calculation (`04_deficit.py`) ✅

This is the paper's policy output. Formulas are given exactly:

```
Estimated Consumption = per_capita_kg × population
Surplus / Deficit      = Forecasted Production − Estimated Consumption
```

```python
import pandas as pd

pop  = pd.read_csv("data/interim/population.csv")          # Year, population
pc   = percapita("Rice and products", "Rice and products") # kg per capita
prod_fcst = pd.read_csv("data/processed/rice_forecast.csv")

# align years, then:
# Estimated consumption in tonnes = (kg_per_capita * population) / 1000
#   because kg → tonnes is ÷1000
merged["est_consumption_tonnes"] = (
    merged["kg_per_capita"] * merged["population"] / 1000
)
merged["surplus_deficit"] = (
    merged["forecast_production"] - merged["est_consumption_tonnes"]
)
merged.to_csv("data/processed/rice_deficit.csv", index=False)
```

⚠️ **Unit consistency is everything here.** Production is in **tonnes**, per-capita is in **kg**, population is in **persons**. kg × persons = kg, then ÷1000 → tonnes so it matches production. Get one unit wrong and the deficit is off by 1000×.

**Anchor check — onions:** the model should show a **deficit near −150,000 tonnes in 2024**. The full set of deficit crops should be **garlic, groundnuts, onions, sweet potatoes** (negative), with the other 11 in surplus.

---

## 8. Putting it together — run order

```bash
python src/01_extract.py    # builds data/interim/*.csv
python src/02_eda.py        # saves EDA plots to outputs/
python src/03_model.py      # loops 15 crops, saves comparison + forecast CSVs
python src/04_deficit.py    # builds deficit/surplus per crop
```

Final deliverables to compare against the paper:
1. A 15-row table: **crop | best model | MAPE** → compare to the paper's claim that Auto ARIMA wins most production series.
2. A deficit table → should isolate the **same 4 deficit crops**.

---

## 9. Replication log (fill this in as you go)

Keep `replication_log.md` and record every ⚠️ decision. These are the things the paper left undocumented, so future-you (and any reviewer) needs them written down:

1. Library versions: PyCaret ___, Prophet ___, pmdarima ___, pandas ___, statsmodels ___
2. Exact FAO **Element** strings used (production / per-capita / population)
3. Exact **Item** label for each of the 15 crops (production and FBS may differ)
4. Per-capita unit confirmed = **kg/capita/yr**; population unit confirmed = **persons** (after ×1000)
5. `session_id` (seed) = ___ ; `fold` count = ___ ; `seasonal_period` = 1
6. Row count per crop (expected 63 for 1961–2023); how you handled the "60 vs 63" discrepancy
7. FBS methodology seam: historic ≤ 2009, new ≥ 2010; note any jump at 2010
8. Any crop missing from FBS (so consumption couldn't be computed)
9. Auto ARIMA: did any crop trigger the MAPE > 10% re-tune rule? Record the final max_p/max_q/max_order.

---

## 10. Known pitfalls (quick reference)

| Symptom | Likely cause | Fix |
|---|---|---|
| `NO ROWS for <crop>` | item label mismatch | print `Item.unique()`, copy exact string |
| Deficit numbers 1000× too small/large | unit error (population in 1000s, or kg vs tonnes) | re-check Section 2.5 and 7 |
| Rice MAPE far from 0.054 | wrong FAO element, or library version | use Element = "Production", pin versions |
| Encoding error on `read_csv` | FAO uses latin-1 | `encoding="latin-1"` |
| PyCaret setup fails on index | needs PeriodIndex | `pd.PeriodIndex(idx, freq="Y")` |
| Per-capita has a jump near 2010 | FBS methodology break | expected; document it |
| Model forces seasonality | seasonal_period not set | set `seasonal_period = 1` |

---

**Bottom line:** lock the item labels and units first (Sections 1–2), confirm the rice and onion anchors (Sections 6–7), and write every ⚠️ choice into your log. Once the two anchors match, you have faithfully reproduced the paper — and you'll also have the clean data you need later to fix the issues we discussed.
