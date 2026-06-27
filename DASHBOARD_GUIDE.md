# Interactive Dashboard Guide
## Philippine Food-Security Forecasting — Streamlit + Plotly

---

## Quick Start

```bash
# from the project root
source .venv/bin/activate
streamlit run dashboard.py
```

Then open `http://localhost:8501` in your browser.

---

## Architecture Overview

```
dashboard.py              ← single-file Streamlit app
│
├── Sidebar navigation    ← st.radio (6 pages)
│
├── Page 1 🏠 Overview
├── Page 2 📈 Production History
├── Page 3 🔬 EDA
├── Page 4 🤖 Model Comparison
├── Page 5 🔮 Forecast Explorer
└── Page 6 🌾 Food Security
```

All data is read from files already produced by `src/01_extract.py` through `src/04_deficit.py`.
Nothing is recomputed inside the dashboard — it is purely a visualisation layer.

---

## Data Flow Into the Dashboard

```
data/interim/prod_*.csv          → Page 2 (production history)
                                   Page 5 (historical baseline in forecast chart)

data/interim/percapita_*.csv     → Page 6 Tab 3 (per-capita trends)

data/interim/population.csv      → Page 6 Tab 3 (population chart)
                                   Page 6 Tab 2 (consumption calculation display)

outputs/eda/*.html               → Page 3 (embedded Plotly HTML)

outputs/model_summary.csv        → Page 4 (best model bar chart)
outputs/<crop>_comparison.csv    → Page 4 (MAPE heatmap + detail table)

data/processed/<crop>_forecast.csv       → Page 5 (PyCaret line)
data/processed/<crop>_arima_forecast.csv → Page 5 (ARIMA line + CI band)
data/processed/<crop>_prophet_forecast.csv→ Page 5 (Prophet line + CI band)

data/processed/<crop>_deficit.csv → Page 6 Tab 2 (deep-dive)
outputs/deficit_summary.csv       → Page 1 (KPIs), Page 6 Tab 1 (treemap)
```

---

## Page-by-Page Design

---

### Page 1 — Overview 🏠

**Purpose:** Give the reader the headline story in 30 seconds.

**Components:**

| Component | Library | What it shows |
|-----------|---------|---------------|
| 5 KPI metric cards | `st.metric` | Crops analysed, deficit count, surplus count, worst deficit, best surplus |
| Horizontal bar chart | `plotly.graph_objects.Bar` | All 12 computable crops sorted by surplus/deficit magnitude; red = deficit, green = surplus |
| Status table | `st.dataframe` with row colouring | Crop, value in tonnes, DEFICIT/SURPLUS label |
| Info box | `st.info` | Explains garlic/eggplant/cabbage exclusion |
| Two-column finding summary | `st.markdown` | Paper vs replication comparison |

**Design choice — horizontal bar chart:**  
Sorting crops from most deficit (left/bottom) to highest surplus makes the severity ranking immediately readable. A vertical bar would overflow the x-axis labels at 12 crops.

---

### Page 2 — Production History 📈

**Purpose:** Explore raw production trends for any combination of crops.

**Components:**

| Component | Library | What it shows |
|-----------|---------|---------------|
| Multi-select crop picker | `st.multiselect` | Choose 1–15 crops |
| Normalize toggle | `st.checkbox` | Re-index all series to 1961 = 100 so growth rates are comparable regardless of scale differences |
| Line chart | `plotly.graph_objects.Scatter` | Historical production 1961–2024 |
| Summary stats table | `st.dataframe` | First value, last value, growth multiplier, peak and peak year |

**Design choice — normalize option:**  
Without normalization, sugarcane (24M t) swamps coconut (11M t) swamps onion (259K t). With the index, all 15 crops are on the same 0–600 scale and their relative growth stories become visible.

---

### Page 3 — EDA 🔬

**Purpose:** Inspect the statistical properties of each crop's production series.

**Components:**

| Component | Library | What it shows |
|-----------|---------|---------------|
| ADF result banner | `st.success` / `st.warning` | Stationary or not, with p-value |
| 4 tabs | `st.tabs` | Decomposition · Differencing · ACF · PACF |
| Embedded Plotly HTML | `streamlit.components.v1.html` | The interactive PyCaret plots produced by `02_eda.py` |

**Design choice — embed HTML, not re-render:**  
PyCaret's `plot_model` already produced publication-quality interactive Plotly charts and saved them as HTML. Embedding them directly (`components.html`) avoids re-running the expensive PyCaret setup and keeps the dashboard startup fast.

**ADF p-value hardcoded:**  
The ADF results are stored as a dictionary inside the dashboard rather than re-running PyCaret (which would require a 10–30 s setup per crop). If you want live recalculation, replace the dictionary with a cached function that reads from a saved stats CSV.

---

### Page 4 — Model Comparison 🤖

**Purpose:** Show which model wins for each crop and how all models rank.

**Components:**

| Component | Library | What it shows |
|-----------|---------|---------------|
| Best-model bar chart | `plotly.graph_objects.Bar` | One bar per crop, coloured by MAPE severity; model name printed on bar |
| 10% threshold line | `fig.add_hline` | The paper's re-tune rule: MAPE > 10% → raise ARIMA limits |
| MAPE heatmap | `plotly.express.imshow` | Crops (columns) × top-10 models (rows); red = high MAPE, green = low |
| Detailed rank table | `st.dataframe` | Full metric table for selected crop with conditional colour gradient |

**Design choice — top-10 model heatmap:**  
All 24 models in a heatmap is too dense. Filtering to the 10 lowest-median-MAPE models keeps the heatmap readable and focuses on the realistic candidates.

**Colour scale `RdYlGn_r`:**  
Reversed so **green = low MAPE = good** is intuitive (standard in error metric visualisations).

---

### Page 5 — Forecast Explorer 🔮

**Purpose:** Compare the three modelling approaches on a single crop and inspect confidence intervals.

**Components:**

| Component | Library | What it shows |
|-----------|---------|---------------|
| Crop selector | `st.selectbox` | All 15 crops |
| Three toggle checkboxes | `st.checkbox` | Show/hide PyCaret, ARIMA, Prophet independently |
| Unified line chart | `plotly.graph_objects` | Historical (solid blue) + up to 3 forecast lines |
| ARIMA CI band | `go.Scatter` fill `tonexty` | Shaded 95% confidence interval in orange |
| Prophet CI band | `go.Scatter` fill `tonexty` | Shaded 95% confidence interval in blue-green |
| Vertical split line | `fig.add_vline` | Marks the boundary between historical and forecast |
| Collapsible data table | `st.expander` | All forecast numbers for all three methods |

**Design choice — `hovermode="x unified"`:**  
When you hover over any year, all three model values appear in a single tooltip. This makes it easy to compare the three forecasts at a glance without clicking.

**Confidence intervals:**  
Only ARIMA and Prophet produce CIs. PyCaret `compare_models` returns a point forecast only (the best model from the cross-validated ranking). ARIMA CIs come from `pmdarima` and widen with the horizon. Prophet CIs come from its built-in uncertainty sampling.

---

### Page 6 — Food Security 🌾

**Purpose:** The policy output — which crops will run short and by how much over 20 years.

Three tabs:

#### Tab 1 — All Crops Overview

| Component | Library | What it shows |
|-----------|---------|---------------|
| Treemap | `plotly.express.treemap` | All 12 crops grouped by DEFICIT/surplus status; size = magnitude, colour = direction |
| Small multiples (4×3 grid) | `st.columns` + `go.Bar` | One mini bar chart per crop showing the full 20-year surplus/deficit trajectory |

**Design choice — treemap:**  
A treemap simultaneously encodes status (grouping), magnitude (area), and direction (colour) in one compact view. It makes sugarcane's massive surplus and onion's small deficit both visible in proportion.

**Design choice — small multiples:**  
12 crops on one chart is unreadable. 12 small identical charts let you scan across all crops quickly and spot patterns (e.g. which deficits are growing vs shrinking over the 20-year horizon).

#### Tab 2 — Crop Deep-Dive

| Component | Library | What it shows |
|-----------|---------|---------------|
| 4 KPI cards | `st.metric` | Status 2025, status 2044, number of deficit years, per-capita used |
| Stacked area chart | `go.Scatter` fill `tozeroy` | Production forecast vs estimated consumption overlaid — the gap between them is the deficit/surplus |
| Annual surplus/deficit bar chart | `go.Bar` | Year-by-year surplus (green) / deficit (red) |
| Full data table | `st.expander` + `st.dataframe` | All 20 years with all intermediate calculation columns |

**Design choice — stacked area for production vs consumption:**  
The gap between the two filled areas is visually the surplus or deficit. When consumption (red area) exceeds production (green area), the red bleeds through — intuitively signalling a shortage without needing a subtitle.

#### Tab 3 — Per-Capita Trends

| Component | Library | What it shows |
|-----------|---------|---------------|
| Multi-select crop picker | `st.multiselect` | Compare per-capita supply across crops |
| Per-capita line chart | `go.Scatter` | FBS kg/capita/yr from 1961 to 2023; seam at 2009→2010 annotated |
| Population area chart | `go.Scatter` fill `tozeroy` | FAO population projection 1961–2044; split line at 2024 |

---

## Visualisation Library Choices

| Library | Role | Why |
|---------|------|-----|
| **Plotly** | All charts | Already installed via PyCaret; fully interactive (hover, zoom, pan); renders natively in Streamlit via `st.plotly_chart` |
| **Streamlit components** | Embed PyCaret HTML | Lets us reuse the already-produced EDA plots without re-running PyCaret |
| **Streamlit native** | Metrics, tables, layout | `st.metric`, `st.dataframe`, `st.tabs`, `st.columns` handle layout without extra dependencies |

**Alternatives considered:**

| Library | Why not chosen |
|---------|---------------|
| Altair / Vega-Lite | Excellent for declarative charts, but less flexible for CI band fills and multi-trace overlays |
| Matplotlib / Seaborn | Static; poor Streamlit integration; no hover |
| Bokeh | Interactive but heavier; Plotly is simpler for this use case |
| Dash (Plotly) | More control but requires defining callbacks; Streamlit's reactive model is faster to build |

---

## Caching Strategy

Every data loader is decorated with `@st.cache_data`:

```python
@st.cache_data
def load_production(crop):
    df = pd.read_csv(f"data/interim/prod_{crop}.csv", ...)
    return df
```

This means:
- First call reads from disk.
- Subsequent calls (same arguments) return the cached result instantly.
- Switching crops or pages does not re-read files.
- Cache is invalidated only when the underlying file changes.

---

## Extending the Dashboard

| Addition | Where to add | How |
|----------|-------------|-----|
| Add a new crop | All pages auto-pick from `CROPS` list | Update `CROPS` and `FBS_CROPS` constants at the top |
| Add a new forecast method | Page 5 | Add a new checkbox + `go.Scatter` trace |
| Show 2024 actuals alongside forecast | Page 5 | Extend `load_production` to include 2024 and filter `fcst` to start from 2025 |
| Export chart as PNG | Any page | Plotly toolbar (camera icon) is built-in |
| Export data as CSV | Any table | Add `st.download_button` wrapping `df.to_csv()` |
| Live ADF results | Page 3 | Replace hardcoded dict with a cached CSV saved by `02_eda.py` |

---

## File Summary

```
dashboard.py          ← run this with streamlit
DASHBOARD_GUIDE.md    ← this guide

data/
  interim/            ← all inputs for dashboard
  processed/          ← all inputs for dashboard

outputs/
  eda/                ← HTML plots embedded in Page 3
  model_summary.csv   ← drives Page 4 bar chart
  *_comparison.csv    ← drives Page 4 heatmap and detail table
  deficit_summary.csv ← drives Page 1 and Page 6 Tab 1
```
