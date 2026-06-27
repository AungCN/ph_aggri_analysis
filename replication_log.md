# Replication Log

## 1. Library versions (venv: Python 3.11.15)

| Package | Version |
|---------|---------|
| pycaret | 3.3.2 |
| prophet | 1.3.0 |
| pmdarima | 2.0.4 |
| pandas | 2.1.4 |
| statsmodels | 0.14.6 |
| numpy | 1.26.4 |
| matplotlib | 3.7.5 |

Run: `source .venv/bin/activate && python src/03_model.py`

---

## 2. FAO Element strings used

| Purpose | Element string |
|---------|---------------|
| Production | `Production` |
| Per-capita supply | `Food supply quantity (kg/capita/yr)` |
| Population | `Total Population - Both sexes` |

---

## 3. Exact FAO Item labels

### Production file (`Production_Crops_Livestock_E_All_Data_(Normalized).zip`)

| Crop key | FAO Item label |
|----------|---------------|
| rice | `Rice` |
| corn | `Maize (corn)` |
| coconut | `Coconuts, in shell` |
| sugarcane | `Sugar cane` |
| banana | `Bananas` |
| pineapple | `Pineapples` |
| coffee | `Coffee, green` |
| cacao | `Cocoa beans` |
| cassava | `Cassava, fresh` |
| sweet_potato | `Sweet potatoes` |
| groundnut | `Groundnuts, excluding shelled` |
| onion | `Onions and shallots, dry (excluding dehydrated)` |
| **garlic** | **`Green garlic`** ← paper table says `Garlic`; real FAO label for Philippines is `Green garlic` |
| eggplant | `Eggplants (aubergines)` |
| cabbage | `Cabbages` |

### FBS new file (2010–2023)

| Crop key | FAO Item label |
|----------|---------------|
| rice | `Rice and products` |
| corn | `Maize and products` |
| coconut | `Coconuts - Incl Copra` |
| sugarcane | `Sugar cane` |
| banana | `Bananas` |
| pineapple | `Pineapples and products` |
| coffee | `Coffee and products` |
| cacao | `Cocoa Beans and products` |
| cassava | `Cassava and products` |
| sweet_potato | `Sweet potatoes` |
| groundnut | `Groundnuts` |
| onion | `Onions` |
| garlic | NOT IN FBS |
| eggplant | NOT IN FBS |
| cabbage | NOT IN FBS |

### FBS historic file (1961–2013)

Same as FBS new except:
- rice → `Rice (Milled Equivalent)`
- groundnut → `Groundnuts (Shelled Eq)`

---

## 4. Unit confirmations

- **Per-capita supply**: `Food supply quantity (kg/capita/yr)` — units are **kg/capita/year** ✅
- **Population**: FAO value is in **1,000 persons** → multiplied by 1,000 to get actual headcount
  - Verification: Philippines 2020 = **112,081,264** ✅ (expected ~110M)
- **Production**: in **tonnes** ✅
- **Consumption formula**: `kg_per_capita × population / 1000` → converts to tonnes ✅

---

## 5. Modeling decisions (⚠️ undocumented in paper)

| Setting | Value chosen | Reason |
|---------|-------------|--------|
| `session_id` | 123 | arbitrary fixed seed for reproducibility |
| `fold` | 3 | not stated; 3 is a common default for short annual series |
| `seasonal_period` | 1 | annual data; no within-year seasonality |

---

## 6. Row count

- Production data range: **1961–2024**, **64 rows** per crop
- Paper says "60 instances" in one place and implies ~63 elsewhere (1961–2023)
- Decision: use all 64 rows (1961–2024); the extra 2024 data point improves coverage
- FBS per-capita data: 1961–2023, **63 rows** (no 2024 in balance sheets yet)

---

## 7. FBS methodology seam

- Historic file: 1961–2013
- New file: 2010–2023
- Stitching rule: historic ≤ 2009, new ≥ 2010
- A small jump may appear at year 2010 due to FAO methodology revision — this is expected

---

## 8. Crops missing from FBS

The following crops have **no** `Food supply quantity (kg/capita/yr)` data in either FBS file:
- **garlic** (labeled `Green garlic` in production)
- **eggplant**
- **cabbage**

Consequence: surplus/deficit cannot be computed for these 3 crops.
The paper reports garlic as a deficit crop — it likely used a different consumption source or assumption not documented in the paper.

---

## 9. Auto ARIMA re-tune rule (⚠️ to fill in after running 03_model.py)

The paper says: if test MAPE > 10%, raise max_p / max_q / max_order and re-run.
`03_model.py` implements a 3-pass escalation: (5,5,10) → (8,8,15) → (10,10,20).

Record final params per crop after running:

| Crop | Final (max_p, max_q, max_order) | In-sample MAPE |
|------|--------------------------------|----------------|
| rice | (fill after run) | |
| ... | | |
