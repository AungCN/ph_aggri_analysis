"""
Step 1 — Data extraction and cleaning.

Reads the four FAO zip files, filters for the Philippines, and writes
per-crop production CSVs, a population CSV, and per-capita supply CSVs
to data/interim/.
"""

import os
import zipfile
import pandas as pd

os.makedirs("data/interim", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ---------------------------------------------------------------------------
# Confirmed FAO item strings (verified against actual zip contents)
# ---------------------------------------------------------------------------
CROP_ITEMS = {
    "rice":         "Rice",
    "corn":         "Maize (corn)",
    "coconut":      "Coconuts, in shell",
    "sugarcane":    "Sugar cane",
    "banana":       "Bananas",
    "pineapple":    "Pineapples",
    "coffee":       "Coffee, green",
    "cacao":        "Cocoa beans",
    "cassava":      "Cassava, fresh",
    "sweet_potato": "Sweet potatoes",
    "groundnut":    "Groundnuts, excluding shelled",
    "onion":        "Onions and shallots, dry (excluding dehydrated)",
    "garlic":       "Green garlic",    # Philippines uses "Green garlic" not "Garlic"
    "eggplant":     "Eggplants (aubergines)",
    "cabbage":      "Cabbages",
}

# FBS item names differ from production item names.
# New file (2010-2023) uses these labels.
FBS_ITEMS_NEW = {
    "rice":         "Rice and products",
    "corn":         "Maize and products",
    "coconut":      "Coconuts - Incl Copra",
    "sugarcane":    "Sugar cane",
    "banana":       "Bananas",
    "pineapple":    "Pineapples and products",
    "coffee":       "Coffee and products",
    "cacao":        "Cocoa Beans and products",
    "cassava":      "Cassava and products",
    "sweet_potato": "Sweet potatoes",
    "groundnut":    "Groundnuts",
    "onion":        "Onions",
    # garlic, eggplant, cabbage NOT present in FBS — cannot compute consumption
}

# Historic file (1961-2013) uses slightly different names for some crops.
FBS_ITEMS_OLD = {
    "rice":         "Rice (Milled Equivalent)",
    "corn":         "Maize and products",
    "coconut":      "Coconuts - Incl Copra",
    "sugarcane":    "Sugar cane",
    "banana":       "Bananas",
    "pineapple":    "Pineapples and products",
    "coffee":       "Coffee and products",
    "cacao":        "Cocoa Beans and products",
    "cassava":      "Cassava and products",
    "sweet_potato": "Sweet potatoes",
    "groundnut":    "Groundnuts (Shelled Eq)",
    "onion":        "Onions",
    # garlic, eggplant, cabbage NOT present in FBS historic
}


def load_fao_zip(path: str) -> pd.DataFrame:
    """Load the main CSV from a FAO normalized zip file."""
    z = zipfile.ZipFile(path)
    csv_name = max(z.namelist(), key=lambda n: z.getinfo(n).file_size)
    print(f"  loading {csv_name} ...")
    return pd.read_csv(z.open(csv_name), encoding="latin-1", low_memory=False)


# ---------------------------------------------------------------------------
# 1. Production series — one CSV per crop
# ---------------------------------------------------------------------------
print("\n=== Production series ===")
prod = load_fao_zip("data/raw/Production_Crops_Livestock_E_All_Data_(Normalized).zip")

prod_ph = prod[
    (prod["Area"] == "Philippines") &
    (prod["Element"] == "Production")
].copy()

for key, item in CROP_ITEMS.items():
    s = prod_ph[prod_ph["Item"] == item][["Year", "Value"]].copy()
    if s.empty:
        print(f"  WARNING: NO ROWS for {key} ({item!r}) — fix the item name!")
        continue
    s = s.sort_values("Year")
    s["Year"] = pd.to_datetime(s["Year"], format="%Y")
    s = s.set_index("Year")
    s.to_csv(f"data/interim/prod_{key}.csv")
    print(f"  {key}: {s.index.min().year}–{s.index.max().year}, {len(s)} rows")


# ---------------------------------------------------------------------------
# 2. Population series
# ---------------------------------------------------------------------------
print("\n=== Population series ===")
pop = load_fao_zip("data/raw/Population_E_All_Data_(Normalized).zip")

pop_ph = pop[
    (pop["Area"] == "Philippines") &
    (pop["Element"] == "Total Population - Both sexes")
][["Year", "Value"]].copy()

pop_ph = pop_ph.sort_values("Year")
# FAO population is in 1000 persons — multiply by 1000 to get actual headcount
pop_ph["population"] = pop_ph["Value"] * 1000
pop_ph[["Year", "population"]].to_csv("data/interim/population.csv", index=False)

sample_2020 = pop_ph[pop_ph["Year"] == 2020]["population"].values
print(f"  Rows: {len(pop_ph)}, year range: {pop_ph['Year'].min()}–{pop_ph['Year'].max()}")
if len(sample_2020):
    print(f"  Philippines 2020 population: {sample_2020[0]:,.0f} (expected ~110,000,000)")


# ---------------------------------------------------------------------------
# 3. Per-capita food supply (FBS) — stitch historic ≤ 2009 + new ≥ 2010
# ---------------------------------------------------------------------------
print("\n=== Per-capita food supply (FBS) ===")
fbs_new = load_fao_zip("data/raw/FoodBalanceSheets_E_All_Data_(Normalized).zip")
fbs_old = load_fao_zip("data/raw/FoodBalanceSheetsHistoric_E_All_Data_(Normalized).zip")

FBS_ELEMENT = "Food supply quantity (kg/capita/yr)"


def fbs_series(df: pd.DataFrame, item: str) -> pd.DataFrame:
    return df[
        (df["Area"] == "Philippines") &
        (df["Element"] == FBS_ELEMENT) &
        (df["Item"] == item)
    ][["Year", "Value"]].sort_values("Year")


def build_percapita(key: str, item_new: str, item_old: str) -> pd.DataFrame:
    old_part = fbs_series(fbs_old, item_old)
    old_part = old_part[old_part["Year"] <= 2009]

    new_part = fbs_series(fbs_new, item_new)
    new_part = new_part[new_part["Year"] >= 2010]

    stitched = pd.concat([old_part, new_part]).sort_values("Year")
    stitched = stitched.rename(columns={"Value": "kg_per_capita"})
    return stitched


missing_fbs = []
for key in CROP_ITEMS:
    if key not in FBS_ITEMS_NEW:
        missing_fbs.append(key)
        print(f"  SKIP {key}: not in FBS (no per-capita data available)")
        continue

    s = build_percapita(key, FBS_ITEMS_NEW[key], FBS_ITEMS_OLD.get(key, FBS_ITEMS_NEW[key]))
    if s.empty:
        print(f"  WARNING: empty per-capita series for {key}")
        missing_fbs.append(key)
        continue

    out_path = f"data/interim/percapita_{key}.csv"
    s.to_csv(out_path, index=False)
    print(f"  {key}: {s['Year'].min()}–{s['Year'].max()}, {len(s)} rows → {out_path}")

print(f"\nCrops with NO FBS per-capita data: {missing_fbs}")
print("(These crops cannot have surplus/deficit computed — note in replication log)")

print("\n=== Extraction complete. Check data/interim/ for output files. ===")
