"""
Step 4 — Deficit / surplus calculation.

Formula (✅ from paper):
  Estimated Consumption (tonnes) = kg_per_capita × population / 1000
  Surplus / Deficit (tonnes)     = Forecasted Production − Estimated Consumption

Outputs:
  data/processed/<crop>_deficit.csv  — year, forecast, consumption, surplus_deficit
  outputs/deficit_summary.csv        — crop | 2024 deficit | status (surplus/deficit)
"""

import os
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Crops for which we have FBS per-capita data
FBS_CROPS = [
    "rice", "corn", "coconut", "sugarcane", "banana",
    "pineapple", "coffee", "cacao", "cassava", "sweet_potato",
    "groundnut", "onion",
    # garlic, eggplant, cabbage: NOT in FBS — no deficit calculable
]

# Population projections: FAO data extends to 2100
pop_df = pd.read_csv("data/interim/population.csv")
pop_df = pop_df.set_index("Year")


def get_population(year: int) -> float:
    if year in pop_df.index:
        return float(pop_df.loc[year, "population"])
    # fallback: linear interpolation for years not in table
    return float(pop_df["population"].reindex(
        pop_df.index.tolist() + [year]
    ).interpolate().loc[year])


summary_rows = []

for crop in FBS_CROPS:
    forecast_path = f"data/processed/{crop}_forecast.csv"
    percapita_path = f"data/interim/percapita_{crop}.csv"

    if not os.path.exists(forecast_path):
        print(f"SKIP {crop}: no forecast file (run 03_model.py first)")
        continue
    if not os.path.exists(percapita_path):
        print(f"SKIP {crop}: no per-capita file")
        continue

    # Load PyCaret forecast
    fcst = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
    # PyCaret forecast column is typically "y_pred"
    pred_col = [c for c in fcst.columns if "pred" in c.lower() or "yhat" in c.lower()]
    if pred_col:
        fcst = fcst.rename(columns={pred_col[0]: "forecast_production"})
    else:
        fcst.columns = ["forecast_production"] + list(fcst.columns[1:])

    # Keep only future years (forecast horizon)
    fcst.index = pd.PeriodIndex(fcst.index, freq="Y")
    fcst = fcst[fcst.index.year > 2024]    # adjust if your data ends earlier

    if fcst.empty:
        # Try to find forecast years in the file
        fcst = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
        pred_col = [c for c in fcst.columns if "pred" in c.lower() or "yhat" in c.lower()]
        if pred_col:
            fcst = fcst.rename(columns={pred_col[0]: "forecast_production"})
        else:
            fcst.columns = ["forecast_production"] + list(fcst.columns[1:])
        fcst.index = pd.PeriodIndex(fcst.index, freq="Y")

    # Per-capita food supply (historical mean used for forecast years — ✅ paper approach)
    pc = pd.read_csv(percapita_path)
    # Use the last 5 years average as the per-capita value for future years
    recent_pc = pc["kg_per_capita"].tail(5).mean()

    rows = []
    for period in fcst.index:
        year = int(period.year)
        prod_val = float(fcst.loc[period, "forecast_production"])
        pop_val = get_population(year)
        # kg × persons → kg, then ÷1000 → tonnes
        consumption_tonnes = recent_pc * pop_val / 1000
        surplus_deficit = prod_val - consumption_tonnes
        rows.append({
            "year": year,
            "forecast_production_tonnes": prod_val,
            "population": pop_val,
            "kg_per_capita": recent_pc,
            "est_consumption_tonnes": consumption_tonnes,
            "surplus_deficit_tonnes": surplus_deficit,
        })

    deficit_df = pd.DataFrame(rows)
    deficit_df.to_csv(f"data/processed/{crop}_deficit.csv", index=False)

    val_2024 = deficit_df[deficit_df["year"] == 2024]
    if not val_2024.empty:
        sd = float(val_2024["surplus_deficit_tonnes"].iloc[0])
        status = "DEFICIT" if sd < 0 else "surplus"
        print(f"  {crop:15s}: 2024 surplus/deficit = {sd:+,.0f} tonnes  [{status}]")
        summary_rows.append({"crop": crop, "surplus_deficit_2024": sd, "status": status})
    else:
        first = deficit_df.iloc[0]
        sd = float(first["surplus_deficit_tonnes"])
        year = int(first["year"])
        status = "DEFICIT" if sd < 0 else "surplus"
        print(f"  {crop:15s}: {year} surplus/deficit = {sd:+,.0f} tonnes  [{status}]")
        summary_rows.append({"crop": crop, "surplus_deficit_2024": sd, "status": status})


# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("outputs/deficit_summary.csv", index=False)

print("\n=== Deficit / surplus summary ===")
print(summary_df.to_string(index=False))

deficit_crops = summary_df[summary_df["status"] == "DEFICIT"]["crop"].tolist()
print(f"\nDeficit crops: {deficit_crops}")
print("(Paper expects: garlic, groundnuts, onions, sweet_potato — but garlic has no FBS data)")

# Simple bar chart of surplus/deficit in first forecast year
if not summary_df.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#d62728" if s < 0 else "#2ca02c"
              for s in summary_df["surplus_deficit_2024"]]
    ax.bar(summary_df["crop"], summary_df["surplus_deficit_2024"] / 1e6, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Surplus / Deficit (million tonnes)")
    ax.set_title("Forecast Surplus / Deficit by Crop (first forecast year)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/deficit_chart.png", dpi=150)
    print("\nChart saved to outputs/deficit_chart.png")

print("\n=== Deficit analysis complete. ===")
