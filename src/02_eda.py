"""
Step 2 — Exploratory Data Analysis.

For each of the 15 crops, runs decomposition, differencing plots,
ACF/PACF, and ADF stationarity test via PyCaret TSForecastingExperiment.
Saves all plots as HTML (Plotly) to outputs/eda/.
"""

import os
import shutil
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
os.makedirs("outputs/eda", exist_ok=True)

CROPS = [
    "rice", "corn", "coconut", "sugarcane", "banana",
    "pineapple", "coffee", "cacao", "cassava", "sweet_potato",
    "groundnut", "onion", "garlic", "eggplant", "cabbage",
]

# PyCaret saves plots under these exact filenames in the cwd
PLOT_FILENAMES = {
    "decomp": "Classical Decomposition.html",
    "diff":   "Difference Plot.html",
    "acf":    "Auto Correlation (ACF).html",
    "pacf":   "Partial Auto Correlation (PACF).html",
}


def save_plot(exp, plot: str, crop: str, **kwargs) -> str:
    """Call exp.plot_model, move the saved file to outputs/eda/, return dest path."""
    exp.plot_model(plot=plot, save=True, verbose=False, **kwargs)
    src = PLOT_FILENAMES[plot]
    dest = f"outputs/eda/{crop}_{plot}.html"
    if os.path.exists(src):
        shutil.move(src, dest)
        return dest
    else:
        return f"WARNING: expected file '{src}' not found"


def run_eda(crop: str) -> None:
    from pycaret.time_series import TSForecastingExperiment

    df = pd.read_csv(f"data/interim/prod_{crop}.csv",
                     index_col="Year", parse_dates=True)
    df.index = pd.PeriodIndex(df.index, freq="Y")

    exp = TSForecastingExperiment()
    exp.setup(
        data=df,
        target="Value",
        fh=20,              # ✅ 20-year forecast horizon
        fold=2,             # ⚠️ fold=3 needs 80+ rows; we have 64 so using 2
        seasonal_period=1,  # annual data — no within-year seasonality
        session_id=123,     # ⚠️ fixed seed (not stated in paper)
        verbose=False,
    )

    # ADF stationarity test
    print(f"\n--- {crop.upper()} ADF stationarity test ---")
    stats = exp.check_stats(test="adf")
    stationary = stats.iloc[0]["Value"]
    p_value    = stats.iloc[1]["Value"]
    print(f"  Stationary: {stationary}   p-value: {p_value:.6f}")

    # Decomposition
    dest = save_plot(exp, "decomp", crop)
    print(f"  decomp → {dest}")

    # First-difference
    dest = save_plot(exp, "diff", crop, data_kwargs={"order_list": [1]})
    print(f"  diff   → {dest}")

    # ACF
    dest = save_plot(exp, "acf", crop)
    print(f"  acf    → {dest}")

    # PACF
    dest = save_plot(exp, "pacf", crop)
    print(f"  pacf   → {dest}")


if __name__ == "__main__":
    for crop in CROPS:
        print(f"\n{'='*50}")
        print(f"EDA: {crop}")
        print(f"{'='*50}")
        try:
            run_eda(crop)
        except Exception as e:
            print(f"  ERROR for {crop}: {e}")

    print("\n=== EDA complete. Plots saved to outputs/eda/ ===")
