"""
Step 3 — Modeling.

Two tracks for each of the 15 crops:
  Track A: PyCaret compare_models() — finds the best model by MAPE.
  Track B: manual Auto ARIMA (pmdarima) + Prophet.

Outputs per crop:
  outputs/<crop>_comparison.csv   — full metric table from PyCaret
  data/processed/<crop>_forecast.csv — 20-year forecast from best model
  data/processed/<crop>_arima_forecast.csv — Auto ARIMA forecast
  data/processed/<crop>_prophet_forecast.csv — Prophet forecast

Summary:
  outputs/model_summary.csv — crop | best_model | MAPE across all 15 crops
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

CROPS = [
    "rice", "corn", "coconut", "sugarcane", "banana",
    "pineapple", "coffee", "cacao", "cassava", "sweet_potato",
    "groundnut", "onion", "garlic", "eggplant", "cabbage",
]

FORECAST_HORIZON = 20  # ✅ from paper


def load_crop(crop: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/interim/prod_{crop}.csv",
                     index_col="Year", parse_dates=True)
    df.index = pd.PeriodIndex(df.index, freq="Y")
    return df


# ---------------------------------------------------------------------------
# Track A — PyCaret compare_models
# ---------------------------------------------------------------------------
def track_a_pycaret(crop: str, df: pd.DataFrame) -> dict:
    from pycaret.time_series import TSForecastingExperiment

    exp = TSForecastingExperiment()
    exp.setup(
        data=df,
        target="Value",
        fh=FORECAST_HORIZON,
        fold=2,             # fold=3 needs 80+ rows; we have 64
        seasonal_period=1,
        session_id=123,
        verbose=False,
    )

    best = exp.compare_models(sort="MAPE", verbose=False)
    results = exp.pull()
    results.to_csv(f"outputs/{crop}_comparison.csv")

    final = exp.finalize_model(best)
    fcst = exp.predict_model(final, fh=FORECAST_HORIZON, verbose=False)
    fcst.to_csv(f"data/processed/{crop}_forecast.csv")

    # extract best model name and MAPE from results table
    top_row = results.iloc[0]
    best_model = str(top_row.name) if hasattr(top_row, "name") else results.index[0]
    best_mape = float(top_row.get("MAPE", top_row.iloc[3]))

    return {"best_model": best_model, "mape": best_mape}


# ---------------------------------------------------------------------------
# Track B — Auto ARIMA
# ---------------------------------------------------------------------------
def track_b_arima(crop: str, df: pd.DataFrame) -> pd.DataFrame:
    from pmdarima import auto_arima

    y = df["Value"].astype(float).values
    years = df.index.year

    # ✅ paper decision rule: if MAPE > 10% after first fit, raise limits
    for max_p, max_q, max_ord in [(5, 5, 10), (8, 8, 15), (10, 10, 20)]:
        model = auto_arima(
            y,
            seasonal=False,
            max_p=max_p, max_q=max_q,
            max_order=max_ord,
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
        )
        # compute in-sample MAPE to check the 10% rule
        fitted = model.predict_in_sample()
        mape = np.mean(np.abs((y[1:] - fitted[1:]) / (y[1:] + 1e-8)))
        if mape <= 0.10 or (max_p == 10):
            break

    print(f"  ARIMA order: {model.order}, in-sample MAPE: {mape:.4f}")

    # forecast FORECAST_HORIZON years ahead
    last_year = int(years[-1])
    future_years = list(range(last_year + 1, last_year + FORECAST_HORIZON + 1))
    fc_vals, conf_int = model.predict(n_periods=FORECAST_HORIZON, return_conf_int=True)

    fcst_df = pd.DataFrame({
        "Year": future_years,
        "forecast": fc_vals,
        "lower_ci": conf_int[:, 0],
        "upper_ci": conf_int[:, 1],
    })
    fcst_df.to_csv(f"data/processed/{crop}_arima_forecast.csv", index=False)
    return fcst_df


# ---------------------------------------------------------------------------
# Track B — Prophet
# ---------------------------------------------------------------------------
def track_b_prophet(crop: str, df: pd.DataFrame) -> pd.DataFrame:
    from prophet import Prophet

    pdf = (df.reset_index()
             .rename(columns={"Year": "ds", "Value": "y"}))
    pdf["ds"] = pdf["ds"].dt.to_timestamp()

    m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                daily_seasonality=False)  # ✅ annual data, no seasonality
    m.fit(pdf)

    future = m.make_future_dataframe(periods=FORECAST_HORIZON, freq="YS")
    forecast = m.predict(future)
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        f"data/processed/{crop}_prophet_forecast.csv", index=False
    )
    return forecast


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
summary_rows = []

for crop in CROPS:
    print(f"\n{'='*55}")
    print(f"Modeling: {crop}")
    print(f"{'='*55}")

    df = load_crop(crop)
    print(f"  Series: {df.index.min()}–{df.index.max()}, {len(df)} rows")

    # Track A
    try:
        print("  [Track A] PyCaret compare_models ...")
        result = track_a_pycaret(crop, df)
        print(f"  Best: {result['best_model']}, MAPE: {result['mape']:.4f}")
        summary_rows.append({
            "crop": crop,
            "best_model": result["best_model"],
            "mape": result["mape"],
        })
    except Exception as e:
        print(f"  Track A ERROR: {e}")
        summary_rows.append({"crop": crop, "best_model": "ERROR", "mape": None})

    # Track B — Auto ARIMA
    try:
        print("  [Track B] Auto ARIMA ...")
        track_b_arima(crop, df)
    except Exception as e:
        print(f"  Auto ARIMA ERROR: {e}")

    # Track B — Prophet
    try:
        print("  [Track B] Prophet ...")
        track_b_prophet(crop, df)
        print(f"  Prophet forecast saved.")
    except Exception as e:
        print(f"  Prophet ERROR: {e}")

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("outputs/model_summary.csv", index=False)
print("\n=== Modeling complete. ===")
print(summary_df.to_string(index=False))
print("\nSummary saved to outputs/model_summary.csv")
