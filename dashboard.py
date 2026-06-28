"""
Philippine Food-Security Forecasting Dashboard
Streamlit + Plotly interactive dashboard covering:
  - Overview (KPI cards + surplus/deficit summary)
  - Production History (1961-2024 trends)
  - EDA (decomposition, differencing, ACF, PACF)
  - Model Comparison (MAPE heatmap + best-model table)
  - Forecast Explorer (PyCaret vs ARIMA vs Prophet)
  - Food Security (20-year surplus/deficit projections)

Run:
    source .venv/bin/activate
    streamlit run dashboard.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PH Food Security Forecast",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CROPS = [
    "rice", "corn", "coconut", "sugarcane", "banana",
    "pineapple", "coffee", "cacao", "cassava", "sweet_potato",
    "groundnut", "onion", "garlic", "eggplant", "cabbage",
]

FBS_CROPS = [
    "rice", "corn", "coconut", "sugarcane", "banana",
    "pineapple", "coffee", "cacao", "cassava", "sweet_potato",
    "groundnut", "onion",
]

CROP_LABELS = {c: c.replace("_", " ").title() for c in CROPS}

COLOR_DEFICIT  = "#e63946"
COLOR_SURPLUS  = "#2a9d8f"
COLOR_FORECAST = "#e9c46a"
COLOR_ARIMA    = "#f4a261"
COLOR_PROPHET  = "#a8dadc"
COLOR_HIST     = "#457b9d"

# ─────────────────────────────────────────────
# DATA LOADERS  (cached so they only read once)
# ─────────────────────────────────────────────
@st.cache_data
def load_production(crop):
    df = pd.read_csv(f"data/interim/prod_{crop}.csv",
                     index_col="Year", parse_dates=True)
    df.index = df.index.year
    df.columns = ["production"]
    return df

@st.cache_data
def load_percapita(crop):
    path = f"data/interim/percapita_{crop}.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data
def load_population():
    pop = pd.read_csv("data/interim/population.csv")
    return pop[pop["Year"].between(1961, 2044)]

@st.cache_data
def load_forecast(crop):
    path = f"data/processed/{crop}_forecast.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)   # index is already integer years (2025–2044)
    df.columns = ["forecast"]
    return df

@st.cache_data
def load_arima_forecast(crop):
    path = f"data/processed/{crop}_arima_forecast.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data
def load_prophet_forecast(crop):
    path = f"data/processed/{crop}_prophet_forecast.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"]).dt.year
    return df[df["ds"] >= 2025].reset_index(drop=True)

@st.cache_data
def load_deficit(crop):
    path = f"data/processed/{crop}_deficit.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data
def load_model_summary():
    return pd.read_csv("outputs/model_summary.csv")

@st.cache_data
def load_comparison(crop):
    path = f"outputs/{crop}_comparison.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0)

@st.cache_data
def load_deficit_summary():
    return pd.read_csv("outputs/deficit_summary.csv")

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/99/Flag_of_the_Philippines.svg",
             width=60)
    st.title("PH Food Security")
    st.caption("Philippine Agricultural Forecast · FAO Data 1961–2044")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📈 Production History",
            "🔬 EDA",
            "🤖 Model Comparison",
            "🔮 Forecast Explorer",
            "🌾 Food Security",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: FAO FAOSTAT · Models: PyCaret 3.3, pmdarima, Prophet")

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🌾 Philippine Food Security Forecast")
    st.markdown(
        "A replication of the FAO-based food production forecasting study using "
        "**PyCaret**, **Auto ARIMA**, and **Facebook Prophet** on 15 Philippine crops (1961–2024)."
    )
    st.divider()

    summary = load_deficit_summary()
    n_deficit = (summary["status"] == "DEFICIT").sum()
    n_surplus = (summary["status"] == "surplus").sum()
    worst = summary.loc[summary["surplus_deficit_2024"].idxmin()]
    best  = summary.loc[summary["surplus_deficit_2024"].idxmax()]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Crops Analysed", "15")
    c2.metric("Deficit Crops",  f"{n_deficit}", delta=f"{n_deficit} need action", delta_color="inverse")
    c3.metric("Surplus Crops",  f"{n_surplus}", delta=f"{n_surplus} self-sufficient", delta_color="normal")
    c4.metric("Largest Deficit", CROP_LABELS[worst["crop"]],
              delta=f"{worst['surplus_deficit_2024']/1e6:+.2f}M t", delta_color="inverse")
    c5.metric("Largest Surplus", CROP_LABELS[best["crop"]],
              delta=f"{best['surplus_deficit_2024']/1e6:+.2f}M t", delta_color="normal")

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Surplus / Deficit by Crop — First Forecast Year (2025)")
        df_plot = summary.sort_values("surplus_deficit_2024")
        df_plot["label"] = df_plot["crop"].map(CROP_LABELS)
        df_plot["color"] = df_plot["surplus_deficit_2024"].apply(
            lambda x: COLOR_DEFICIT if x < 0 else COLOR_SURPLUS
        )
        df_plot["value_M"] = df_plot["surplus_deficit_2024"] / 1e6

        fig = go.Figure(go.Bar(
            x=df_plot["value_M"],
            y=df_plot["label"],
            orientation="h",
            marker_color=df_plot["color"],
            text=df_plot["value_M"].apply(lambda x: f"{x:+.2f}M t"),
            textposition="outside",
            hovertemplate="%{y}: %{x:.3f}M tonnes<extra></extra>",
        ))
        fig.add_vline(x=0, line_color="black", line_width=1.5)
        fig.update_layout(
            height=400, xaxis_title="Million tonnes",
            margin=dict(l=10, r=80, t=10, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Status Table")
        tbl = summary.copy()
        tbl["Crop"] = tbl["crop"].map(CROP_LABELS)
        tbl["Surplus / Deficit (t)"] = tbl["surplus_deficit_2024"].apply(lambda x: f"{x:+,.0f}")
        tbl["Status"] = tbl["status"].str.upper()
        tbl = tbl[["Crop", "Surplus / Deficit (t)", "Status"]].reset_index(drop=True)

        def colour_status(val):
            return f"color: {COLOR_DEFICIT}" if val == "DEFICIT" else f"color: {COLOR_SURPLUS}"

        st.dataframe(
            tbl.style.map(colour_status, subset=["Status"]),
            use_container_width=True, hide_index=True, height=400,
        )

    st.divider()
    st.subheader("3 Crops Without Food Security Data")
    st.info(
        "**Garlic, Eggplant, Cabbage** have production forecasts but no FAO Food Balance Sheet "
        "per-capita supply data, so surplus/deficit cannot be computed. They appear in all "
        "forecast files but are excluded from the food security analysis.",
        icon="⚠️",
    )

    st.subheader("Key Finding — Paper vs Replication")
    c1, c2 = st.columns(2)
    c1.markdown("""
**Paper's stated deficit crops**
- Garlic *(no FBS data — unverifiable)*
- Groundnuts ✅
- Onions ✅
- Sweet Potatoes ✅
""")
    c2.markdown("""
**Our replication adds**
- Rice *(imports inflate FBS per-capita)*
- Coffee *(major importer)*
- Cacao *(major importer)*
""")

# ─────────────────────────────────────────────
# PAGE 2 — PRODUCTION HISTORY
# ─────────────────────────────────────────────
elif page == "📈 Production History":
    st.title("📈 Historical Production Trends")
    st.caption("FAO production data · Philippines · 1961–2024 · Unit: tonnes")

    selected = st.multiselect(
        "Select crops to compare",
        options=CROPS,
        default=["rice", "corn", "coconut", "sugarcane"],
        format_func=lambda c: CROP_LABELS[c],
    )

    normalize = st.checkbox("Normalize to 1961 baseline (index = 100)", value=False)

    if not selected:
        st.warning("Select at least one crop.")
    else:
        fig = go.Figure()
        for crop in selected:
            df = load_production(crop)
            y = df["production"].values.astype(float)
            if normalize:
                y = y / y[0] * 100
            fig.add_trace(go.Scatter(
                x=df.index, y=y,
                mode="lines+markers", name=CROP_LABELS[crop],
                marker=dict(size=4),
                hovertemplate=f"<b>{CROP_LABELS[crop]}</b><br>Year: %{{x}}<br>{'Index' if normalize else 'Tonnes'}: %{{y:,.1f}}<extra></extra>",
            ))

        fig.update_layout(
            height=480,
            yaxis_title="Index (1961 = 100)" if normalize else "Tonnes",
            xaxis_title="Year",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
        fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Summary Statistics")
        rows = []
        for crop in selected:
            df = load_production(crop)
            s = df["production"]
            rows.append({
                "Crop": CROP_LABELS[crop],
                "First year": int(df.index.min()),
                "Last year": int(df.index.max()),
                "1961 (t)": f"{s.iloc[0]:,.0f}",
                "2024 (t)": f"{s.iloc[-1]:,.0f}",
                "Growth ×": f"{s.iloc[-1]/s.iloc[0]:.1f}×",
                "Peak (t)": f"{s.max():,.0f}",
                "Peak year": int(df.index[s.values.argmax()]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# PAGE 3 — EDA
# ─────────────────────────────────────────────
elif page == "🔬 EDA":
    st.title("🔬 Exploratory Data Analysis")
    st.caption("PyCaret TSForecastingExperiment · Decomposition · Stationarity · Autocorrelation")

    crop = st.selectbox("Select crop", CROPS, format_func=lambda c: CROP_LABELS[c])

    # ADF result banner
    adf_results = {
        "rice": (False, 0.9833), "corn": (False, 0.9346), "coconut": (False, 0.2963),
        "sugarcane": (True, 0.0122), "banana": (False, 0.5259), "pineapple": (False, 0.9732),
        "coffee": (False, 0.4398), "cacao": (False, 0.3846), "cassava": (False, 0.3097),
        "sweet_potato": (False, 0.4200), "groundnut": (False, 0.0823), "onion": (False, 0.9991),
        "garlic": (False, 0.9335), "eggplant": (False, 0.8307), "cabbage": (False, 0.3977),
    }
    is_stationary, p_value = adf_results[crop]
    if is_stationary:
        st.success(f"ADF Test: **Stationary** (p = {p_value:.4f}) — the series has no unit root at α = 0.05", icon="✅")
    else:
        st.warning(f"ADF Test: **Non-stationary** (p = {p_value:.4f}) — unit root present; differencing is needed before ARIMA", icon="⚠️")

    tab_decomp, tab_diff, tab_acf, tab_pacf = st.tabs(
        ["📊 Decomposition", "📉 Differencing", "📐 ACF", "📐 PACF"]
    )

    def embed_eda_html(plot_type):
        path = f"outputs/eda/{crop}_{plot_type}.html"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=520, scrolling=False)
        else:
            st.error(f"Plot not found: {path}. Run 02_eda.py first.")

    with tab_decomp:
        st.markdown("**Classical Decomposition** — separates the series into trend, seasonality, and residual components.")
        embed_eda_html("decomp")

    with tab_diff:
        st.markdown("**First-order Differencing** — removes trend so the series becomes closer to stationary.")
        embed_eda_html("diff")

    with tab_acf:
        st.markdown("**Autocorrelation Function (ACF)** — shows correlation with lagged values; helps choose the MA order (q) for ARIMA.")
        embed_eda_html("acf")

    with tab_pacf:
        st.markdown("**Partial Autocorrelation Function (PACF)** — removes indirect correlations; helps choose the AR order (p) for ARIMA.")
        embed_eda_html("pacf")

# ─────────────────────────────────────────────
# PAGE 4 — MODEL COMPARISON
# ─────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    st.caption("PyCaret compare_models() · Metric: MAPE (lower = better) · fold = 2 · fh = 20")

    summary = load_model_summary()

    # ── Best model per crop ──────────────────
    st.subheader("Best Model per Crop")
    fig = go.Figure()
    colours = [COLOR_DEFICIT if m > 0.30 else COLOR_FORECAST if m > 0.20 else COLOR_SURPLUS
               for m in summary["mape"]]
    fig.add_trace(go.Bar(
        x=summary["crop"].map(CROP_LABELS),
        y=summary["mape"],
        text=summary["best_model"],
        textposition="outside",
        marker_color=colours,
        hovertemplate="<b>%{x}</b><br>Best model: %{text}<br>MAPE: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0.10, line_dash="dot", line_color="green",
                  annotation_text="10% threshold (paper rule)", annotation_position="top right")
    fig.update_layout(
        height=420, yaxis_title="MAPE", yaxis_tickformat=".0%",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_tickangle=-35,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    st.plotly_chart(fig, use_container_width=True)

    # ── Full MAPE heatmap ────────────────────
    st.subheader("Full Model × Crop MAPE Heatmap")
    st.caption("Shows top-10 models by median MAPE across all crops. Darker = lower MAPE = better.")

    all_models = []
    for crop in CROPS:
        df = load_comparison(crop)
        if df is not None:
            df = df[["MAPE"]].copy()
            df.columns = [CROP_LABELS[crop]]
            all_models.append(df)

    if all_models:
        heat = pd.concat(all_models, axis=1)
        # keep top-10 models (lowest median MAPE across crops)
        heat["_median"] = heat.median(axis=1)
        heat = heat.sort_values("_median").head(10).drop(columns="_median")

        fig_heat = px.imshow(
            heat.values,
            x=heat.columns.tolist(),
            y=heat.index.tolist(),
            color_continuous_scale="RdYlGn_r",
            text_auto=".3f",
            aspect="auto",
            labels=dict(color="MAPE"),
        )
        fig_heat.update_layout(
            height=400,
            xaxis_title="", yaxis_title="Model",
            margin=dict(l=150, r=20, t=20, b=80),
            coloraxis_colorbar=dict(title="MAPE", tickformat=".0%"),
        )
        fig_heat.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Detailed comparison for one crop ────
    st.subheader("Detailed Model Ranking — Single Crop")
    crop_sel = st.selectbox("Select crop", CROPS, format_func=lambda c: CROP_LABELS[c],
                            key="model_crop")
    df_comp = load_comparison(crop_sel)
    if df_comp is not None:
        # index = model short-codes; 'Model' column already exists with display names
        # drop the index so columns stay unique (avoids Styler duplicate-column error)
        df_comp = df_comp.reset_index(drop=True)
        df_comp.insert(0, "Rank", range(1, len(df_comp) + 1))
        cols_show = ["Rank", "Model", "MAPE", "MAE", "RMSE", "MASE", "SMAPE", "R2"]
        cols_show = [c for c in cols_show if c in df_comp.columns]
        display_df = df_comp[cols_show].copy()
        st.dataframe(
            display_df.style
                .format({"MAPE": "{:.4f}", "MAE": "{:,.0f}", "RMSE": "{:,.0f}",
                         "MASE": "{:.4f}", "SMAPE": "{:.4f}", "R2": "{:.4f}"})
                .background_gradient(subset=["MAPE"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True, height=420,
        )

# ─────────────────────────────────────────────
# PAGE 5 — FORECAST EXPLORER
# ─────────────────────────────────────────────
elif page == "🔮 Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.caption("20-year production forecast · 2025–2044 · Compare three modelling approaches")

    col1, col2 = st.columns([2, 3])
    with col1:
        crop = st.selectbox("Select crop", CROPS, format_func=lambda c: CROP_LABELS[c])
    with col2:
        show_pycaret = st.checkbox("PyCaret best model", value=True)
        show_arima   = st.checkbox("Auto ARIMA (with 95% CI)", value=True)
        show_prophet = st.checkbox("Prophet (with 95% CI)", value=True)

    hist = load_production(crop)
    fcst = load_forecast(crop)
    arima = load_arima_forecast(crop)
    prophet = load_prophet_forecast(crop)

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["production"],
        mode="lines", name="Historical",
        line=dict(color=COLOR_HIST, width=2),
        hovertemplate="<b>Historical</b><br>Year: %{x}<br>%{y:,.0f} t<extra></extra>",
    ))

    # Vertical split line
    last_hist_year = int(hist.index.max())
    fig.add_vline(x=last_hist_year + 0.5, line_dash="dash",
                  line_color="#888888", line_width=1,
                  annotation_text="Forecast →", annotation_position="top right")

    # PyCaret forecast
    if show_pycaret and fcst is not None:
        model_name = load_model_summary().set_index("crop").loc[crop, "best_model"]
        fig.add_trace(go.Scatter(
            x=fcst.index, y=fcst["forecast"],
            mode="lines+markers", name=f"PyCaret ({model_name})",
            line=dict(color=COLOR_FORECAST, width=2.5, dash="solid"),
            marker=dict(size=5),
            hovertemplate="<b>PyCaret</b><br>Year: %{x}<br>%{y:,.0f} t<extra></extra>",
        ))

    # ARIMA forecast with CI
    if show_arima and arima is not None:
        fig.add_trace(go.Scatter(
            x=arima["Year"], y=arima["forecast"],
            mode="lines", name="Auto ARIMA",
            line=dict(color=COLOR_ARIMA, width=2, dash="dot"),
            hovertemplate="<b>ARIMA</b><br>Year: %{x}<br>%{y:,.0f} t<extra></extra>",
        ))
        fig.add_traces([
            go.Scatter(x=arima["Year"], y=arima["upper_ci"],
                       mode="lines", line=dict(width=0), showlegend=False,
                       hoverinfo="skip"),
            go.Scatter(x=arima["Year"], y=arima["lower_ci"],
                       mode="lines", line=dict(width=0),
                       fill="tonexty", fillcolor="rgba(244,162,97,0.20)",
                       name="ARIMA 95% CI", hoverinfo="skip"),
        ])

    # Prophet forecast with CI
    if show_prophet and prophet is not None:
        fig.add_trace(go.Scatter(
            x=prophet["ds"], y=prophet["yhat"],
            mode="lines", name="Prophet",
            line=dict(color=COLOR_PROPHET, width=2, dash="dashdot"),
            hovertemplate="<b>Prophet</b><br>Year: %{x}<br>%{y:,.0f} t<extra></extra>",
        ))
        fig.add_traces([
            go.Scatter(x=prophet["ds"], y=prophet["yhat_upper"],
                       mode="lines", line=dict(width=0), showlegend=False,
                       hoverinfo="skip"),
            go.Scatter(x=prophet["ds"], y=prophet["yhat_lower"],
                       mode="lines", line=dict(width=0),
                       fill="tonexty", fillcolor="rgba(168,218,220,0.25)",
                       name="Prophet 95% CI", hoverinfo="skip"),
        ])

    fig.update_layout(
        height=500,
        title=f"{CROP_LABELS[crop]} — Production History + 20-Year Forecast",
        yaxis_title="Tonnes", xaxis_title="Year",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast data table
    with st.expander("📋 View forecast numbers"):
        rows = {"Year": list(range(2025, 2045))}
        if fcst is not None:
            rows["PyCaret (t)"] = fcst["forecast"].values[:20]
        if arima is not None:
            rows["ARIMA (t)"] = arima["forecast"].values[:20]
            rows["ARIMA lower CI"] = arima["lower_ci"].values[:20]
            rows["ARIMA upper CI"] = arima["upper_ci"].values[:20]
        if prophet is not None:
            rows["Prophet (t)"] = prophet["yhat"].values[:20]
        tbl = pd.DataFrame(rows)
        st.dataframe(
            tbl.style.format({c: "{:,.0f}" for c in tbl.columns if c != "Year"}),
            use_container_width=True, hide_index=True,
        )

# ─────────────────────────────────────────────
# PAGE 6 — FOOD SECURITY
# ─────────────────────────────────────────────
elif page == "🌾 Food Security":
    st.title("🌾 Food Security Analysis")
    st.caption("Surplus / Deficit = Forecasted Production − Estimated Consumption · 2025–2044")

    tab_all, tab_crop, tab_pc = st.tabs(
        ["📊 All Crops Overview", "🔍 Crop Deep-Dive", "📉 Per-Capita Trends"]
    )

    # ── Tab 1: All crops overview ────────────
    with tab_all:
        st.markdown("**Surplus / Deficit in 2025 (first forecast year) across all crops with FBS data.**")

        summary = load_deficit_summary()
        summary["label"] = summary["crop"].map(CROP_LABELS)

        fig = px.treemap(
            summary,
            path=["status", "label"],
            values=summary["surplus_deficit_2024"].abs(),
            color="surplus_deficit_2024",
            color_continuous_scale=["#e63946", "#ffffff", "#2a9d8f"],
            color_continuous_midpoint=0,
            custom_data=["surplus_deficit_2024"],
            title="Treemap: Size = Magnitude · Colour = Deficit (red) vs Surplus (green)",
        )
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Surplus/Deficit: %{customdata[0]:+,.0f} t<extra></extra>"
        )
        fig.update_layout(height=420, margin=dict(t=50, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # 20-yr projection for all crops as small multiples
        st.subheader("20-Year Surplus / Deficit Trajectory — All Crops")
        n_cols = 3
        crop_chunks = [FBS_CROPS[i:i+n_cols] for i in range(0, len(FBS_CROPS), n_cols)]
        for chunk in crop_chunks:
            cols = st.columns(n_cols)
            for col, crop in zip(cols, chunk):
                deficit = load_deficit(crop)
                if deficit is None:
                    continue
                color = [COLOR_DEFICIT if v < 0 else COLOR_SURPLUS
                         for v in deficit["surplus_deficit_tonnes"]]
                fig_mini = go.Figure(go.Bar(
                    x=deficit["year"],
                    y=deficit["surplus_deficit_tonnes"] / 1e3,
                    marker_color=color,
                    hovertemplate="Year: %{x}<br>%{y:+,.0f}K t<extra></extra>",
                ))
                fig_mini.add_hline(y=0, line_color="black", line_width=1)
                fig_mini.update_layout(
                    title=dict(text=CROP_LABELS[crop], font=dict(size=13)),
                    height=220, margin=dict(l=10, r=10, t=35, b=30),
                    yaxis_title="000 t", xaxis_title="",
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False,
                )
                fig_mini.update_xaxes(showgrid=False, tickangle=-45, dtick=5)
                fig_mini.update_yaxes(showgrid=True, gridcolor="#eeeeee")
                col.plotly_chart(fig_mini, use_container_width=True)

    # ── Tab 2: Crop deep-dive ────────────────
    with tab_crop:
        crop = st.selectbox("Select crop", FBS_CROPS, format_func=lambda c: CROP_LABELS[c],
                            key="fs_crop")
        deficit = load_deficit(crop)
        pc      = load_percapita(crop)
        pop     = load_population()
        hist    = load_production(crop)

        if deficit is None:
            st.error("No deficit data found. Run 04_deficit.py first.")
        else:
            # KPI row
            first = deficit.iloc[0]
            last  = deficit.iloc[-1]
            sd_2025 = first["surplus_deficit_tonnes"]
            sd_2044 = last["surplus_deficit_tonnes"]
            years_deficit = (deficit["surplus_deficit_tonnes"] < 0).sum()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Status 2025", "DEFICIT" if sd_2025 < 0 else "SURPLUS",
                      delta=f"{sd_2025/1e3:+,.0f}K t", delta_color="inverse" if sd_2025 < 0 else "normal")
            k2.metric("Status 2044", "DEFICIT" if sd_2044 < 0 else "SURPLUS",
                      delta=f"{sd_2044/1e3:+,.0f}K t", delta_color="inverse" if sd_2044 < 0 else "normal")
            k3.metric("Deficit years (20yr)", f"{years_deficit} / 20",
                      delta_color="off")
            k4.metric("Per-capita used", f"{first['kg_per_capita']:.2f} kg/cap/yr",
                      delta="5-yr avg 2019–2023", delta_color="off")

            st.divider()

            # Stacked area: production vs consumption
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=deficit["year"], y=deficit["est_consumption_tonnes"] / 1e6,
                mode="lines", name="Estimated Consumption",
                fill="tozeroy", fillcolor="rgba(230,57,70,0.15)",
                line=dict(color=COLOR_DEFICIT, width=2),
                hovertemplate="Consumption: %{y:.3f}M t<extra></extra>",
            ))
            fig2.add_trace(go.Scatter(
                x=deficit["year"], y=deficit["forecast_production_tonnes"] / 1e6,
                mode="lines", name="Forecasted Production",
                fill="tozeroy", fillcolor="rgba(42,157,143,0.20)",
                line=dict(color=COLOR_SURPLUS, width=2),
                hovertemplate="Production: %{y:.3f}M t<extra></extra>",
            ))
            fig2.update_layout(
                title=f"{CROP_LABELS[crop]} — Production vs Consumption (2025–2044)",
                height=380, yaxis_title="Million tonnes", xaxis_title="Year",
                plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
                legend=dict(orientation="h", y=1.12),
            )
            fig2.update_xaxes(showgrid=True, gridcolor="#eeeeee")
            fig2.update_yaxes(showgrid=True, gridcolor="#eeeeee")
            st.plotly_chart(fig2, use_container_width=True)

            # Surplus/Deficit bar
            fig3 = go.Figure(go.Bar(
                x=deficit["year"],
                y=deficit["surplus_deficit_tonnes"] / 1e3,
                marker_color=[COLOR_DEFICIT if v < 0 else COLOR_SURPLUS
                              for v in deficit["surplus_deficit_tonnes"]],
                hovertemplate="Year: %{x}<br>%{y:+,.1f}K t<extra></extra>",
            ))
            fig3.add_hline(y=0, line_color="black", line_width=1.5)
            fig3.update_layout(
                title="Annual Surplus / Deficit",
                height=300, yaxis_title="000 tonnes", xaxis_title="Year",
                plot_bgcolor="white", paper_bgcolor="white",
            )
            fig3.update_yaxes(showgrid=True, gridcolor="#eeeeee")
            st.plotly_chart(fig3, use_container_width=True)

            # Data table
            with st.expander("📋 Full deficit data table"):
                tbl = deficit.copy()
                tbl.columns = ["Year", "Production (t)", "Population", "kg/cap/yr",
                               "Consumption (t)", "Surplus/Deficit (t)"]
                st.dataframe(
                    tbl.style.format({
                        "Production (t)": "{:,.0f}", "Population": "{:,.0f}",
                        "kg/cap/yr": "{:.2f}", "Consumption (t)": "{:,.0f}",
                        "Surplus/Deficit (t)": "{:+,.0f}",
                    }),
                    use_container_width=True, hide_index=True,
                )

    # ── Tab 3: Per-capita trends ─────────────
    with tab_pc:
        st.markdown(
            "**FAO Food Balance Sheet — Food supply quantity (kg/capita/yr).**  "
            "Stitched series: historic file ≤ 2009, new file ≥ 2010."
        )
        selected_pc = st.multiselect(
            "Select crops", FBS_CROPS,
            default=["rice", "onion", "groundnut", "sweet_potato"],
            format_func=lambda c: CROP_LABELS[c], key="pc_crops",
        )

        if selected_pc:
            fig4 = go.Figure()
            for crop in selected_pc:
                pc = load_percapita(crop)
                if pc is None:
                    continue
                fig4.add_trace(go.Scatter(
                    x=pc["Year"], y=pc["kg_per_capita"],
                    mode="lines+markers", name=CROP_LABELS[crop],
                    marker=dict(size=4),
                    hovertemplate=f"<b>{CROP_LABELS[crop]}</b><br>Year: %{{x}}<br>%{{y:.2f}} kg/cap<extra></extra>",
                ))
            fig4.add_vline(x=2009.5, line_dash="dot", line_color="#888888",
                           annotation_text="FBS seam (2009→2010)",
                           annotation_position="top right")
            fig4.update_layout(
                height=440, title="Per-capita Food Supply (kg/capita/yr) · 1961–2023",
                yaxis_title="kg / capita / year", xaxis_title="Year",
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=1.12),
                hovermode="x unified",
            )
            fig4.update_xaxes(showgrid=True, gridcolor="#eeeeee")
            fig4.update_yaxes(showgrid=True, gridcolor="#eeeeee")
            st.plotly_chart(fig4, use_container_width=True)

            # Population overlay
            st.subheader("Population Growth (FAO Projection · 1961–2044)")
            pop = load_population()
            fig5 = go.Figure(go.Scatter(
                x=pop["Year"], y=pop["population"] / 1e6,
                mode="lines", fill="tozeroy",
                fillcolor="rgba(69,123,157,0.15)",
                line=dict(color=COLOR_HIST, width=2),
                hovertemplate="Year: %{x}<br>Population: %{y:.1f}M<extra></extra>",
            ))
            fig5.add_vline(x=2024, line_dash="dash", line_color="#888",
                           annotation_text="← Historical | Projected →")
            fig5.update_layout(
                height=280, yaxis_title="Million persons", xaxis_title="Year",
                plot_bgcolor="white", paper_bgcolor="white",
            )
            fig5.update_xaxes(showgrid=True, gridcolor="#eeeeee")
            fig5.update_yaxes(showgrid=True, gridcolor="#eeeeee")
            st.plotly_chart(fig5, use_container_width=True)
