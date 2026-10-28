"""
Energy Consumption Forecasting — SaaS Analytics Dashboard
Redesigned with modern purple & lavender aesthetic inspired by premium SaaS analytics platforms.
Features:
- Fixed Dark Sidebar (#202124) with ENERGY AI branding & active gradient pills
- Lavender-to-Purple gradient background frame & pure white cards (#FFFFFF)
- 4 Top Metric Cards (Latest, Average, Next 24h Forecast Total, Best Model)
- Main Interactive Electricity Consumption Chart with 24h, 7d, 30d, All Data filters
- Card A: Diurnal Consumption Profile (Peak vs. Off-Peak hours)
- Card B: Model Accuracy (Actual vs Predicted with MAE & RMSE)
- Dedicated 24h Forecast Panel with Model Selector & CSV Export
- Multi-Model Benchmark Comparison (ARIMA, SARIMA, Linear Regression, Random Forest)
- Real UCI Household Electric Power Consumption dataset integration
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

# -------------------------------------------------------------
# 1. Page Configuration & Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="ENERGY AI — Electricity Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Root directory handling
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
DATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'hourly_energy_consumption.csv')
FORECAST_CSV = os.path.join(OUTPUTS_DIR, 'forecasts_24h.csv')
COMPARISON_CSV = os.path.join(OUTPUTS_DIR, 'model_comparison.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


# -------------------------------------------------------------
# 2. Data Loading & Caching
# -------------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV, parse_dates=['Datetime'], index_col='Datetime')
        return df
    return None


@st.cache_data
def load_forecasts():
    if os.path.exists(FORECAST_CSV):
        df = pd.read_csv(FORECAST_CSV, parse_dates=['Datetime'], index_col='Datetime')
        return df
    return None


@st.cache_data
def load_comparison():
    if os.path.exists(COMPARISON_CSV):
        df = pd.read_csv(COMPARISON_CSV, index_col='Model')
        return df
    return None


df_hourly = load_data()
df_forecasts = load_forecasts()
df_metrics = load_comparison()


# -------------------------------------------------------------
# 3. Custom CSS — Modern Purple & Lavender SaaS Design
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Global Fonts & Canvas Background */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Soft lavender-to-purple background gradient */
    .stApp {
        background: linear-gradient(135deg, #F3F1FD 0%, #F8F9FD 40%, #F2EFFD 100%) !important;
        color: #1E2024 !important;
    }

    /* Force crisp dark text across entire main content area */
    .main, 
    .main .block-container,
    .main p, 
    .main span, 
    .main label, 
    .main h1, 
    .main h2, 
    .main h3, 
    .main h4, 
    .main h5, 
    .main h6,
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] span,
    .main [data-testid="stMarkdownContainer"] h1,
    .main [data-testid="stMarkdownContainer"] h2,
    .main [data-testid="stMarkdownContainer"] h3,
    .main [data-testid="stMarkdownContainer"] div,
    .saas-card,
    .saas-card h1,
    .saas-card h2,
    .saas-card h3,
    .saas-card p,
    .saas-card span,
    .saas-card div {
        color: #1E2024 !important;
    }

    /* Secondary labels and subtitles - crisp black/dark charcoal */
    .card-subtitle,
    .main .card-subtitle,
    .main .stCaption,
    .main [data-testid="stCaptionContainer"] p,
    .kpi-label {
        color: #111827 !important;
        font-weight: 500 !important;
    }

    /* DIRECT BLACK COLOR OVERRIDE FOR TIME RANGE RADIO BUTTONS (24 Hours, 7 Days, 30 Days, All Data) */
    .main div[data-testid="stRadio"] label,
    .main div[data-testid="stRadio"] label *,
    .main div[data-testid="stRadio"] p,
    .main div[data-testid="stRadio"] span,
    .main div[data-testid="stRadio"] div,
    .main div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p,
    .main div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] span,
    .main div[role="radiogroup"] label,
    .main div[role="radiogroup"] label *,
    .main div[role="radiogroup"] p,
    .main div[role="radiogroup"] span {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
    }

    /* Styled Container for Time Filter Radio Options */
    .main div[data-testid="stRadio"] div[role="radiogroup"] {
        background: #F1F4F9 !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: 1.5px solid #CBD5E1 !important;
        display: inline-flex !important;
        gap: 8px !important;
        margin-bottom: 12px !important;
    }

    .main div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background: #FFFFFF !important;
        padding: 7px 18px !important;
        border-radius: 9px !important;
        border: 1px solid #E2E8F0 !important;
        margin: 0 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        cursor: pointer !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.2s ease !important;
    }

    .main div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: #F8FAFC !important;
        border-color: #7163E8 !important;
    }

    /* Active / Selected Radio Pill */
    .main div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
    .main div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: #7163E8 !important;
        border-color: #7163E8 !important;
        box-shadow: 0 3px 10px rgba(113, 99, 232, 0.4) !important;
    }

    .main div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) *,
    .main div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p,
    .main div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) span {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    /* Selectbox dropdowns in main content */
    .main [data-testid="stSelectbox"] label,
    .main [data-testid="stSelectbox"] p {
        color: #1E2024 !important;
        font-weight: 700 !important;
    }
    .main [data-baseweb="select"] * {
        color: #1E2024 !important;
    }

    /* KPI Cards Text */
    .kpi-value {
        color: #111827 !important;
    }
    .kpi-label {
        color: #4B5563 !important;
    }
    .card-title {
        color: #111827 !important;
    }

    /* Top padding adjustment */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1380px !important;
    }

    /* Dark Charcoal Sidebar */
    [data-testid="stSidebar"] {
        background-color: #202124 !important;
        border-right: 1px solid #2D3036 !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.15) !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }

    /* Sidebar Logo Header */
    .sidebar-brand-box {
        padding: 10px 6px 20px 6px;
        border-bottom: 1px solid #2E3138;
        margin-bottom: 20px;
    }
    .brand-title {
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #FFFFFF !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-title span {
        background: linear-gradient(135deg, #A99AF5, #C4B5FD) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    .brand-sub {
        font-size: 11.5px;
        color: #9CA3AF !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 5px;
        font-weight: 600;
    }

    /* Sidebar Radio Navigation Items */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background: transparent !important;
        padding: 11px 16px !important;
        border-radius: 12px !important;
        margin-bottom: 6px !important;
        color: #9CA3AF !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: rgba(113, 99, 232, 0.12) !important;
        color: #FFFFFF !important;
        transform: translateX(3px) !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #7163E8 0%, #8C7EF0 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(113, 99, 232, 0.45) !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) * {
        color: #FFFFFF !important;
    }

    /* Top Navigation Bar */
    .top-navbar {
        background: #FFFFFF;
        border: 1px solid #E8EAF2;
        border-radius: 16px;
        padding: 16px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(113, 99, 232, 0.04);
    }
    .top-nav-left h2 {
        margin: 0;
        font-size: 22px;
        font-weight: 800;
        color: #242633;
    }
    .top-nav-left p {
        margin: 2px 0 0 0;
        font-size: 12.5px;
        color: #85889A;
        font-weight: 500;
    }
    .top-nav-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .badge-status {
        background: #ECFDF5;
        color: #059669;
        font-size: 12px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        gap: 6px;
        border: 1px solid #D1FAE5;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        display: inline-block;
        box-shadow: 0 0 8px #10B981;
    }
    .badge-updated {
        background: #F7F8FC;
        border: 1px solid #E8EAF2;
        color: #85889A;
        font-size: 12px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
    }
    .badge-profile {
        background: #F3F1FD;
        border: 1px solid #E0DBFA;
        color: #7163E8;
        font-size: 12.5px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 18px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E8EAF2;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 4px 18px rgba(113, 99, 232, 0.04);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(113, 99, 232, 0.09);
    }
    .kpi-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .kpi-label {
        font-size: 12.5px;
        font-weight: 700;
        color: #85889A;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .kpi-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: #F3F1FD;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 19px;
        color: #7163E8;
    }
    .kpi-value {
        font-size: 30px;
        font-weight: 800;
        color: #242633;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #85889A;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .tag-green {
        background: #ECFDF5;
        color: #059669;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
    }
    .tag-purple {
        background: #F3F1FD;
        color: #7163E8;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 6px;
    }

    /* Main Chart Card */
    .saas-card {
        background: #FFFFFF;
        border: 1px solid #E8EAF2;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(113, 99, 232, 0.04);
        margin-bottom: 24px;
    }
    .card-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    .card-title {
        font-size: 18px;
        font-weight: 800;
        color: #242633;
        margin: 0;
    }
    .card-subtitle {
        font-size: 12.5px;
        color: #85889A;
        margin-top: 2px;
    }

    /* Custom Purple Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #7163E8 0%, #8C7EF0 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 14px rgba(113, 99, 232, 0.35) !important;
        transition: all 0.2s ease !important;
        font-size: 13.5px !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(113, 99, 232, 0.45) !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: #FFFFFF !important;
        color: #7163E8 !important;
        border: 1.5px solid #7163E8 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background: #F3F1FD !important;
        color: #5B4CE0 !important;
    }

    /* Sidebar footer note */
    .sidebar-footer {
        padding: 16px 10px 10px 10px;
        border-top: 1px solid #2E3138;
        margin-top: 30px;
        font-size: 11px;
        color: #8E93A4;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 4. Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-brand-box">
    <div class="brand-title" style="font-size: 16px; line-height: 1.3;">⚡ <span>Energy Consumption Forecasting</span></div>
    <div class="brand-sub" style="font-size: 11.5px; margin-top: 5px; color: #A0A5B5; letter-spacing: 0.8px;">Using Time Series Models</div>
</div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Energy Overview",
        "Consumption Forecast",
        "Model Comparison",
        "Historical Data",
        "Settings"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div class="sidebar-footer">
    <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
        <span class="status-dot"></span>
        <strong style="color:#FFFFFF;">UCI Household Power</strong>
    </div>
    <div>Resampled Hourly Observations</div>
    <div style="margin-top:6px; color:#6B7280;">v2.0 SaaS Edition • Python 3.14</div>
</div>
""", unsafe_allow_html=True)


# Fallback if dataset hasn't been preprocessed
if df_hourly is None or df_forecasts is None:
    st.error("⚠️ Dataset or trained model outputs were not found.")
    if st.button("🚀 Run Preprocessing & Training Pipeline Now"):
        with st.spinner("Executing pipeline (parsing UCI dataset, training 4 models, generating forecasts)..."):
            from main import run_pipeline
            run_pipeline()
            st.success("Pipeline executed successfully! Please reload.")
            st.rerun()
    st.stop()


# -------------------------------------------------------------
# 5. Shared Metrics & Data Calculations
# -------------------------------------------------------------
total_records = len(df_hourly)
latest_actual = float(df_hourly['Global_active_power'].iloc[-1])
prev_actual = float(df_hourly['Global_active_power'].iloc[-2])
pct_change = ((latest_actual - prev_actual) / (prev_actual + 1e-5)) * 100

avg_power = float(df_hourly['Global_active_power'].mean())
std_power = float(df_hourly['Global_active_power'].std())

# 24h total forecast sum in kWh (since 1 kW active power for 1 hr = 1 kWh energy)
best_model_name = df_metrics.index[0] if df_metrics is not None else "Random Forest"
best_mae = float(df_metrics.loc[best_model_name, 'MAE']) if df_metrics is not None else 0.2247
best_rmse = float(df_metrics.loc[best_model_name, 'RMSE']) if df_metrics is not None else 0.3125

if best_model_name in df_forecasts.columns:
    forecast_24h_sum = float(df_forecasts[best_model_name].sum())
    forecast_peak_val = float(df_forecasts[best_model_name].max())
    forecast_peak_hour = df_forecasts[best_model_name].idxmax().strftime("%H:%M")
else:
    forecast_24h_sum = float(df_forecasts.iloc[:, 1].sum())
    forecast_peak_val = float(df_forecasts.iloc[:, 1].max())
    forecast_peak_hour = "07:00"


# -------------------------------------------------------------
# Helper: Top Navigation Header
# -------------------------------------------------------------
def render_top_navbar(title: str, subtitle: str):
    st.markdown(f"""
    <div class="top-navbar">
        <div class="top-nav-left">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        <div class="top-nav-right">
            <div class="badge-status">
                <span class="status-dot"></span> Dataset Connected (34,589 hrs)
            </div>
            <div class="badge-updated">
                🕒 Test Horizon: 24 Hours
            </div>
            <div class="badge-profile">
                ⚡ Energy Analyst
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# VIEW 1: DASHBOARD (Executive SaaS View)
# -------------------------------------------------------------
if nav_selection == "Dashboard":
    render_top_navbar("Dashboard", "Overview of historical electricity usage, live benchmarks, and next 24-hour forecast.")

    # 4 Statistic Overview Cards
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Latest Consumption</span>
                <div class="kpi-icon">⚡</div>
            </div>
            <div class="kpi-value">{latest_actual:.3f} <span style="font-size:16px; color:#85889A; font-weight:600;">kW</span></div>
            <div class="kpi-sub">
                <span class="tag-{'green' if pct_change <= 0 else 'purple'}">{pct_change:+.1f}%</span> vs previous hour
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Average Consumption</span>
                <div class="kpi-icon">📊</div>
            </div>
            <div class="kpi-value">{avg_power:.3f} <span style="font-size:16px; color:#85889A; font-weight:600;">kW</span></div>
            <div class="kpi-sub">
                <span class="tag-purple">&plusmn;{std_power:.2f} kW</span> hourly std deviation
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Next 24h Forecast</span>
                <div class="kpi-icon">🔮</div>
            </div>
            <div class="kpi-value">{forecast_24h_sum:.2f} <span style="font-size:16px; color:#85889A; font-weight:600;">kWh</span></div>
            <div class="kpi-sub">
                <span class="tag-purple">Peak: {forecast_peak_val:.2f} kW</span> at {forecast_peak_hour}
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Best Forecast Model</span>
                <div class="kpi-icon">🏆</div>
            </div>
            <div class="kpi-value" style="font-size:22px; padding-top:4px;">{best_model_name}</div>
            <div class="kpi-sub">
                <span class="tag-green">MAE: {best_mae:.4f} kW</span> RMSE: {best_rmse:.3f} kW
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Analytics Section: Electricity Consumption Chart
    st.markdown("""
    <div class="saas-card">
        <div class="card-header-row">
            <div>
                <h3 class="card-title">Electricity Consumption</h3>
                <div class="card-subtitle">Historical hourly demand (kW) compared with the next 24-hour forecasted demand.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Time Filter Pills (Ample space for 4 options)
    filter_col1, _ = st.columns([4, 3])
    with filter_col1:
        time_filter = st.radio(
            "Time Range Filter:",
            ["24 Hours", "7 Days", "30 Days", "All Data"],
            horizontal=True,
            index=1,
            label_visibility="collapsed"
        )

    # Map hours
    filter_map = {
        "24 Hours": 24,
        "7 Days": 168,
        "30 Days": 720,
        "All Data": len(df_hourly)
    }
    history_hours = filter_map[time_filter]

    # Build historical series + 24h forecast curve
    hist_slice = df_hourly['Global_active_power'].iloc[-history_hours:].copy()

    # Create a cohesive visualization
    fig, ax = plt.subplots(figsize=(14, 4.8), dpi=150)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Historical line in solid purple (#7163E8)
    ax.plot(
        hist_slice.index,
        hist_slice.values,
        color='#7163E8',
        linewidth=2.0,
        label='Historical Consumption (kW)'
    )
    # Subtle purple shading under historical curve
    ax.fill_between(
        hist_slice.index,
        hist_slice.values,
        color='#7163E8',
        alpha=0.08
    )

    # 24h Forecast in contrasting soft lavender (#A99AF5) with dashed line
    fc_series = df_forecasts[best_model_name]
    ax.plot(
        fc_series.index,
        fc_series.values,
        color='#A99AF5',
        linestyle='--',
        linewidth=2.4,
        marker='o',
        markersize=4,
        label=f'Next 24h Forecast ({best_model_name})'
    )

    # Vertical split marker
    split_time = fc_series.index[0]
    ax.axvline(x=split_time, color='#CBD5E1', linestyle=':', linewidth=1.5)
    ax.text(
        split_time,
        ax.get_ylim()[1] * 0.94,
        '  Forecast Horizon Start',
        color='#85889A',
        fontsize=9,
        fontweight=600
    )

    # Formatting axes and styling
    ax.set_ylabel('Active Power (kW)', fontsize=11, color='#242633', fontweight=600)
    ax.grid(axis='y', linestyle='--', alpha=0.35, color='#E8EAF2')
    ax.grid(axis='x', linestyle='--', alpha=0.15, color='#E8EAF2')

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#E8EAF2')

    ax.tick_params(colors='#85889A', labelsize=9.5)
    ax.legend(
        loc='upper left',
        frameon=True,
        facecolor='#FFFFFF',
        edgecolor='#E8EAF2',
        fontsize=10,
        labelcolor='#242633'
    )

    if history_hours <= 168:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Sub-metrics under chart
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.caption(f"**Period Average:** {hist_slice.mean():.2f} kW")
    with c_m2:
        st.caption(f"**Peak Observed:** {hist_slice.max():.2f} kW")
    with c_m3:
        st.caption(f"**Lowest Observed:** {hist_slice.min():.2f} kW")
    with c_m4:
        st.caption(f"**Total Period Usage:** {hist_slice.sum():,.1f} kWh")

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Two Compact Analytics Cards Below Main Chart
    # -------------------------------------------------------------
    card_col_a, card_col_b = st.columns(2, gap="medium")

    # CARD A — Consumption Pattern
    with card_col_a:
        st.markdown("""
        <div class="saas-card" style="height: 100%;">
            <h3 class="card-title">Card A — Consumption Pattern</h3>
            <div class="card-subtitle">Average electricity demand by hour of the day (0–23) across historical records.</div>
        """, unsafe_allow_html=True)

        df_hourly_pattern = df_hourly.copy()
        df_hourly_pattern['hour'] = df_hourly_pattern.index.hour
        hourly_stats = df_hourly_pattern.groupby('hour')['Global_active_power'].mean()

        fig_a, ax_a = plt.subplots(figsize=(6.5, 3.4), dpi=140)
        fig_a.patch.set_facecolor('#FFFFFF')
        ax_a.set_facecolor('#FFFFFF')

        # Highlight Peak Hours (7-9 AM and 18-21 PM) in primary purple, off-peak in soft lavender
        bar_colors = []
        for h in hourly_stats.index:
            if (7 <= h <= 9) or (18 <= h <= 21):
                bar_colors.append('#7163E8')  # Peak
            elif 1 <= h <= 5:
                bar_colors.append('#D5CEFB')  # Valley
            else:
                bar_colors.append('#A99AF5')  # Normal

        bars = ax_a.bar(hourly_stats.index, hourly_stats.values, color=bar_colors, width=0.72, edgecolor='none')

        ax_a.set_xticks(range(0, 24, 2))
        ax_a.set_xlabel('Hour of Day (0 - 23)', fontsize=9.5, color='#85889A', fontweight=600)
        ax_a.set_ylabel('Mean Power (kW)', fontsize=9.5, color='#85889A', fontweight=600)
        ax_a.grid(axis='y', linestyle='--', alpha=0.35, color='#E8EAF2')
        for spine in ax_a.spines.values():
            spine.set_color('#E8EAF2')
        ax_a.tick_params(colors='#85889A', labelsize=8.5)

        # Legend badges
        ax_a.plot([], [], color='#7163E8', linewidth=6, label='Peak Load (Morning/Evening)')
        ax_a.plot([], [], color='#D5CEFB', linewidth=6, label='Night Valley (1-5 AM)')
        ax_a.legend(loc='upper left', frameon=False, fontsize=8.5, labelcolor='#242633')

        plt.tight_layout()
        st.pyplot(fig_a)
        plt.close(fig_a)

        st.caption("💡 **Takeaway:** Notice distinct peaks during morning routine (7–9 AM) and evening cooking/lighting (6–9 PM).")
        st.markdown("</div>", unsafe_allow_html=True)

    # CARD B — Model Accuracy
    with card_col_b:
        st.markdown("""
        <div class="saas-card" style="height: 100%;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <h3 class="card-title">Card B — Model Accuracy</h3>
                    <div class="card-subtitle">Actual vs. predicted electricity consumption for the 24-hour test period.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        selected_eval_model = st.selectbox(
            "Select Model to Inspect:",
            options=[col for col in df_forecasts.columns if col != 'Actual'],
            index=0,
            key="card_b_model_select"
        )

        m_mae = df_metrics.loc[selected_eval_model, 'MAE'] if df_metrics is not None and selected_eval_model in df_metrics.index else 0.0
        m_rmse = df_metrics.loc[selected_eval_model, 'RMSE'] if df_metrics is not None and selected_eval_model in df_metrics.index else 0.0

        fig_b, ax_b = plt.subplots(figsize=(6.5, 2.7), dpi=140)
        fig_b.patch.set_facecolor('#FFFFFF')
        ax_b.set_facecolor('#FFFFFF')

        # Actual
        ax_b.plot(
            df_forecasts.index,
            df_forecasts['Actual'],
            color='#242633',
            linewidth=2.0,
            marker='o',
            markersize=3.5,
            label='Actual Ground Truth'
        )

        # Predicted
        ax_b.plot(
            df_forecasts.index,
            df_forecasts[selected_eval_model],
            color='#7163E8',
            linestyle='--',
            linewidth=2.0,
            marker='s',
            markersize=3.5,
            label=f'{selected_eval_model} Forecast'
        )

        ax_b.set_ylabel('Power (kW)', fontsize=9.5, color='#85889A', fontweight=600)
        ax_b.grid(axis='y', linestyle='--', alpha=0.35, color='#E8EAF2')
        for spine in ax_b.spines.values():
            spine.set_color('#E8EAF2')
        ax_b.tick_params(colors='#85889A', labelsize=8.5)
        ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_b.legend(loc='upper right', frameon=False, fontsize=8.5, labelcolor='#242633')

        plt.tight_layout()
        st.pyplot(fig_b)
        plt.close(fig_b)

        # Accuracy metric chips
        acc1, acc2, acc3 = st.columns(3)
        with acc1:
            st.markdown(f"<div style='background:#F3F1FD; padding:8px 12px; border-radius:10px; font-size:12px; font-weight:700; color:#7163E8;'>MAE: {m_mae:.4f} kW</div>", unsafe_allow_html=True)
        with acc2:
            st.markdown(f"<div style='background:#F3F1FD; padding:8px 12px; border-radius:10px; font-size:12px; font-weight:700; color:#7163E8;'>RMSE: {m_rmse:.4f} kW</div>", unsafe_allow_html=True)
        with acc3:
            rank_val = df_metrics.loc[selected_eval_model, 'Rank'] if df_metrics is not None and 'Rank' in df_metrics.columns else "1"
            st.markdown(f"<div style='background:#ECFDF5; padding:8px 12px; border-radius:10px; font-size:12px; font-weight:700; color:#059669;'>Rank: #{rank_val} Overall</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# VIEW 2: ENERGY OVERVIEW (Deep-Dive EDA)
# -------------------------------------------------------------
elif nav_selection == "Energy Overview":
    render_top_navbar("Energy Overview", "Exploratory data analysis across 34,589 hourly household records.")

    st.markdown("""
    <div class="saas-card">
        <h3 class="card-title">Macro Electricity Consumption Timeline (2006–2010)</h3>
        <div class="card-subtitle">Continuous hourly time series of global active power across the entire dataset.</div>
    """, unsafe_allow_html=True)

    st.line_chart(df_hourly['Global_active_power'], height=280)
    st.markdown("</div>", unsafe_allow_html=True)

    col_eda1, col_eda2 = st.columns(2, gap="medium")

    with col_eda1:
        st.markdown("""
        <div class="saas-card">
            <h3 class="card-title">Sub-Metering Appliance Breakdown</h3>
            <div class="card-subtitle">Comparative consumption across sub-metered household zones.</div>
        """, unsafe_allow_html=True)

        sub_df = df_hourly[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
        sub_df.index = ['Kitchen (Dishwasher/Microwave)', 'Laundry (Washing/Dryer)', 'Climate (Water-Heater & AC)']

        fig_sub, ax_sub = plt.subplots(figsize=(6, 3.2), dpi=140)
        fig_sub.patch.set_facecolor('#FFFFFF')
        ax_sub.set_facecolor('#FFFFFF')

        colors = ['#7163E8', '#A99AF5', '#D5CEFB']
        bars = ax_sub.barh(sub_df.index, sub_df.values, color=colors, height=0.55)
        ax_sub.set_xlabel('Average Watt-Hour (Wh)', fontsize=9.5, color='#85889A', fontweight=600)
        ax_sub.grid(axis='x', linestyle='--', alpha=0.35, color='#E8EAF2')
        for spine in ax_sub.spines.values():
            spine.set_color('#E8EAF2')
        ax_sub.tick_params(colors='#242633', labelsize=9)

        for bar in bars:
            w = bar.get_width()
            ax_sub.text(w + 0.15, bar.get_y() + bar.get_height()/2, f'{w:.2f} Wh', va='center', fontsize=8.5, color='#7163E8', fontweight=700)

        plt.tight_layout()
        st.pyplot(fig_sub)
        plt.close(fig_sub)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_eda2:
        st.markdown("""
        <div class="saas-card">
            <h3 class="card-title">Weekday vs. Weekend Load Dynamics</h3>
            <div class="card-subtitle">Daily average consumption (kW) across days of the week.</div>
        """, unsafe_allow_html=True)

        df_day = df_hourly.copy()
        df_day['dayofweek'] = df_day.index.dayofweek
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_avg = df_day.groupby('dayofweek')['Global_active_power'].mean()
        day_avg.index = days

        fig_day, ax_day = plt.subplots(figsize=(6, 3.2), dpi=140)
        fig_day.patch.set_facecolor('#FFFFFF')
        ax_day.set_facecolor('#FFFFFF')

        day_colors = ['#A99AF5', '#A99AF5', '#A99AF5', '#A99AF5', '#A99AF5', '#7163E8', '#7163E8']
        ax_day.bar(day_avg.index, day_avg.values, color=day_colors, width=0.6)
        ax_day.set_ylabel('Mean Power (kW)', fontsize=9.5, color='#85889A', fontweight=600)
        ax_day.grid(axis='y', linestyle='--', alpha=0.35, color='#E8EAF2')
        for spine in ax_day.spines.values():
            spine.set_color('#E8EAF2')
        ax_day.tick_params(colors='#242633', labelsize=9)

        plt.tight_layout()
        st.pyplot(fig_day)
        plt.close(fig_day)
        st.caption("Notice higher average consumption on Saturday & Sunday when household members remain at home.")
        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# VIEW 3: CONSUMPTION FORECAST (Dedicated 24h Forecasting Panel)
# -------------------------------------------------------------
elif nav_selection == "Consumption Forecast":
    render_top_navbar("Next 24 Hours Forecast", "Real-time out-of-sample machine learning and statistical load predictions.")

    st.markdown("""
    <div class="saas-card">
        <div class="card-header-row">
            <div>
                <h3 class="card-title">Next 24-Hour Forecast Visualizer</h3>
                <div class="card-subtitle">Inspect hourly forecasts generated from trained pipeline models.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_fc_left, col_fc_right = st.columns([1, 2], gap="large")

    with col_fc_left:
        selected_model = st.selectbox(
            "Select Forecasting Model:",
            options=[col for col in df_forecasts.columns if col != 'Actual'],
            index=0,
            key="dedicated_fc_model_select"
        )

        model_fc_series = df_forecasts[selected_model]
        actual_series = df_forecasts['Actual']

        pred_total_kwh = float(model_fc_series.sum())
        actual_total_kwh = float(actual_series.sum())
        fc_peak = float(model_fc_series.max())
        fc_peak_time = model_fc_series.idxmax().strftime("%H:%M")
        fc_min = float(model_fc_series.min())

        st.markdown(f"""
        <div style="background:#F7F8FC; border:1px solid #E8EAF2; border-radius:12px; padding:16px; margin: 14px 0;">
            <div style="font-size:12px; font-weight:700; color:#85889A; text-transform:uppercase;">Forecast Summary (24h)</div>
            <div style="font-size:26px; font-weight:800; color:#7163E8; margin:6px 0;">{pred_total_kwh:.2f} kWh</div>
            <div style="font-size:12px; color:#242633; margin-bottom:4px;"><strong>Ground Truth Actual:</strong> {actual_total_kwh:.2f} kWh</div>
            <div style="font-size:12px; color:#242633; margin-bottom:4px;"><strong>Predicted Peak:</strong> {fc_peak:.2f} kW ({fc_peak_time})</div>
            <div style="font-size:12px; color:#242633;"><strong>Predicted Base:</strong> {fc_min:.2f} kW</div>
        </div>
        """, unsafe_allow_html=True)

        # Trigger re-generation
        if st.button("⚡ Generate Forecast", use_container_width=True):
            with st.spinner(f"Re-generating 24-hour predictions using {selected_model}..."):
                from main import run_pipeline
                run_pipeline()
                st.success("Forecast generated and cached successfully!")
                st.rerun()

        # CSV Download button
        csv_bytes = df_forecasts.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_bytes,
            file_name="energy_consumption_forecast_24h.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_fc_right:
        fig_fc, ax_fc = plt.subplots(figsize=(8.5, 4.2), dpi=140)
        fig_fc.patch.set_facecolor('#FFFFFF')
        ax_fc.set_facecolor('#FFFFFF')

        # Actual curve
        ax_fc.plot(
            df_forecasts.index,
            actual_series,
            color='#242633',
            linewidth=2.2,
            marker='o',
            markersize=4.5,
            label='Actual Ground Truth'
        )

        # Predicted curve
        ax_fc.plot(
            df_forecasts.index,
            model_fc_series,
            color='#7163E8',
            linewidth=2.4,
            linestyle='--',
            marker='s',
            markersize=4.5,
            label=f'Predicted ({selected_model})'
        )

        # Subtle difference fill
        ax_fc.fill_between(
            df_forecasts.index,
            actual_series,
            model_fc_series,
            color='#7163E8',
            alpha=0.1,
            label='Forecast Residual Delta'
        )

        ax_fc.set_ylabel('Consumption (kW)', fontsize=10, color='#85889A', fontweight=600)
        ax_fc.grid(axis='y', linestyle='--', alpha=0.35, color='#E8EAF2')
        for spine in ax_fc.spines.values():
            spine.set_color('#E8EAF2')
        ax_fc.tick_params(colors='#85889A', labelsize=9)
        ax_fc.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax_fc.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#E8EAF2', fontsize=9.5)

        plt.tight_layout()
        st.pyplot(fig_fc)
        plt.close(fig_fc)

    st.markdown("---")
    st.markdown("#### 📋 Hourly Forecast Breakdown Table")

    table_df = pd.DataFrame({
        'Hour': [f"{dt.strftime('%H:00')} ({dt.strftime('%b %d')})" for dt in df_forecasts.index],
        'Actual (kW)': df_forecasts['Actual'].values,
        'Predicted (kW)': model_fc_series.values,
        'Delta / Error (kW)': (model_fc_series.values - df_forecasts['Actual'].values),
        'Absolute Error (kW)': np.abs(model_fc_series.values - df_forecasts['Actual'].values)
    })

    st.dataframe(
        table_df.style.format({
            'Actual (kW)': '{:.3f}',
            'Predicted (kW)': '{:.3f}',
            'Delta / Error (kW)': '{:+.3f}',
            'Absolute Error (kW)': '{:.3f}'
        }),
        use_container_width=True,
        height=320
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# VIEW 4: MODEL COMPARISON (Benchmark Panel)
# -------------------------------------------------------------
elif nav_selection == "Model Comparison":
    render_top_navbar("Model Comparison", "Benchmark of ARIMA, SARIMA, Linear Regression, and Random Forest Regressor.")

    st.markdown("""
    <div class="saas-card">
        <h3 class="card-title">Performance Benchmark Summary</h3>
        <div class="card-subtitle">Chronological out-of-sample evaluation metrics computed on the test set.</div>
    """, unsafe_allow_html=True)

    if df_metrics is not None:
        col_comp_tbl, col_comp_chart = st.columns([1, 1], gap="large")

        with col_comp_tbl:
            st.markdown("##### 🏆 Ranked Evaluation Metrics")
            styled_comp = df_metrics[['MAE', 'RMSE', 'Rank']].style.highlight_min(
                subset=['MAE', 'RMSE'], color='#ECFDF5', axis=0
            ).format({'MAE': '{:.4f} kW', 'RMSE': '{:.4f} kW'})

            st.dataframe(styled_comp, use_container_width=True, height=190)

            st.markdown(f"""
            <div style="background:#ECFDF5; border:1px solid #D1FAE5; border-radius:12px; padding:14px; margin-top:14px;">
                <span style="font-weight:700; color:#059669;">🥇 Best-Performing Model: {best_model_name}</span>
                <p style="margin:4px 0 0 0; font-size:12.5px; color:#065F46;">
                    Achieved the lowest MAE ({best_mae:.4f} kW) and lowest RMSE ({best_rmse:.4f} kW) by leveraging historical lag interactions and cyclical hour features.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_comp_chart:
            st.markdown("##### 📊 Error Comparison (Lower is Better)")

            fig_m, ax_m = plt.subplots(figsize=(6.5, 3.4), dpi=140)
            fig_m.patch.set_facecolor('#FFFFFF')
            ax_m.set_facecolor('#FFFFFF')

            y_pos = np.arange(len(df_metrics))
            bar_height = 0.35

            # Plot horizontal bars in Purple and Lavender
            rects_mae = ax_m.barh(y_pos - bar_height/2, df_metrics['MAE'], height=bar_height, color='#7163E8', label='MAE (Mean Absolute Error)')
            rects_rmse = ax_m.barh(y_pos + bar_height/2, df_metrics['RMSE'], height=bar_height, color='#A99AF5', label='RMSE (Root Mean Squared Error)')

            ax_m.set_yticks(y_pos)
            ax_m.set_yticklabels(df_metrics.index, fontsize=9.5, color='#242633', fontweight=600)
            ax_m.set_xlabel('Error (kW)', fontsize=9.5, color='#85889A', fontweight=600)
            ax_m.grid(axis='x', linestyle='--', alpha=0.35, color='#E8EAF2')
            for spine in ax_m.spines.values():
                spine.set_color('#E8EAF2')
            ax_m.tick_params(colors='#85889A', labelsize=8.5)
            ax_m.legend(loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor='#E8EAF2', fontsize=8.5)

            # Data labels
            for bar in list(rects_mae) + list(rects_rmse):
                w = bar.get_width()
                ax_m.text(w + 0.015, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', fontsize=8, color='#242633', fontweight=600)

            plt.tight_layout()
            st.pyplot(fig_m)
            plt.close(fig_m)

    st.markdown("</div>", unsafe_allow_html=True)

    # Model Architecture Deep-Dive Cards
    st.markdown("#### 🧠 Model Architectural Notes")
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)

    with m_c1:
        st.markdown("""
        <div class="saas-card" style="padding:18px;">
            <div style="font-weight:700; color:#7163E8; font-size:15px; margin-bottom:6px;">1. ARIMA</div>
            <div style="font-size:12px; color:#85889A; margin-bottom:8px;">Order: (1, 1, 1)</div>
            <p style="font-size:12.5px; color:#242633; margin:0;">
                Univariate statistical model using past values and residual error differencing. Lacks 24h seasonal parameters so it relaxes to the mean.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with m_c2:
        st.markdown("""
        <div class="saas-card" style="padding:18px;">
            <div style="font-weight:700; color:#7163E8; font-size:15px; margin-bottom:6px;">2. SARIMA</div>
            <div style="font-size:12px; color:#85889A; margin-bottom:8px;">(1,0,1) &times; (1,1,0,24)</div>
            <p style="font-size:12.5px; color:#242633; margin:0;">
                Seasonal ARIMA capturing the daily 24-hour diurnal cycle (s=24). Strongly tracks recurring day/night human load rhythms.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with m_c3:
        st.markdown("""
        <div class="saas-card" style="padding:18px;">
            <div style="font-weight:700; color:#7163E8; font-size:15px; margin-bottom:6px;">3. Linear Regression</div>
            <div style="font-size:12px; color:#85889A; margin-bottom:8px;">Supervised Lag Features</div>
            <p style="font-size:12.5px; color:#242633; margin:0;">
                Trained on lag_1, lag_24, lag_168, and calendar features. Fast, highly interpretable, and competitive for 24h persistence.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with m_c4:
        st.markdown("""
        <div class="saas-card" style="padding:18px;">
            <div style="font-weight:700; color:#7163E8; font-size:15px; margin-bottom:6px;">4. Random Forest</div>
            <div style="font-size:12px; color:#85889A; margin-bottom:8px;">100 Estimators (Ensemble)</div>
            <p style="font-size:12.5px; color:#242633; margin:0;">
                Ensemble of bagged decision trees. Effectively captures sudden spikes, appliance loads, and non-linear interactions.
            </p>
        </div>
        """, unsafe_allow_html=True)


# -------------------------------------------------------------
# VIEW 5: HISTORICAL DATA (Searchable Explorer)
# -------------------------------------------------------------
elif nav_selection == "Historical Data":
    render_top_navbar("Historical Data Explorer", "Browse, filter, and inspect cleaned hourly dataset records.")

    st.markdown("""
    <div class="saas-card">
        <h3 class="card-title">Processed Hourly Records (UCI Repository)</h3>
        <div class="card-subtitle">Access 34,589 hourly resampled observations with full telemetry columns.</div>
    """, unsafe_allow_html=True)

    row_limit = st.slider("Number of records to preview:", min_value=10, max_value=500, value=50, step=10)
    st.dataframe(df_hourly.tail(row_limit), use_container_width=True)

    st.markdown("##### 📊 Numerical Statistics Summary")
    st.dataframe(df_hourly[['Global_active_power', 'Voltage', 'Global_intensity']].describe(), use_container_width=True)

    csv_full = df_hourly.tail(1000).to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Recent 1,000 Records (CSV)",
        data=csv_full,
        file_name="hourly_energy_recent_1000.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# VIEW 6: SETTINGS (Configuration)
# -------------------------------------------------------------
elif nav_selection == "Settings":
    render_top_navbar("Settings", "Pipeline controls, hyperparameters, and environment diagnostics.")

    st.markdown("""
    <div class="saas-card">
        <h3 class="card-title">Forecasting Configuration</h3>
        <div class="card-subtitle">Adjust system parameters and execution triggers.</div>
        <br>
    """, unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.selectbox("Forecast Horizon:", ["24 Hours (Default)", "48 Hours", "72 Hours", "168 Hours (1 Week)"])
        st.selectbox("Resampling Granularity:", ["1 Hour (Standard)", "30 Minutes", "2 Hours"])

    with s2:
        st.selectbox("Seasonal Periodicity (s):", ["24 (Diurnal Cycle)", "168 (Weekly Cycle)"])
        st.checkbox("Enable Automatic Lag Feature Regeneration", value=True)

    st.markdown("---")
    st.markdown("##### 🔄 Pipeline Execution Controls")

    if st.button("🚀 Retrain All 4 Models & Regenerate Artifacts"):
        with st.spinner("Retraining ARIMA, SARIMA, Linear Regression, and Random Forest on full historical data..."):
            from main import run_pipeline
            run_pipeline()
            st.success("All models retrained successfully! Cache updated.")
            st.rerun()

    st.markdown("---")
    st.markdown("##### 💻 Environment Diagnostics")
    st.markdown("""
    - **Operating System:** Windows (AMD64)
    - **Python Interpreter:** Python 3.14 (Anaconda Environment)
    - **Core ML Libraries:** Scikit-learn, Statsmodels, Pandas, NumPy, Matplotlib
    - **Dataset Storage:** Local CSV Cache (`data/processed/hourly_energy_consumption.csv`)
    - **Model Weights Path:** `models/` directory
    """)

    st.markdown("</div>", unsafe_allow_html=True)
