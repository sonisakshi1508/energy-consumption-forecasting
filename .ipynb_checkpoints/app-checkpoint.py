"""
Interactive Streamlit Dashboard for Energy Consumption Forecasting.
Displays:
1. Executive KPI Summary
2. Exploratory Data Analysis (EDA) & Historical Trends
3. Model Comparison (ARIMA, SARIMA, Linear Regression, Random Forest)
4. 24-Hour Electricity Consumption Forecasts vs. Actuals
5. Feature Importance & Consumption Drivers
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project root path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
DATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'hourly_energy_consumption.csv')
FORECAST_CSV = os.path.join(OUTPUTS_DIR, 'forecasts_24h.csv')
COMPARISON_CSV = os.path.join(OUTPUTS_DIR, 'model_comparison.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


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


# Sidebar
st.sidebar.title("⚡ Energy Forecast AI")
st.sidebar.markdown(
    "**UCI Household Electric Power Consumption**  \n"
    "Predicting the next 24-hour electricity usage with statistical and machine learning models."
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "📈 Executive Dashboard",
        "📊 Exploratory Data Analysis (EDA)",
        "🤖 Model Comparison & Metrics",
        "🔮 24-Hour Forecast vs Actuals",
        "🔍 Feature Importance & Drivers",
        "ℹ️ About the Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Machine Learning Lab • Python 3.14")

# Load Cached Data
df_hourly = load_data()
df_forecasts = load_forecasts()
df_metrics = load_comparison()

# If data or forecasts are not yet ready
if df_hourly is None or df_forecasts is None:
    st.warning("⚠️ Processed dataset or forecasts not found. Please run the training pipeline first.")
    if st.button("🚀 Run Training Pipeline Now"):
        with st.spinner("Executing pipeline (loading dataset, training 4 models, generating forecasts)..."):
            from main import run_pipeline
            run_pipeline()
            st.success("Pipeline executed successfully! Please refresh or click below.")
            st.rerun()
    st.stop()


# -------------------------------------------------------------
# 1. Executive Dashboard
# -------------------------------------------------------------
if menu == "📈 Executive Dashboard":
    st.title("⚡ Energy Consumption Forecasting Dashboard")
    st.markdown("Overview of electricity usage, model benchmarks, and next 24-hour forecast.")

    # KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    total_records = len(df_hourly)
    avg_power = df_hourly['Global_active_power'].mean()
    max_power = df_hourly['Global_active_power'].max()
    best_model = df_metrics.index[0] if df_metrics is not None else "N/A"
    best_rmse = df_metrics.iloc[0]['RMSE'] if df_metrics is not None else 0.0

    with kpi1:
        st.metric("Total Hourly Records", f"{total_records:,}")
    with kpi2:
        st.metric("Average Power Usage", f"{avg_power:.2f} kW")
    with kpi3:
        st.metric("Peak Power Usage", f"{max_power:.2f} kW")
    with kpi4:
        st.metric("Top Performing Model", f"{best_model}", f"RMSE: {best_rmse:.3f} kW")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🔮 24-Hour Forecast Snapshot")
        st.line_chart(df_forecasts)

    with col_right:
        st.subheader("🏆 Model Benchmark Summary")
        if df_metrics is not None:
            st.dataframe(df_metrics[['MAE', 'RMSE', 'Rank']], use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(df_metrics.index, df_metrics['RMSE'], color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'][:len(df_metrics)])
            ax.set_xlabel('RMSE (Lower is Better)')
            ax.set_title('Model RMSE Comparison')
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', fontsize=9)
            st.pyplot(fig)


# -------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# -------------------------------------------------------------
elif menu == "📊 Exploratory Data Analysis (EDA)":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Detailed breakdown of temporal electricity consumption dynamics.")

    time_window = st.selectbox(
        "Select Time Window to Inspect:",
        ["Last 7 Days (168 Hours)", "Last 30 Days (720 Hours)", "Last 90 Days", "Full Historical Timeline"]
    )

    hours_map = {
        "Last 7 Days (168 Hours)": 168,
        "Last 30 Days (720 Hours)": 720,
        "Last 90 Days": 2160,
        "Full Historical Timeline": len(df_hourly)
    }
    selected_hours = hours_map[time_window]
    plot_df = df_hourly.iloc[-selected_hours:]

    st.subheader(f"⚡ Hourly Electricity Consumption ({time_window})")
    st.line_chart(plot_df['Global_active_power'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⏰ 24-Hour Diurnal Pattern")
        df_copy = df_hourly.copy()
        df_copy['hour'] = df_copy.index.hour
        hourly_stats = df_copy.groupby('hour')['Global_active_power'].agg(['mean', 'std'])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hourly_stats.index, hourly_stats['mean'], marker='o', color='#2ca02c', linewidth=2, label='Mean Power (kW)')
        ax.fill_between(
            hourly_stats.index,
            hourly_stats['mean'] - hourly_stats['std'],
            hourly_stats['mean'] + hourly_stats['std'],
            color='#2ca02c', alpha=0.15, label='± 1 Std Dev'
        )
        ax.set_xlabel("Hour of Day (0 to 23)")
        ax.set_ylabel("Global Active Power (kW)")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)
        st.caption("Notice the distinct peaks in morning (7-9 AM) and evening (7-9 PM).")

    with col2:
        st.subheader("📅 Day-of-Week Variation")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        df_copy['dayofweek'] = df_copy.index.dayofweek
        weekly_stats = df_copy.groupby('dayofweek')['Global_active_power'].mean()
        weekly_stats.index = [days[i] for i in weekly_stats.index]

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.bar(weekly_stats.index, weekly_stats.values, color='#3498db')
        ax2.set_ylabel("Average Active Power (kW)")
        ax2.set_xlabel("Day of Week")
        ax2.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig2)
        st.caption("Comparison between weekday and weekend energy utilization.")


# -------------------------------------------------------------
# 3. Model Comparison & Metrics
# -------------------------------------------------------------
elif menu == "🤖 Model Comparison & Metrics":
    st.title("🤖 Model Training & Performance Comparison")
    st.markdown("Rigorous benchmark of statistical and machine learning models using chronological train-test splitting.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Evaluation Metrics Table")
        st.dataframe(
            df_metrics.style.highlight_min(subset=['MAE', 'RMSE'], color='#c8e6c9', axis=0),
            use_container_width=True
        )

    with col2:
        st.subheader("📊 Error Comparison (MAE vs RMSE)")
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(df_metrics))
        width = 0.35

        ax.bar(x - width/2, df_metrics['MAE'], width, label='MAE', color='#3498db')
        ax.bar(x + width/2, df_metrics['RMSE'], width, label='RMSE', color='#e74c3c')
        ax.set_xticks(x)
        ax.set_xticklabels(df_metrics.index, rotation=15)
        ax.set_ylabel("Error (kW)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("💡 Model Overview & Architecture")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        with st.expander("1. ARIMA (Autoregressive Integrated Moving Average)", expanded=True):
            st.write("""
            - **Category**: Classical statistical time series model.
            - **Mechanism**: Models a point as a linear function of past observations (AR) and past white noise errors (MA), with differencing (I) for stationarity.
            - **Strength**: Quick baseline for linear autocorrelations.
            """)
        with st.expander("2. SARIMA (Seasonal ARIMA)", expanded=True):
            st.write("""
            - **Category**: Seasonal statistical time series model.
            - **Mechanism**: Incorporates seasonal autoregression, differencing, and moving averages with period `s=24`.
            - **Strength**: Captures recurring daily 24-hour human activity cycles.
            """)

    with m_col2:
        with st.expander("3. Linear Regression", expanded=True):
            st.write("""
            - **Category**: Supervised Machine Learning.
            - **Features**: Lag terms (`lag_1, lag_2, lag_3, lag_24, lag_48, lag_168`), rolling means (6h, 24h), and calendar variables.
            - **Strength**: Fast, interpretable, weights recent history directly.
            """)
        with st.expander("4. Random Forest Regressor", expanded=True):
            st.write("""
            - **Category**: Ensemble Machine Learning (Bagging Decision Trees).
            - **Features**: Historical consumption lags, rolling volatility, cyclical sine/cosine hour encodings.
            - **Strength**: Captures complex non-linear consumption peaks and feature interactions.
            """)


# -------------------------------------------------------------
# 4. 24-Hour Forecast vs Actuals
# -------------------------------------------------------------
elif menu == "🔮 24-Hour Forecast vs Actuals":
    st.title("🔮 24-Hour Electricity Consumption Forecast")
    st.markdown("Side-by-side comparison between ground truth actual consumption and model predictions.")

    models_to_display = st.multiselect(
        "Select Models to Display on Chart:",
        options=[col for col in df_forecasts.columns if col != 'Actual'],
        default=[col for col in df_forecasts.columns if col != 'Actual']
    )

    chart_cols = ['Actual'] + models_to_display
    st.line_chart(df_forecasts[chart_cols])

    st.subheader("📑 Forecast Data Table (Hourly Observations)")
    st.dataframe(df_forecasts[chart_cols].style.format("{:.3f}"), use_container_width=True)

    csv_data = df_forecasts.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download 24-Hour Forecasts as CSV",
        data=csv_data,
        file_name="energy_forecasts_24h.csv",
        mime="text/csv"
    )


# -------------------------------------------------------------
# 5. Feature Importance & Drivers
# -------------------------------------------------------------
elif menu == "🔍 Feature Importance & Drivers":
    st.title("🔍 Feature Importance & Consumption Drivers")
    st.markdown("Analysis of which historical and calendar features most heavily drive electricity consumption predictions.")

    import joblib
    rf_model_path = os.path.join(MODELS_DIR, 'random_forest.joblib')

    if os.path.exists(rf_model_path):
        saved = joblib.load(rf_model_path)
        rf_model = saved['model']
        feature_names = saved['features']

        importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        importances.tail(12).plot(kind='barh', ax=ax, color='#1f77b4')
        ax.set_title("Top 12 Most Influential Features (Random Forest)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Relative Importance Score", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

        st.info("💡 **Insight**: Notice that immediate past demand (`lag_1`) and same-time-yesterday demand (`lag_24`) are typically the strongest predictors, followed by the hour of the day.")
    else:
        st.info("Random Forest model file not found. Run the pipeline to view feature importance.")


# -------------------------------------------------------------
# 6. About the Project
# -------------------------------------------------------------
elif menu == "ℹ️ About the Project":
    st.title("ℹ️ About the Project")
    st.markdown("""
    ### **Energy Consumption Forecasting Using Time-Series Machine Learning Models**

    #### 1. Objective
    To forecast household electricity usage for the next 24 hours by benchmarking classical statistical models against modern machine learning algorithms.

    #### 2. Dataset
    - **Dataset**: UCI Individual Household Electric Power Consumption dataset.
    - **Original records**: Over 2,075,000 minute-level observations across 47 months.
    - **Resampling**: Aggregated to continuous hourly average kilowatt (kW) usage.

    #### 3. Models Evaluated
    1. **ARIMA (1, 1, 1)**
    2. **SARIMA (1, 0, 1) x (1, 1, 0, 24)**
    3. **Linear Regression with Lags & Calendar Features**
    4. **Random Forest Regressor (100 estimators)**

    #### 4. Evaluation Metrics
    - **MAE (Mean Absolute Error)**: Average absolute magnitude of forecast errors in kW.
    - **RMSE (Root Mean Squared Error)**: Quadratic metric penalizing larger forecast deviations.
    """)
