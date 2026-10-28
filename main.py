"""
Main execution pipeline for Energy Consumption Forecasting.
Executes end-to-end:
1. Data loading and hourly resampling.
2. Exploratory data analysis (EDA) and visualization generation.
3. Feature engineering (lags, rolling stats, calendar features).
4. Chronological train-test split.
5. Training the 4 models: ARIMA, SARIMA, Linear Regression, Random Forest.
6. Model evaluation (MAE and RMSE).
7. 24-hour ahead electricity consumption forecasting.
8. Artifact saving (forecast CSV, model files, evaluation plots).
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_hourly_data
from src.features import create_time_series_features, prepare_ml_data
from src.models import (
    ARIMAForecaster,
    SARIMAForecaster,
    LinearRegressionForecaster,
    RandomForestForecaster
)
from src.utils import (
    calculate_metrics,
    compare_models,
    plot_historical_consumption,
    plot_hourly_profile,
    plot_actual_vs_predicted,
    plot_metrics_comparison,
    save_forecasts_to_csv
)


def run_pipeline(forecast_horizon: int = 24):
    print("=" * 70)
    print("[*] ENERGY CONSUMPTION FORECASTING PIPELINE")
    print("=" * 70)

    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Load and Preprocess Hourly Data
    # -------------------------------------------------------------
    print("\n[Step 1/6] Loading and preprocessing dataset...")
    df_hourly = load_hourly_data()
    print(f" [OK] Dataset successfully loaded with {len(df_hourly):,} hourly records.")
    print(f" [OK] Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")
    print(f" [OK] Target variable: 'Global_active_power' (mean: {df_hourly['Global_active_power'].mean():.2f} kW)")

    # -------------------------------------------------------------
    # Step 2: Exploratory Data Visualizations
    # -------------------------------------------------------------
    print("\n[Step 2/6] Generating EDA visualizations...")
    hist_plot_path = os.path.join(output_dir, 'historical_consumption.png')
    plot_historical_consumption(df_hourly, hours=720, save_path=hist_plot_path)
    print(f" [OK] Saved: {hist_plot_path}")

    profile_plot_path = os.path.join(output_dir, 'hourly_consumption_profile.png')
    plot_hourly_profile(df_hourly, save_path=profile_plot_path)
    print(f" [OK] Saved: {profile_plot_path}")

    # -------------------------------------------------------------
    # Step 3: Feature Engineering & Chronological Split
    # -------------------------------------------------------------
    print("\n[Step 3/6] Engineering time-series features (lags, rolling stats, calendar)...")
    df_features = create_time_series_features(
        df_hourly,
        target_col='Global_active_power',
        lags=[1, 2, 3, 24, 48, 168],
        rolling_windows=[6, 24]
    )
    print(f" [OK] Feature matrix created with {df_features.shape[1]} columns across {len(df_features):,} records.")

    print(f"\n[*] Chronologically splitting data (reserving final {forecast_horizon} hours for testing)...")
    X_train, y_train, X_test, y_test, feature_cols = prepare_ml_data(
        df_features,
        target_col='Global_active_power',
        test_size=forecast_horizon
    )
    print(f" [OK] Training samples: {len(X_train):,} hours")
    print(f" [OK] Testing horizon:  {len(X_test):,} hours ({y_test.index[0]} to {y_test.index[-1]})")

    # Time series target for statistical models
    ts_train = df_hourly['Global_active_power'].iloc[:-forecast_horizon]
    ts_test = df_hourly['Global_active_power'].iloc[-forecast_horizon:]

    # Dictionary to store forecasts and metrics
    forecasts = {}
    metrics = {}

    # -------------------------------------------------------------
    # Step 4: Train and Evaluate Models
    # -------------------------------------------------------------
    print("\n[Step 4/6] Training and forecasting with 4 models...")

    # Model 1: ARIMA
    print("\n--- Model 1: ARIMA (AutoRegressive Integrated Moving Average) ---")
    arima = ARIMAForecaster(order=(1, 1, 1))
    arima.fit(ts_train, max_samples=1440)
    arima_preds = arima.predict(steps=forecast_horizon)
    arima_preds.index = ts_test.index
    forecasts['ARIMA'] = arima_preds
    metrics['ARIMA'] = calculate_metrics(ts_test, arima_preds)
    arima.save(os.path.join(models_dir, 'arima_model.pkl'))
    print(f" ARIMA  -> MAE: {metrics['ARIMA']['MAE']:.4f} kW | RMSE: {metrics['ARIMA']['RMSE']:.4f} kW")

    # Model 2: SARIMA
    print("\n--- Model 2: SARIMA (Seasonal ARIMA with 24-Hour Periodicity) ---")
    sarima = SARIMAForecaster(order=(1, 0, 1), seasonal_order=(1, 1, 0, 24))
    sarima.fit(ts_train, max_samples=720)
    sarima_preds = sarima.predict(steps=forecast_horizon)
    sarima_preds.index = ts_test.index
    forecasts['SARIMA'] = sarima_preds
    metrics['SARIMA'] = calculate_metrics(ts_test, sarima_preds)
    sarima.save(os.path.join(models_dir, 'sarima_model.pkl'))
    print(f" SARIMA -> MAE: {metrics['SARIMA']['MAE']:.4f} kW | RMSE: {metrics['SARIMA']['RMSE']:.4f} kW")

    # Model 3: Linear Regression
    print("\n--- Model 3: Linear Regression (Lag & Calendar Features) ---")
    lr = LinearRegressionForecaster()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    forecasts['Linear Regression'] = lr_preds
    metrics['Linear Regression'] = calculate_metrics(y_test, lr_preds)
    lr.save(os.path.join(models_dir, 'linear_regression.joblib'))
    print(f" Linear Regression -> MAE: {metrics['Linear Regression']['MAE']:.4f} kW | RMSE: {metrics['Linear Regression']['RMSE']:.4f} kW")

    # Model 4: Random Forest Regressor
    print("\n--- Model 4: Random Forest Regressor (Non-Linear Historical Patterns) ---")
    rf = RandomForestForecaster(n_estimators=100, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    forecasts['Random Forest'] = rf_preds
    metrics['Random Forest'] = calculate_metrics(y_test, rf_preds)
    rf.save(os.path.join(models_dir, 'random_forest.joblib'))
    print(f" Random Forest     -> MAE: {metrics['Random Forest']['MAE']:.4f} kW | RMSE: {metrics['Random Forest']['RMSE']:.4f} kW")

    # -------------------------------------------------------------
    # Step 5: Compare Models and Generate Forecast CSV
    # -------------------------------------------------------------
    print("\n[Step 5/6] Model Performance Comparison:")
    comparison_df = compare_models(metrics)
    print(comparison_df.to_string())

    comparison_csv_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_csv_path)
    print(f"\n [OK] Model comparison metrics saved to: {comparison_csv_path}")

    # Create 24-hour Forecast Comparison Table
    forecast_df = pd.DataFrame({'Actual': ts_test})
    for m_name, preds in forecasts.items():
        forecast_df[m_name] = preds
    forecast_df.index.name = 'Datetime'

    forecast_csv_path = os.path.join(output_dir, 'forecasts_24h.csv')
    save_forecasts_to_csv(forecast_df, forecast_csv_path)

    # -------------------------------------------------------------
    # Step 6: Evaluation Visualizations
    # -------------------------------------------------------------
    print("\n[Step 6/6] Generating evaluation plots...")

    # Actual vs Predicted Plot
    actual_pred_plot_path = os.path.join(output_dir, 'actual_vs_predicted_24h.png')
    plot_actual_vs_predicted(
        ts_test,
        forecasts,
        title='Next 24-Hour Electricity Consumption: Actual vs. Model Forecasts',
        save_path=actual_pred_plot_path
    )
    print(f" [OK] Saved: {actual_pred_plot_path}")

    # Metrics comparison bar chart
    metrics_plot_path = os.path.join(output_dir, 'model_comparison.png')
    plot_metrics_comparison(comparison_df, save_path=metrics_plot_path)
    print(f" [OK] Saved: {metrics_plot_path}")

    print("\n" + "=" * 70)
    print(" [OK] Pipeline execution finished successfully!")
    print(f" All outputs and figures saved in: {output_dir}")
    print(f" All trained models saved in:      {models_dir}")
    print("=" * 70)

    return comparison_df, forecast_df


if __name__ == '__main__':
    run_pipeline()
