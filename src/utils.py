"""
Utility functions for evaluation metrics, plotting, and file saving.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculates MAE and RMSE evaluation metrics.
    """
    # Align indices if needed
    common_idx = y_true.index.intersection(y_pred.index)
    actual = y_true.loc[common_idx]
    predicted = y_pred.loc[common_idx]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    return {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4)
    }


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Creates a summary comparison DataFrame from multiple model results.
    """
    df = pd.DataFrame(results).T
    df.index.name = 'Model'
    # Rank by RMSE (lower is better)
    df['Rank'] = df['RMSE'].rank(ascending=True).astype(int)
    return df.sort_values(by='Rank')


def plot_historical_consumption(
    df: pd.DataFrame,
    target_col: str = 'Global_active_power',
    hours: int = 720,
    save_path: str = None
):
    """
    Plots historical electricity consumption over a specified recent window.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 5))

    plot_data = df.iloc[-hours:] if len(df) > hours else df
    ax.plot(plot_data.index, plot_data[target_col], color='#1f77b4', linewidth=1.2, label='Hourly Consumption (kW)')

    ax.set_title(f'Historical Electricity Consumption (Last {hours // 24} Days)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestamp', fontsize=11)
    ax.set_ylabel('Global Active Power (kW)', fontsize=11)
    ax.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig


def plot_hourly_profile(
    df: pd.DataFrame,
    target_col: str = 'Global_active_power',
    save_path: str = None
):
    """
    Plots the average 24-hour diurnal consumption pattern across the day.
    """
    data = df.copy()
    data['hour'] = data.index.hour
    hourly_avg = data.groupby('hour')[target_col].agg(['mean', 'std'])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    hours = np.arange(24)
    ax.plot(hours, hourly_avg['mean'], color='#2ca02c', marker='o', linewidth=2, label='Average Consumption (kW)')
    ax.fill_between(
        hours,
        hourly_avg['mean'] - hourly_avg['std'],
        hourly_avg['mean'] + hourly_avg['std'],
        color='#2ca02c',
        alpha=0.18,
        label='± 1 Std Dev'
    )

    ax.set_xticks(hours)
    ax.set_title('Average Daily Electricity Consumption Profile by Hour', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day (0 - 23)', fontsize=11)
    ax.set_ylabel('Global Active Power (kW)', fontsize=11)
    ax.legend(loc='upper left')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig


def plot_actual_vs_predicted(
    y_true: pd.Series,
    predictions: Dict[str, pd.Series],
    title: str = '24-Hour Electricity Consumption: Actual vs. Predicted',
    save_path: str = None
):
    """
    Plots actual values against predictions from multiple models.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual
    ax.plot(y_true.index, y_true.values, 'k-', linewidth=2.5, label='Actual Consumption', marker='o')

    # Color palette for models
    palette = {
        'ARIMA': '#e377c2',
        'SARIMA': '#1f77b4',
        'Linear Regression': '#ff7f0e',
        'Random Forest': '#2ca02c'
    }

    # Plot each model's forecast
    for model_name, preds in predictions.items():
        color = palette.get(model_name, None)
        common_idx = y_true.index.intersection(preds.index)
        if len(common_idx) > 0:
            ax.plot(
                common_idx,
                preds.loc[common_idx].values,
                linestyle='--',
                linewidth=1.8,
                marker='s',
                markersize=4,
                color=color,
                label=f'{model_name}'
            )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestamp (Hourly)', fontsize=11)
    ax.set_ylabel('Global Active Power (kW)', fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    plt.xticks(rotation=25)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    save_path: str = None
):
    """
    Plots a side-by-side bar chart of MAE and RMSE across all models.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, metrics_df['MAE'], width, label='MAE (Mean Absolute Error)', color='#3498db')
    rects2 = ax.bar(x + width/2, metrics_df['RMSE'], width, label='RMSE (Root Mean Squared Error)', color='#e74c3c')

    ax.set_title('Model Performance Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Error (kW)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index, fontsize=11)
    ax.legend()

    # Add numeric labels on top of bars
    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig


def save_forecasts_to_csv(forecast_df: pd.DataFrame, filepath: str):
    """Saves forecasts DataFrame to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    forecast_df.to_csv(filepath)
    print(f"[Utils] Forecasts saved to: {filepath}")
