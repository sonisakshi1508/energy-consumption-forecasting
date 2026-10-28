"""
Feature engineering module for time-series machine learning models.
Extracts calendar features, lag features, and rolling statistics from hourly data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def create_time_series_features(
    df: pd.DataFrame,
    target_col: str = 'Global_active_power',
    lags: List[int] = [1, 2, 3, 24, 48, 168],
    rolling_windows: List[int] = [6, 24]
) -> pd.DataFrame:
    """
    Creates time-series features for machine learning models:
    - Calendar features (hour, day of week, month, weekend indicator, cyclic encodings)
    - Lag features (historical consumption at t-1, t-2, t-3, t-24, t-48, t-168)
    - Rolling window statistics (mean, std over 6h and 24h)

    Parameters:
        df: Processed DataFrame indexed by Datetime.
        target_col: Name of the target variable to forecast.
        lags: List of lag intervals (in hours).
        rolling_windows: Windows for rolling aggregations.

    Returns:
        pd.DataFrame: DataFrame containing original data + engineered features,
                      with NaNs from lags dropped.
    """
    data = df.copy()

    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # 1. Calendar / Datetime Features
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    data['quarter'] = data.index.quarter
    data['month'] = data.index.month
    data['dayofyear'] = data.index.dayofyear
    data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)

    # Cyclical encodings for hour (preserves hour 23 -> hour 0 continuity)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24.0)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24.0)

    # 2. Historical Lag Features
    for lag in lags:
        data[f'lag_{lag}'] = data[target_col].shift(lag)

    # 3. Rolling Window Statistics (computed only on past data using shift(1) to avoid lookahead leakage)
    for window in rolling_windows:
        data[f'rolling_mean_{window}h'] = data[target_col].shift(1).rolling(window=window).mean()
        data[f'rolling_std_{window}h'] = data[target_col].shift(1).rolling(window=window).std()

    # Drop NaNs created by lag and rolling window operations
    clean_data = data.dropna()

    return clean_data


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: int = 168
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits time-series data chronologically without shuffling.

    Parameters:
        df: DataFrame with DatetimeIndex.
        test_size: Number of most recent hourly observations to reserve for the test set.
                   Defaults to 168 (1 full week), from which 24-hour evaluation and forecasts
                   can be extracted.

    Returns:
        train_df, test_df
    """
    if len(df) <= test_size:
        raise ValueError(f"Dataset length ({len(df)}) is too small for test_size ({test_size})")

    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()

    return train_df, test_df


def prepare_ml_data(
    df_features: pd.DataFrame,
    target_col: str = 'Global_active_power',
    test_size: int = 168
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Prepares X_train, y_train, X_test, y_test for ML models (Linear Regression, Random Forest).

    Returns:
        X_train, y_train, X_test, y_test, feature_names
    """
    train_df, test_df = chronological_train_test_split(df_features, test_size=test_size)

    # Exclude other raw sub-metering / target columns from feature matrix
    exclude_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, y_train, X_test, y_test, feature_cols
