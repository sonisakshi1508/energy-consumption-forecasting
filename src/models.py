"""
Forecasting models module.
Implements:
1. ARIMA (Autoregressive Integrated Moving Average)
2. SARIMA (Seasonal Autoregressive Integrated Moving Average)
3. Linear Regression (Lag & Calendar features)
4. Random Forest Regressor (Historical consumption & non-linear patterns)
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAForecaster:
    """ARIMA statistical time-series forecasting model."""

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, y_train: pd.Series, max_samples: int = 1440):
        """
        Fits ARIMA model on the most recent observations for computational efficiency.
        """
        # Use recent window (e.g. up to 1440 hours / 60 days) to balance stability and speed
        series = y_train.iloc[-max_samples:] if len(y_train) > max_samples else y_train
        self.model = ARIMA(series, order=self.order)
        self.fitted_model = self.model.fit()
        return self

    def predict(self, steps: int = 24) -> pd.Series:
        """Forecasts out-of-sample steps."""
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted yet.")
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

    def save(self, filepath: str):
        joblib.dump(self.fitted_model, filepath)

    def load(self, filepath: str):
        self.fitted_model = joblib.load(filepath)
        return self


class SARIMAForecaster:
    """SARIMA seasonal time-series forecasting model."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 24)
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None

    def fit(self, y_train: pd.Series, max_samples: int = 720):
        """
        Fits SARIMA model on the most recent observations.
        Using max_samples=720 (30 days of hourly data) ensures quick convergence
        while fully capturing the 24-hour diurnal seasonality.
        """
        series = y_train.iloc[-max_samples:] if len(y_train) > max_samples else y_train
        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model = self.model.fit(disp=False, maxiter=50)
        return self

    def predict(self, steps: int = 24) -> pd.Series:
        """Forecasts out-of-sample steps."""
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted yet.")
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

    def save(self, filepath: str):
        joblib.dump(self.fitted_model, filepath)

    def load(self, filepath: str):
        self.fitted_model = joblib.load(filepath)
        return self


class LinearRegressionForecaster:
    """Linear Regression forecasting model using lag and calendar features."""

    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X_test[self.feature_names])
        # Power consumption cannot be negative
        preds = np.clip(preds, a_min=0, a_max=None)
        return pd.Series(preds, index=X_test.index)

    def save(self, filepath: str):
        joblib.dump({'model': self.model, 'features': self.feature_names}, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['features']
        return self


class RandomForestForecaster:
    """Random Forest Regressor forecasting model using historical consumption features."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 15, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(X_test[self.feature_names])
        preds = np.clip(preds, a_min=0, a_max=None)
        return pd.Series(preds, index=X_test.index)

    def get_feature_importances(self) -> pd.Series:
        """Returns sorted feature importances."""
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model is not fitted yet.")
        importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
        return importances.sort_values(ascending=False)

    def save(self, filepath: str):
        joblib.dump({'model': self.model, 'features': self.feature_names}, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['features']
        return self
