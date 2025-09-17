"""
==================================================================
FORECASTING MODELS MODULE
------------------------------------------------------------------
This module provides a unified interface for initializing, training, 
and generating forecasts using multiple time series models. It supports 
both traditional statistical methods and machine learning approaches, 
enabling flexible experimentation and benchmarking across use cases.

It is designed to work on multi-series retail demand forecasting, 
scalable from SKU-level to higher hierarchies (Brand, Store, Region, 
System) with hierarchical reconciliation to ensure aggregate 
consistency.

✔️ Supports multiple models: ARIMA, ETS, Prophet, Linear Regression…  
✔️ Scales across thousands of time series in parallel via StatsForecast  
✔️ Enables hierarchical reconciliation for consistent bottom-up totals  

⚠️ This module is strictly limited to *model training and prediction*.  
It does **not** handle data ingestion, preprocessing, feature 
engineering, or visualization — those are managed in separate modules.
==================================================================
"""
import pandas as pd
import numpy as np

# StatsForecast modules
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, 
    ETS, 
    SeasonalNaive, 
    CrostonOptimized, 
    IMAPA, 
    BATS, 
    TBATS
)

# Hierarchical Forecasting
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace

# Prophet
from prophet import Prophet

# Scikit-learn ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Gradient Boosting ML
import xgboost as xgb
import lightgbm as lgb

# 1. FORECASTING MODELS
class ForecastingModels:
    def __init__(self, freq="D", horizon=30):
        """
        freq: frequency of data (D = daily, M = monthly, etc.)
        horizon: forecast horizon
        """
        self.freq = freq
        self.horizon = horizon

    # -------------------
    # 1.1. Train StatsForecast models across multiple time series in parallel
    # -------------------       
    def fit_statsforecast(self, df, models=None, group_cols=None):
        """
        .

        df: DataFrame [unique_id, ds, y]
        models: list of StatsForecast models
        group_cols: hierarchy levels for reconciliation
        """
        if models is None:
            models = [
                AutoARIMA(season_length=7), 
                ETS(season_length=7), 
                SeasonalNaive(season_length=7)
            ]

        sf = StatsForecast(
            models=models,
            freq=self.freq,
            n_jobs=-1
        )

        fitted = sf.fit(df)
        forecasts = sf.predict(self.horizon)

        if group_cols:
            hier = HierarchicalReconciliation(method=MinTrace(method='ols'))
            forecasts = hier.reconcile(forecasts, df, group_cols)

        return forecasts

    # -------------------
    # Prophet
    # -------------------
    def fit_prophet(self, df):
        """
        Prophet model for one series.
        For multi-series, loop through each SKU/store.
        """
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df.rename(columns={"ds": "ds", "y": "y"}))
        future = m.make_future_dataframe(periods=self.horizon, freq=self.freq)
        forecast = m.predict(future)
        return forecast

    # -------------------
    # Linear Regression
    # -------------------
    def fit_linear_regression(self, df):
        """
        Simple Linear Regression baseline.
        """
        df = df.copy()
        df["t"] = np.arange(len(df))
        X, y = df[["t"]], df["y"]

        model = LinearRegression()
        model.fit(X, y)

        future_t = np.arange(len(df), len(df) + self.horizon)
        future_pred = model.predict(future_t.reshape(-1, 1))

        forecast = pd.DataFrame({
            "ds": pd.date_range(start=df["ds"].iloc[-1], periods=self.horizon+1, freq=self.freq)[1:],
            "yhat": future_pred
        })
        return forecast

    # -------------------
    # XGBoost
    # -------------------
    def fit_xgboost(self, df, lags=12):
        """
        XGBoost with lag features.
        """
        df = df.copy()
        for lag in range(1, lags+1):
            df[f"lag_{lag}"] = df["y"].shift(lag)
        df = df.dropna()

        X, y = df.drop(columns=["ds", "y"]), df["y"]

        model = xgb.XGBRegressor(objective="reg:squarederror")
        model.fit(X, y)

        last_row = df.iloc[-1:].drop(columns=["ds", "y"])
        preds = []
        for _ in range(self.horizon):
            pred = model.predict(last_row)[0]
            preds.append(pred)
            # shift lag features manually
            last_row = pd.DataFrame([np.append([pred], last_row.values[0][:-1])], columns=last_row.columns)

        forecast = pd.DataFrame({
            "ds": pd.date_range(start=df["ds"].iloc[-1], periods=self.horizon+1, freq=self.freq)[1:],
            "yhat": preds
        })
        return forecast

    # -------------------
    # LightGBM
    # -------------------
    def fit_lightgbm(self, df, lags=12):
        """
        LightGBM with lag features.
        """
        df = df.copy()
        for lag in range(1, lags+1):
            df[f"lag_{lag}"] = df["y"].shift(lag)
        df = df.dropna()

        X, y = df.drop(columns=["ds", "y"]), df["y"]

        model = lgb.LGBMRegressor()
        model.fit(X, y)

        last_row = df.iloc[-1:].drop(columns=["ds", "y"])
        preds = []
        for _ in range(self.horizon):
            pred = model.predict(last_row)[0]
            preds.append(pred)
            last_row = pd.DataFrame([np.append([pred], last_row.values[0][:-1])], columns=last_row.columns)

        forecast = pd.DataFrame({
            "ds": pd.date_range(start=df["ds"].iloc[-1], periods=self.horizon+1, freq=self.freq)[1:],
            "yhat": preds
        })
        return forecast