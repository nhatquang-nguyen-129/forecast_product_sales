"""
==================================================================
ARIMA FORECASTING MODEL
------------------------------------------------------------------
Wrapper for ARIMA/SARIMA forecasting with support for:
- Exogenous features (multivariate regression with time series)
- Multiple cohorts (fit one model per unique_id)

✔️ Supports both ARIMA(p,d,q) and seasonal ARIMA(P,D,Q,s)
✔️ Handles multi-series forecasting in a unified way
✔️ Provides methods consistent with the BaseForecastModel interface

⚠️ Each time series is modeled independently (no cross-series learning).
==================================================================
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from sktime.forecasting.arima import ARIMA
from sktime.forecassting.base import BaseForecaster, ForecastingHorizon

from src.models.base.model import BaseForecastModel

# 1. WRAPPER FOR ARIMA FORECASTING MODEL WITH EXOGENOUS FEATURES AND MULTIPLE COHORTS
class ARIMAModel(BaseForecastModel):
    """
    ----------
    order : tuple, default=(1,1,1)
        ARIMA (p,d,q) parameters
    seasonal_order : tuple, default=(0,0,0,0)
        Seasonal ARIMA (P,D,Q,s)
    """

# 1.1. Initialize ARIMA model
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_lookup: Dict[str, Dict[str, Any]] = {}
        self.fitted_models: Dict[str, BaseForecaster] = {}

# 1.2. Fit ARIMA model for each time series in the DataFrame
    def fit(self, df: pd.DataFrame, id_col: str = "unique_id", y_col: str = "y", 
            ds_col: str = "ds", exog_cols: Optional[list] = None, fh: Optional[int] = None):

        for uid, group in df.groupby(id_col):
            y = pd.Series(group[y_col].values, index=pd.to_datetime(group[ds_col]))
            exog = group[exog_cols] if exog_cols else None

            model = ARIMA(order=self.order, seasonal_order=self.seasonal_order)
            model.fit(y, X=exog, fh=fh)

            self.fitted_models[uid] = model
            self.model_lookup[uid] = {"exog_cols": exog_cols}

        return self

# 1.3. Predict for each fitted series
    def predict(self, horizon: int, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:

        results = []

        for uid, model in self.fitted_models.items():
            exog = None
            if X is not None:
                exog_cols = self.model_lookup[uid]["exog_cols"]
                exog = X[X["unique_id"] == uid][exog_cols] if exog_cols else None

            fh = ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)
            y_pred = model.predict(fh=fh, X=exog)

            results.append(pd.DataFrame({
                "unique_id": uid,
                "ds": y_pred.index,
                "yhat": y_pred.values
            }))

        return pd.concat(results, ignore_index=True)

# 1.4. Update model with new data
    def update(self, df: pd.DataFrame, id_col="unique_id", y_col="y", ds_col="ds",
               exog_cols: Optional[list] = None, update_params=True):

        for uid, group in df.groupby(id_col):
            if uid not in self.fitted_models:
                raise ValueError(f"⚠️ [MODEL] Model update for {uid} has not been fitted yet.")

            y = pd.Series(group[y_col].values, index=pd.to_datetime(group[ds_col]))
            exog = group[exog_cols] if exog_cols else None

            self.fitted_models[uid].update(y=y, X=exog, update_params=update_params)

        return self