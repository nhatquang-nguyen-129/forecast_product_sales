"""
==================================================================
BASE FORECAST MODEL
------------------------------------------------------------------
Abstract base class for all forecasting models in this project.
Every forecasting model (ARIMA, Prophet, LightGBM, Chronos, etc.) 
must inherit from this class to ensure a consistent interface.

✔️ Defines core methods: fit, predict, update
✔️ Provides unified typing for multi-series forecasting
✔️ Allows plugging models into pipelines and benchmarking

⚠️ This is only an abstract interface — no actual logic is here.
==================================================================
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

# 1. WRAPPER FOR ABSTRACT BASE CLASS TO SHARE A CONSISTENT INTERFACE
class BaseForecastModel(ABC):

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        y_col: str = "y",
        ds_col: str = "ds",
        exog_cols: Optional[list] = None,
        fh: Optional[int] = None,
    ):
        """
        Fit the model to the given time series data.

        Parameters
        ----------
        df : DataFrame
            Input dataframe containing time series
        id_col : str
            Column representing series identifier (e.g., SKU, Store)
        y_col : str
            Target variable column (e.g., sales, demand)
        ds_col : str
            Datetime column
        exog_cols : list, optional
            List of exogenous feature columns
        fh : int, optional
            Forecast horizon (steps ahead)
        """
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int,
        X: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate forecasts for the given horizon.

        Parameters
        ----------
        horizon : int
            Forecast horizon (steps ahead)
        X : DataFrame, optional
            Future exogenous features

        Returns
        -------
        DataFrame
            Forecasted values with columns [unique_id, ds, yhat]
        """
        pass

    @abstractmethod
    def update(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        y_col: str = "y",
        ds_col: str = "ds",
        exog_cols: Optional[list] = None,
        update_params: bool = True,
    ):
        """
        Update fitted models with new observations.

        Parameters
        ----------
        df : DataFrame
            New data to update the model
        id_col : str
            Column representing series identifier
        y_col : str
            Target variable column
        ds_col : str
            Datetime column
        exog_cols : list, optional
            List of exogenous feature columns
        update_params : bool
            Whether to re-estimate parameters
        """
        pass
