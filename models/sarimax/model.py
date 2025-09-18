"""
==================================================================
SARIMA FORECASTING MODEL
------------------------------------------------------------------
Wrapper for SARIMA forecasting with support for:
- Seasonal ARIMA (P,D,Q,s) with configurable trend
- Exogenous regressors (optional)
- Multiple cohorts (fit one model per unique_id)

✔️ Supports SARIMA(p,d,q)(P,D,Q,s)
✔️ Handles multi-series forecasting in a unified way
✔️ Provides methods consistent with the BaseForecastModel interface

⚠️ Each time series is modeled independently (no cross-series learning).
==================================================================
"""

from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base.model import BaseForecastModel


class SARIMAModel(BaseForecastModel):
    """
    ----------
    order : tuple, default=(1,1,1)
        ARIMA (p,d,q) parameters
    seasonal_order : tuple, default=(1,1,1,12)
        Seasonal ARIMA (P,D,Q,s) with s = seasonal period
    trend : str or None, optional
        Trend parameter {'n','c','t','ct'}
    """

# 1.1. Initialize SARIMA model
    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 12),
        trend: Optional[str] = None,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        # model_lookup stores metadata and training series for each uid
        # e.g. model_lookup[uid] = {"exog_cols": [...], "y": pd.Series, "exog": pd.DataFrame or None, "last_date": Timestamp, "freq": "D"/"M"/...}
        self.model_lookup: Dict[str, Dict[str, Any]] = {}
        # fitted_models stores the statsmodels SARIMAXResults for each uid
        self.fitted_models: Dict[str, Any] = {}

# 1.2. Fit SARIMA model for each time series in the DataFrame
    def fit(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        y_col: str = "y",
        ds_col: str = "ds",
        exog_cols: Optional[List[str]] = None,
    ):
        """
        Fit SARIMA per unique_id.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain id_col, ds_col (datetime-like), y_col, and optional exog_cols
        exog_cols : list[str] or None
            Column names for exogenous regressors (must be present in df)
        """
        for uid, group in df.groupby(id_col):
            group = group.sort_values(ds_col)
            y = pd.Series(group[y_col].values, index=pd.to_datetime(group[ds_col]))
            exog = group[exog_cols] if exog_cols else None

            model = SARIMAX(
                endog=y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                exog=exog,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False)

            freq = pd.infer_freq(y.index) or (y.index.freqstr if getattr(y.index, "freqstr", None) else None)
            self.fitted_models[uid] = results
            self.model_lookup[uid] = {
                "exog_cols": exog_cols,
                "y": y,
                "exog": exog,
                "last_date": y.index[-1],
                "freq": freq,
            }

        return self

# 1.3. Predict for each fitted series
    def predict(
        self,
        horizon: int,
        X: Optional[pd.DataFrame] = None,
        id_col: str = "unique_id",
        ds_col: str = "ds",
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Predict horizon steps ahead for every fitted unique_id.

        Parameters
        ----------
        horizon : int
            Number of steps (periods) ahead to forecast
        X : pd.DataFrame or None
            If provided, should contain future exog rows with columns including id_col and ds_col
        alpha : float
            Significance level for confidence intervals (default 0.05)

        Returns
        -------
        pd.DataFrame
            columns: unique_id, ds, yhat, yhat_lower, yhat_upper
        """
        results_list = []

        for uid, results_obj in self.fitted_models.items():
            meta = self.model_lookup[uid]
            exog_cols = meta["exog_cols"]
            freq = meta.get("freq") or "D"
            last_date = meta["last_date"]

    # 1.3.1. prepare exog for forecast if provided
            exog_for_pred = None
            if X is not None and exog_cols:
                exog_df = (
                    X[X[id_col] == uid]
                    .sort_values(ds_col)
                    .reset_index(drop=True)
                )[exog_cols]
    
    # 1.3.2. if not enough rows, pad by repeating last row
                if len(exog_df) == 0:
                    exog_for_pred = None
                else:
                    if len(exog_df) >= horizon:
                        exog_for_pred = exog_df.iloc[:horizon].to_numpy()
                    else:
                        # pad by repeating last row to reach horizon
                        last_row = exog_df.iloc[[-1]].to_numpy()
                        pad = np.repeat(last_row, repeats=(horizon - len(exog_df)), axis=0)
                        exog_for_pred = np.vstack([exog_df.to_numpy(), pad])

    # 1.3.3. Call statsmodels forecast
            try:
                forecast_res = results_obj.get_forecast(steps=horizon, exog=exog_for_pred)
                mean_vals = np.asarray(forecast_res.predicted_mean).ravel()
                ci = forecast_res.conf_int(alpha=alpha)
                lower_vals = np.asarray(ci.iloc[:, 0]).ravel()
                upper_vals = np.asarray(ci.iloc[:, 1]).ravel()
            except Exception:
                # fallback: try without exog (if exog caused issue)
                forecast_res = results_obj.get_forecast(steps=horizon)
                mean_vals = np.asarray(forecast_res.predicted_mean).ravel()
                ci = forecast_res.conf_int(alpha=alpha)
                lower_vals = np.asarray(ci.iloc[:, 0]).ravel()
                upper_vals = np.asarray(ci.iloc[:, 1]).ravel()

    # 1.3.4. build proper datetime index for forecasted periods
            try:
                # attempt to construct start = last_date + freq offset
                start = last_date + pd.tseries.frequencies.to_offset(freq)
                ds_index = pd.date_range(start=start, periods=horizon, freq=freq)
            except Exception:
                # default to daily if freq not interpretable
                ds_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

            df_out = pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": ds_index,
                    "yhat": mean_vals,
                    "yhat_lower": lower_vals,
                    "yhat_upper": upper_vals,
                }
            )
            results_list.append(df_out)

        if len(results_list) == 0:
            return pd.DataFrame(columns=["unique_id", "ds", "yhat", "yhat_lower", "yhat_upper"])

        return pd.concat(results_list, ignore_index=True)

# 1.4. Update model with new data
    def update(
        self,
        df: pd.DataFrame,
        id_col: str = "unique_id",
        y_col: str = "y",
        ds_col: str = "ds",
        exog_cols: Optional[List[str]] = None,
        refit: bool = True,
    ):
        """
        Update (append) new observations for each uid and refit model.

        Note: statsmodels' SARIMAXResults does not provide a universal safe incremental
        update for all configurations — here we implement safe behavior by concatenating
        stored training series with new data and refitting the SARIMAX model.

        Parameters
        ----------
        df : pd.DataFrame
            New observations (can contain multiple uids)
        refit : bool
            If True, refit model using the concatenated data. If False, we will raise
            unless alternative incremental logic is implemented.
        """
        for uid, group in df.groupby(id_col):
            if uid not in self.fitted_models:
                raise ValueError(f"⚠️ [MODEL] Model update for {uid} has not been fitted yet.")

            if not refit:
                raise NotImplementedError("Incremental update without refit is not implemented. Set refit=True.")

            group = group.sort_values(ds_col)
            y_new = pd.Series(group[y_col].values, index=pd.to_datetime(group[ds_col]))
            exog_new = group[exog_cols] if exog_cols else None

            # combine with previous training series
            y_old: pd.Series = self.model_lookup[uid]["y"]
            y_combined = pd.concat([y_old, y_new]).sort_index()

            if exog_cols:
                exog_old = self.model_lookup[uid]["exog"]
                if exog_old is None:
                    exog_combined = exog_new
                else:
                    exog_combined = pd.concat([exog_old, exog_new]).sort_index()
            else:
                exog_combined = None

            # refit SARIMAX on combined data
            model = SARIMAX(
                endog=y_combined,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                exog=exog_combined,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False)

            # update stored items
            freq = pd.infer_freq(y_combined.index) or self.model_lookup[uid].get("freq")
            self.fitted_models[uid] = results
            self.model_lookup[uid].update(
                {
                    "y": y_combined,
                    "exog": exog_combined,
                    "last_date": y_combined.index[-1],
                    "freq": freq,
                    "exog_cols": exog_cols or self.model_lookup[uid].get("exog_cols"),
                }
            )

        return self