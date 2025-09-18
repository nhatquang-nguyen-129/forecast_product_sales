

import pandas as pd

import pandas as pd
import numpy as np

# =========================
# 1. Datetime utilities
# =========================
def ensure_datetime(df: pd.DataFrame, date_col: str = "order_date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df

# =========================
# 2. Basic time features
# =========================
def create_time_features(df: pd.DataFrame, date_col: str = "order_date") -> pd.DataFrame:
    df = ensure_datetime(df, date_col)
    
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["hour"] = df[date_col].dt.hour
    df["weekday"] = df[date_col].dt.weekday  # Monday=0
    df["week"] = df[date_col].dt.isocalendar().week
    df["yearmonth"] = df[date_col].dt.to_period("M").astype(str)
    df["month_number"] = (df["year"] - df["year"].min())*12 + df["month"]

    # Cyclical encoding
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["weekday_sin"] = np.sin(2*np.pi*df["weekday"]/7)
    df["weekday_cos"] = np.cos(2*np.pi*df["weekday"]/7)

    # Weekend & month start/end
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    
    return df

# 3. Aggregate / resample
def aggregate_sales(
    df: pd.DataFrame,
    value_col: str = "sales_amount",
    group_cols: list = None,
    freq: str = "M",
    date_col: str = "order_date"
) -> pd.DataFrame:
    """
    Aggregate sales theo frequency (month, week...) với các group_cols.
    """
    df = create_time_features(df, date_col)
    group_cols = group_cols if group_cols else []
    
    if freq.upper() == "M":
        df["period"] = df[date_col].dt.to_period("M")
    elif freq.upper() == "W":
        df["period"] = df[date_col].dt.to_period("W")
    else:
        df["period"] = df[date_col]
    
    df_agg = (
        df.groupby(group_cols + ["period"])[value_col]
        .sum()
        .reset_index()
        .sort_values(group_cols + ["period"])
    )
    
    df_agg["yearmonth"] = df_agg["period"].dt.to_timestamp().dt.to_period("M").astype(str)
    
    return df_agg

# =========================
# 4. Lag features
# =========================
def add_lag_features(df, target_col="sales_amount", group_cols=None, lags=[1,3,6]):
    df = df.copy()
    group_cols = group_cols if group_cols else []
    df = df.sort_values(group_cols + ["yearmonth"])
    
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    return df

# =========================
# 5. Rolling / Moving features
# =========================
def add_rolling_features(df, target_col="sales_amount", group_cols=None, windows=[3,6,12]):
    df = df.copy()
    group_cols = group_cols if group_cols else []
    for w in windows:
        df[f"{target_col}_rolling_mean_{w}"] = df.groupby(group_cols)[target_col].transform(lambda x: x.shift(1).rolling(w).mean())
        df[f"{target_col}_rolling_std_{w}"] = df.groupby(group_cols)[target_col].transform(lambda x: x.shift(1).rolling(w).std())
    return df

# =========================
# 6. Diff / pct change features
# =========================
def add_diff_features(df, target_col="sales_amount", group_cols=None, periods=[1,12]):
    df = df.copy()
    group_cols = group_cols if group_cols else []
    for p in periods:
        df[f"{target_col}_diff_{p}"] = df.groupby(group_cols)[target_col].diff(p)
        df[f"{target_col}_pct_change_{p}"] = df.groupby(group_cols)[target_col].pct_change(p)
    return df

# =========================
# 7. Full pipeline
# =========================
def prepare_features(
    df,
    value_col="sales_amount",
    group_cols=None,
    lags=[1,3,6],
    rolling_windows=[3,6,12],
    diff_periods=[1,12]
):
    df = create_time_features(df)
    df = add_lag_features(df, target_col=value_col, group_cols=group_cols, lags=lags)
    df = add_rolling_features(df, target_col=value_col, group_cols=group_cols, windows=rolling_windows)
    df = add_diff_features(df, target_col=value_col, group_cols=group_cols, periods=diff_periods)
    return df

