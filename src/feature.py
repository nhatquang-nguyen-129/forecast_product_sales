

import pandas as pd

def create_time_features(df: pd.DataFrame, date_col: str = "order_date") -> pd.DataFrame:
    """
    Tạo các time-based features từ cột datetime.
    
    Args:
        df (pd.DataFrame): DataFrame input đã có cột datetime
        date_col (str): Tên cột datetime (default = "order_date")
    
    Returns:
        pd.DataFrame: DataFrame với thêm các feature mới
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Basic time features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["yearmonth"] = df[date_col].dt.to_period("M").astype(str)  # ví dụ "2025-08"
    df["month_number"] = (df["year"] - df["year"].min()) * 12 + df["month"]

    # Tuần & ngày nếu cần
    df["week"] = df[date_col].dt.isocalendar().week
    df["day_of_week"] = df[date_col].dt.dayofweek  # Monday=0, Sunday=6
    
    return df


def aggregate_sales_monthly(
    df: pd.DataFrame,
    value_col: str = "sales_amount",
    group_cols: list = None,
    date_col: str = "order_date"
) -> pd.DataFrame:
    """
    Aggregate dữ liệu sales theo tháng (mặc định toàn bộ, hoặc theo từng group).
    
    Args:
        df (pd.DataFrame): DataFrame input
        value_col (str): Cột giá trị để aggregate (ví dụ: "sales_amount")
        group_cols (list): Cột grouping (ví dụ: ["store_id", "sku_id"])
        date_col (str): Cột datetime
    
    Returns:
        pd.DataFrame: DataFrame aggregated theo Year-Month
    """
    df = create_time_features(df, date_col)
    
    agg_cols = group_cols if group_cols else []
    group_keys = agg_cols + ["yearmonth"]

    df_agg = (
        df.groupby(group_keys)[value_col]
        .sum()
        .reset_index()
        .sort_values(group_keys)
    )
    
    return df_agg


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list = None,
    target_col: str = "sales_amount",
    lags: list = [1, 3, 6]
) -> pd.DataFrame:
    """
    Thêm lag features (dùng cho ML forecasting).
    
    Args:
        df (pd.DataFrame): DataFrame aggregated
        group_cols (list): Cột grouping (vd: ["store_id"])
        target_col (str): Target column (vd: "sales_amount")
        lags (list): Các bước lag (tháng trước, 3 tháng trước, ...)
    
    Returns:
        pd.DataFrame
    """
    df = df.copy()
    group_cols = group_cols if group_cols else []
    
    df = df.sort_values(group_cols + ["yearmonth"])
    
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    
    return df
