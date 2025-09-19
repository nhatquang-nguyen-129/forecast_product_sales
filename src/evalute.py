import numpy as np
import pandas as pd

# 1.1. Mean Absolute Percentage Error"""
def evaludate_mape_forecast(y_true, y_pred) -> float:

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

# 1.2. Root Mean Squared Error
def evaluate_rmse_forecast(y_true, y_pred) -> float:
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# 1.3. Trả về dict chứa MAPE & RMSE
def evaluate_result_forecast(df: pd.DataFrame, actual_col: str, forecast_col: str):
    
    return {
        "MAPE": mape(df[actual_col], df[forecast_col]),
        "RMSE": rmse(df[actual_col], df[forecast_col]),
    }
