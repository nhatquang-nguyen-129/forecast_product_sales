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

def evaluate_regressor(y_true, y_pred, y_train, model_name):
    '''
    Evaluates a regression model based on various metrics and plots.

    Args:
    y_true : The true target values.
    y_pred : The predicted target values.
    y_train : The actual target values from the training set.
    model_name (str): The name of the regression model.

    Returns:
    pandas.DataFrame: A dataframe containing the evaluation metrics.

    Raises:
    CustomException: An error occurred during the evaluation process.
    '''
    try:
        mae = round(mean_absolute_error(y_true, y_pred), 4)
        mse = round(mean_squared_error(y_true, y_pred), 4)
        rmse = round(np.sqrt(mse), 4)
        r2 = round(r2_score(y_true, y_pred), 4)
        mape = round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 4)
        
        # Metrics
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Mean Absolute Percentage Error (MAPE): {mape}')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'R-Squared (R2): {r2}')
        
        # Obtain a dataframe of the metrics.
        df_results = pd.DataFrame({'Model': model_name, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2}, index=['Results'])

        # Residual Plots
        
        # Analyze the results
        plt.figure(figsize=(5, 3))
        plt.title('Actual values vs predicted values', fontweight='bold', fontsize=12, pad=20, loc='left')
        plt.plot([y_train.min(),y_train.max()],[y_train.min(),y_train.max()], linestyle='--', color='#F26419')
        plt.scatter(y_true, y_pred, color='#233D4D')
        plt.xlabel('Actual', loc='left', labelpad=10, fontsize=11)
        plt.ylabel('Predicted', loc='top', labelpad=10, fontsize=11)
        plt.show()
        
        # Distribution of the residuals
        plt.figure(figsize=(5, 3))
        sns.distplot((y_true - y_pred))
        plt.title('Residuals distribution', fontsize=12, fontweight='bold', loc='left', pad=20)
        plt.xlabel('Sales', loc='left', labelpad=10, fontsize=11)
        plt.ylabel('Density', loc='top', labelpad=10, fontsize=11)
        plt.show()

        return df_results

    except Exception as e:
        raise CustomException(e, sys)
    

def compare_actual_predicted(y_true, y_pred):
    '''
    Compares actual and predicted values and calculates the residuals.

    Args:
    y_true : The true target values.
    y_pred : The predicted target values.

    Returns:
    pandas.DataFrame: A dataframe containing the actual, predicted, and residual values.

    Raises:
    CustomException: An error occurred during the comparison process.
    '''
    try:
        actual_pred_df = pd.DataFrame({'Actual': np.round(y_true, 2),
                                    'Predicted': np.round(y_pred, 2), 
                                    'Residual': np.round(np.abs(y_pred - y_true), 2)})
        return actual_pred_df
    except Exception as e:
        raise CustomException(e, sys)
    

# 1. Plots the actual and predicted values against the testing dates
def plot_predictions(testing_dates, y_test, y_pred):
    '''
    Args:
    testing_dates : The dates corresponding to the testing data.
    y_test : The true target values from the testing set.
    y_pred : The predicted target values.

    Raises:
    CustomException: An error occurred during the plotting process.
    '''
    try:
        df_test = pd.DataFrame({'date': testing_dates, 'actual': y_test, 'prediction': y_pred })

        figure, ax = plt.subplots(figsize=(20, 7))
            
        df_test.plot(ax=ax, label='Actual', x='date', y='actual')
        df_test.plot(ax=ax, label='Prediction', x='date', y='prediction')
        
        plt.title('Actual vs Prediction', fontweight='bold', fontsize=25, loc='left', pad=25)
        plt.ylabel('Sales', loc='top', labelpad=25)
        plt.xlabel('Date', loc='left', labelpad=25)
        plt.xticks(rotation=0)

        plt.legend(['Actual', 'Prediction'], loc='upper left')
        plt.show()
    
    except Exception as e:
        raise CustomException(e, sys)