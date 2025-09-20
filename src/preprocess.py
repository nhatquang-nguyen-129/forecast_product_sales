ts_palette = ['#233D4D', '#F26419', '#8AA29E', '#61210F', '#E8E391', '#6A9D98', '#C54F33', '#3E5A4D', '#AA7F41', '#A24422']


# 1.1. Splits the time series data into train and test sets on a chronological order based on the cutoff date
def time_series_split(data, cutoff_date):
    '''
    Args:
    data (pandas.DataFrame): The time series data to be split.
    cutoff_date (str or datetime): The date that separates the training and test sets.

    Raises:
    CustomException: An error occurred during the time series split.

    Returns:
    tuple: A tuple containing two pandas.DataFrame objects, where the first one represents the training set
    with data before the cutoff date, and the second one represents the test set with data on and after the cutoff date.
    '''
    try:
        train = data.loc[data.index < cutoff_date]
        test = data.loc[data.index >= cutoff_date]
        return train, test
    
    except Exception as e:
        raise CustomException(e, sys)
    

# 1.2. Plots the time series data after splitting into train and test sets
def plot_time_series_split(train, test, cutoff_date):
    '''
   

    Args:
    train (pandas.DataFrame): The training data to be plotted.
    test (pandas.DataFrame): The test data to be plotted.
    cutoff_date (str or datetime): The date that separates the training and test sets.

    Raises:
    CustomException: An error occurred during the plotting process.
    '''
    try:
        figure, ax = plt.subplots(figsize=(20, 7))

        train.plot(ax=ax, label='Train', y='sales')
        test.plot(ax=ax, label='Test', y='sales')

        ax.axvline(cutoff_date, color='black', ls='--')

        plt.title('Time series train-test-split', fontsize=25, fontweight='bold', loc='left', pad=25)
        plt.xlabel('Date', loc='left', labelpad=25)
        plt.ylabel('Sales', loc='top', labelpad=25)
        plt.xticks(rotation=0)
        plt.legend(loc='upper left')
        plt.show()
    
    except Exception as e:
        raise CustomException(e, sys)
    

def time_series_cv_report(data, target, test_size=None, gap=0, n_splits=5):
    '''
    Generates a time series cross-validation report and plot for the data.

    Args:
    data (pandas.DataFrame): The time series data.
    target (str): The target variable.
    test_size (int, optional): The size of the test set. Defaults to None.
    gap (int, optional): The gap between train and test sets. Defaults to 0.
    n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.

    Raises:
    CustomException: An error occurred during the time series cross-validation report generation.
    '''
    try:
        # Get sklearn TimeSeriesSplit object to obtain train and validation chronological indexes at each fold.
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

        fig, axes = plt.subplots(n_splits, 1, figsize=(20, 8), sharex=True, sharey=True)

        for fold, (train_index, val_index) in enumerate(tscv.split(data)):
            # Print train and validation indexes at each fold.
            print('-'*30)
            print(f'Fold {fold}')
            print(f'Train: {train_index[0]} to {train_index[-1]}')
            print(f'Validation: {val_index[0]} to {val_index[-1]}')

            # Plot the Time Series Split at each fold.
            axes[fold].plot(data.index, data[target], label='Complete Data', color='green')
            axes[fold].plot(data.iloc[train_index].index, data[target].iloc[train_index], label='Train')
            axes[fold].plot(data.iloc[val_index].index, data[target].iloc[val_index], label='Validation')

            axes[fold].set_title(f'Fold {fold} Time Series Split')
            axes[fold].legend(loc='upper left')

        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        raise CustomException(e, sys)
    

def time_series_cv(data, model, target, test_size=None, gap=0, n_splits=5, log=False, verbose=False, display_score=True):
    '''
    Performs time series cross-validation for the specified model and data.

    Args:
    data (pandas.DataFrame): The time series data.
    model : The machine learning model to be used.
    target (str): The target variable.
    test_size (int, optional): The size of the test set. Defaults to None.
    gap (int, optional): The gap between train and test sets. Defaults to 0.
    n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
    log (bool, optional): Whether a log-transformation was applied to the target variable. Defaults to False.
    verbose (bool, optional): Whether to display verbose output. Defaults to False.
    display_score (bool, optional): Whether to display the cross-validation score. Defaults to True.

    Raises:
    CustomException: An error occurred during the time series cross-validation process.
    '''
    try:
        # Get sklearn TimeSeriesSplit object to obtain train and validation chronological indexes at each fold.
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

        scores = []
        for fold, (train_index, val_index) in enumerate(tscv.split(data)):
            # Obtain train and validation data at fold k.
            train = data.iloc[train_index]
            val = data.iloc[val_index]

            # Obtain predictor and target train and validation sets.
            X_train = train.drop(columns=[target])
            y_train = train[target].copy()
            X_val = val.drop(columns=[target])
            y_val = val[target].copy()

            # Fit the model to the training data.
            model.fit(X_train, y_train)

            # Predict on validation data.
            y_pred = model.predict(X_val)

            # Obtain the validation score at fold k.
            if log:
                score = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_pred)))
            else:
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            
            scores.append(score)

            # Print the results and returning scores array.

            if verbose:
                print('-'*40)
                print(f'Fold {fold}')
                print(f'Score (RMSE) = {round(score, 4)}')
        
        if not display_score:
            return scores
        
        print('-'*60)
        print(f"{type(model).__name__}'s time series cross validation results:")
        print(f'Average validation score = {round(np.mean(scores), 4)}')
        print(f'Standard deviation = {round(np.std(scores), 4)}')

        return scores
    
    except Exception as e:
        raise CustomException(e, sys)
    