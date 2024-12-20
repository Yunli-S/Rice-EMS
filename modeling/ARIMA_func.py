import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.metrics import mean_squared_error
import seaborn as sns
import math
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from itertools import product
import warnings

# Define reuseable walk-forward validation function to calculate validity of model
def arima_model(p, d, q, training, testing, pred_periods, test):
    """
    Calculate the the forecasts for the next pred_periods periods using the given parameters for an ARIMA model
    inputs:
    - p, d, q: parameters for ARIMA
    - training, testing: training and testing datasets
    - pred_periods: number of periods to predict into the future
    - test: boolean to determine whether this model is used for training (0) or testing (1)
    outputs: training / testing RMSE, and the forecasted values
    """
    forecasts = list() # keeps track of all the predicted points
    history = list(training.copy()) # captures walk-forward training data
    testdata = list(testing.copy()) # keeps a list copy for testing data to avoid dataframe row name mismatch
    error = 0 # sum of the errors
    yhat_conf_int = []
    for t in range(len(testdata)):
        obs = testdata[t]
        model = ARIMA(history, order=(p,d,q)) # update the training set to include the latest predicted value
        model_fit = model.fit()
        yhat = model_fit.forecast() # predicted data point
        result = model_fit.get_forecast()
        ci = result.conf_int(0.05)
        yhat_conf_int.append(ci)
        forecasts.append(float(yhat))
        history.append(float(yhat))
        error += (yhat - obs) **2
    rmse = math.sqrt(error/len(testdata))
    if test: # test is the parameter to indicate whether this function is used to predict / test
        return round(rmse, 2), forecasts
    else:
        return round(rmse, 2), forecasts[:pred_periods], yhat_conf_int[:pred_periods]

def find_arima_params(prange, drange, qrange, train, test, pred_period):
    '''
    Find the best parameters for ARIMA model in a given range
    inputs:
    - prange, drange, qrange: the upper limit of the range for p, d, q for the ARIMA model, 
        assuming lower range is 1, to avoid long iterations
    - train, test: training and testing datasets
    - pred_periods: number of periods to predict into the future
    outputs: the optimal parameter values for an ARIMA model in the given range
    '''
    # Generate all different combinations of p, d, q triplets
    # Define the p, d, q parameters to try out
    p = range(0, prange)  # AR order
    d = range(0, drange)  # Differencing order
    q = range(0, qrange)  # MA order
    pdq = list(product(p, d, q))
    best_error = float("inf")
    best_pdq = None

    for param in pdq:
        try:
            error = arima_model(param[0],param[1],param[2],train,train, 1, 0)[0]
        # Compare this model's AIC with the best found so far
            if error < best_error:
                best_error = error
                best_pdq = param
        except:
            continue  # Some combinations might not converge and will throw an error

    return best_pdq

def plot_time_series_model_interval(data, train, train_forecast, test, test_forecast, forecasts, lower_quantile, upper_quantile, init_year):
    '''
    Plots the actual vs predicted values in a well defined line plot
    inputs:
    - data: the whole data set used (training and testing)
    - test: testing dataset
    - test_forecast: the forecasted values for the testing data
    - forecasts: the forecasted values for applying the model to the entire dataset
    - lower_quantile: lower quantile of the forecasted values
    - upper_quantile: upper quantile of the forecasted values
    - init_year: first year on the x-axis for specific graphing purpose
    output: N/A (the plot)
    '''
    df_train_forecast = pd.DataFrame(train_forecast[1], index=train.index)
    df_test_forecast = pd.DataFrame(test_forecast[1], index=test.index)
    df_forecast = pd.DataFrame(forecasts, index=range(test.index[-1]+1, test.index[-1]+len(forecasts)+1))
    # print("training", df_train_forecast)
    # print("testing", df_test_forecast)
    # print("forecast", df_forecast)
    # print(pd.concat([df_train_forecast, df_test_forecast, df_forecast]))
    plt.figure(figsize=(9.5,4))
    plt.xlabel('Academic Year', fontsize=10)
    plt.ylabel('Target', fontsize=10)
    plt.title("Time series prediction for the next 3 years")
    pred_pd = 3
    years = range(init_year, init_year + len(data) + pred_pd)
    
    print("years: ", years)
    print("length of data: ", len(data))
    print("data: ", data)

    # Plot actual data
    plt.plot(years[:len(data)], data, label="Actual", marker='o')
    
    # Plot predicted data
    # train data prediction
    # plt.plot(years[:len(train)], df_train_forecast, color='red', label="Train Predicted", marker='o')
    # test data prediction
    # plt.plot(years[len(data) - len(test):len(years) - pred_pd], df_test_forecast, color='red', label="Test Predicted", marker='o')
    # forecast data prediction
    # plt.plot(years[len(data):], df_forecast, color='blue', label="Forecast Predicted", marker='o')
    preds = pd.concat([df_train_forecast, df_test_forecast, df_forecast])
    plt.plot(years, preds, color='red', label="Predicted", marker='o')

    # Plot confidence intervals
    plt.fill_between(years[len(data):], lower_quantile, upper_quantile, color='red', alpha=0.3, label='95% Confidence Interval')
    
    plt.legend()
    plt.show()


def plot_time_series_model(data, train, train_forecast, test, test_forecast, forecasts, init_year):
    '''
    Plots the actual vs predicted values in a well defined line plot
    inputs:
    - data: the whole data set used (training and testing)
    - test: testing dataset
    - test_forecast: the forecasted values for the testing data
    - forecasts: the forecasted values for applying the model to the entire dataset
    - init_year: first year on the x-axis for specific graphing purpose
    output: N/A (the plot)
    '''
    df_train_forecast = pd.DataFrame(train_forecast[1], index=train.index)
    df_test_forecast = pd.DataFrame(test_forecast[1], index=test.index)
    print(df_test_forecast)
    df_forecast = pd.DataFrame(forecasts, index=range(test.index[-1]+1, test.index[-1]+len(forecasts)+1))
    print(df_forecast)
    pred_pd = 3
    years = range(init_year, init_year + len(data) + pred_pd)
    plt.figure(figsize=(9.5,4))
    plt.xlabel('Academic Year', fontsize=10)
    plt.ylabel('Target', fontsize=10)
    plt.title("Time series prediction for the next 3 years")
    plt.plot(years[:len(data)], data, label="Actual", marker='o')
    df_pred = pd.concat([df_train_forecast, df_test_forecast,df_forecast], ignore_index= True)
    plt.plot(years, df_pred, color='red', label="Predicted", marker='o')
    plt.legend(["Actual", "Predicted"])
    plt.show()

# Construct an error calculation function
def error_calc(train, test, pred_pd):
    '''
    Calculate RMSE of prediction vs actuals
    inputs:
    - train, test: training and testing datasets for the VAR(2) model
    - pred_pd: number of periods to predict
    outupts
    the RMSE calculated, and the predicted values
    '''
    model = sm.tsa.VAR(train)
    results = model.fit(2)
    y_predicted = results.forecast(train.values, steps=pred_pd)
    preds = []
    for y_pred in y_predicted:
        preds.append(y_pred[3]) # the total staff count, variable of interest - column "Total"
    error = 0
    # print(preds)
    test_list = list(test["Total"])
    # print(test_list)
    for i in range(min(len(test), pred_pd)):
        out = preds[i] - test_list[i]
        out = out*out 
        error+=out
    rmse = math.sqrt(error/len(test))
    return rmse, preds

def error(actual, predicted):
    '''
    Calculate RMSE of prediction vs actuals
    inputs:
    - actual, predicted: actual data and predictions
    output
    the RMSE calculated
    '''
    output = 0
    for i in range(len(actual)):
        out = predicted[i] - actual[i]
        out = out*out 
        output+=out
    rmse = math.sqrt(output/len(actual))
    return rmse