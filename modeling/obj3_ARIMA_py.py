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
import ARIMA_func

def predict_item(item_name):
    X = df_budget_general[item_name]
    plot_acf(df_budget_general[item_name])
    plt.show()
    init_year = 2016 # start of the time series
    # 5 years of the data for training, 2 years of the data for testing
    size = 5
    train, test = X[0:size], X[size:len(X)]

    # Define number of periods to predict
    pred_pd = 3
    print(item_name + " training data: \n", train)
    print(item_name + " testing data: \n", test)
    model = ARIMA(train, order=(1,0,1))
    model_fit = model.fit()

    # summary of fit model
    print(model_fit.summary())

    warnings.warn("deprecated", DeprecationWarning)
    warnings.simplefilter("ignore")

    # Parameter tuning for ARIMA model
    arima_params = ARIMA_func.find_arima_params(5, 5, 5, train, train, pred_pd)
    print(item_name + " Optimized parameters:", arima_params)

    # Train + Validation
    train_forecast = ARIMA_func.arima_model(arima_params[0], arima_params[1], arima_params[2], train, train, pred_pd, 1)
    arima_train_rmse = train_forecast[0]
    print(item_name + " Trainining RMSE: ", arima_train_rmse)

    # Test + Validation
    test_forecast = ARIMA_func.arima_model(arima_params[0],arima_params[1],arima_params[2], train, test, pred_pd, 1)
    print(item_name + " Testing RMSE: ", test_forecast[0])

    # Forecast for the next 3 years
    forecasts = ARIMA_func.arima_model(arima_params[0], arima_params[1], arima_params[2], X, X, pred_pd, 0)[1]
    print(item_name + " Forecasts: ", forecasts)

    # Forecast confidence interval for the next 3 years
    forecasts_interval = ARIMA_func.arima_model(arima_params[0], arima_params[1], arima_params[2], X, X, pred_pd, 0)[2]
    print(item_name + " Forecasts: ", forecasts_interval)

    # Construct interval-based predictions to capture upper and lower bounds for future
    lower = []
    upper = []
    for i in range(len(forecasts_interval)):
        lower.append(forecasts_interval[i][0][0])
        upper.append(forecasts_interval[i][0][1])
    print(item_name + " Lower quantile: ", lower)
    print(item_name + " Upper quantile: ", upper)

    # Plot actual vs prediction
    ARIMA_func.plot_time_series_model(X, train, train_forecast, test, test_forecast, forecasts, init_year)
    ARIMA_func.plot_time_series_model_interval(X, train, train_forecast, test, test_forecast, forecasts, lower, upper, init_year)

if __name__ == '__main__':
    df_budget_general = pd.read_csv("./data/obj3_dataset_knn.csv")
    # Predict IC Housing
    predict_item('IC Housing')
    # Predict Uniforms
    predict_item('Uniforms')
    # Predict Medical Supplies
    predict_item('Medical Supplies')
    # Predict Vehicle Maintenace
    predict_item('Vehicle Maintenace')
    # Predict Insurance
    predict_item('Insurance')