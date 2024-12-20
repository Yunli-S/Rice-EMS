import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import warnings
import ARIMA_func
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn import linear_model
# from sklearn.metrics import r2_score


##### ---------------------------------------------------
##### Polynomial Regression

def poly_reg(X, y, pred_years, deg, train_size):
    '''
    Perform polynomial regression to get predictions and errors
    inputs:
    - X, y: predictor and target variables
    - pred_years: value of the predictor variable for the periods whose target variable values are unknown
    - deg: degree of the polynomial regression model
    - train_size: size of the training set for the model
    outputs
    - the predicted values for the target variable for further periods
    - training and testing RMSEs for the model
    '''
    model = np.poly1d(np.polyfit(X[:train_size], y[:train_size], deg))
    line = np.linspace(X[0], pred_years[-1])
    predictions = model(pred_years)
    plt.plot(X, y, marker='o')
    plt.xlabel('Academic Year', fontsize=10)
    plt.ylabel('Number of staff', fontsize=10)
    plt.title("Time series prediction for the next 3 years")
    plt.ylim(70, 120)
    plt.plot(pred_years, predictions, color="blue", marker='o')
    plt.plot(line, model(line), color="blue")
    plt.show()
    existing_predictions = model(X)
    # Training & Testing RMSEs
    train_error, test_error = 0, 0
    for i in range(len(X)):
        if i < train_size:
            out_train = existing_predictions[i] - y[i]
            out_train = out_train*out_train
            train_error+=out_train
        else:
            out_test = existing_predictions[i] - y[i]
            out_test = out_test*out_test
            test_error+=out_test
    train_rmse = math.sqrt(train_error/train_size)
    test_rmse = math.sqrt(test_error/(len(X) - train_size))
    return predictions, train_rmse, test_rmse

    # Polynomial regression with testing and training data same color
    # def poly_reg1(X, y, pred_X, deg):
    #     model = np.poly1d(np.polyfit(X, y, deg))
    #     line = np.linspace(X[0], pred_X[-1])
    #     predictions = model(pred_X)
    #     # print(list(X) + pred_X)
    #     # print(list(y) + list(predictions))
    #     plt.plot(list(X) + pred_X, list(y) + list(predictions), color="blue", marker='o')
    #     plt.plot(line, model(line), color="red")
    #     plt.show()
    #     return predictions, r2_score(y, model(X))
    

def var_pred(train, pred_pd):
    '''
    Calculate the predictions using the VAR(2) model
    inputs:
    - train: training data
    - model: the VAR(2) model obtained after fitting the training data
    '''
    model = sm.tsa.VAR(train)
    results = model.fit(2) # Autoregressive model of order p=2, due to the limited training samples p can only be 1 or 2
                            # for reusability, input different order p as deemed appropriate
    forecast = results.forecast(train.values, steps=pred_pd)
    forecast_df = pd.DataFrame(forecast)
    forecast_df.columns = list(train.columns)
    # results.plot_forecast(pred_pd)
    # conf_int = results.forecast_interval(y=train.values[:-2], steps=pred_pd, alpha=0.05, exog_future=None)
    # results.plot_forecast(pred_pd)
    # plt.show()
    return list(forecast_df["Total"]) # enter the column name based on desired variable

def var_pred_with_interval(train, pred_pd):
    """
    Calculate the predictions using the VAR(2) model and obtain interval estimates.
    Inputs:
    - train: training data
    - pred_pd: number of periods to forecast
    """
    # Fit the VAR model
    model = sm.tsa.VAR(train)
    results = model.fit(2)  # Autoregressive model of order p=2
    
    # Generate point forecasts
    forecast = results.forecast(train.values, steps=pred_pd)
    print(forecast)
    # Calculate the asymptotic covariance matrix of the forecast errors
    cov_matrix = results.forecast_cov(pred_pd)
    
    # Extract the diagonal elements from each 2-dimensional covariance matrix
    variances = [np.diag(cov_matrix[i]) for i in range(pred_pd)]
    
    # Apply a floor to the variances to avoid numerical instability
    min_variance = 1e-1
    variances = [np.maximum(var, min_variance) for var in variances]
    
    # Calculate interval estimates using the asymptotic covariance matrix
    lower_quantile = forecast - 1.96 * np.sqrt(variances)
    upper_quantile = forecast + 1.96 * np.sqrt(variances)
    
    # Only keep the variable of interest (Total, with index 3 in training dataframe columns), 
    # as VAR predicts for all variables in the dataset
    forecast_p = []
    lower_q = []
    upper_q = []
    for i in range(len(forecast)):
        forecast_p.append(forecast[i][3])
        lower_q.append(lower_quantile[i][3])
        upper_q.append(upper_quantile[i][3])
    return forecast_p, lower_q, upper_q



if __name__ == '__main__':
    # Read and visualize data file into dataframe:
    df = pd.read_csv('./data/staff_data_model.csv')
    df


    plt.plot(df["Year"], df["Total"])
    plt.xlabel("Year")
    plt.ylabel("Total number of staff")
    plt.title("Academic Yearly staff count over 2015-2024")
    plt.show()

    ##### ---------------------------------------------------------------------------------------------
    ##### ARIMA Model to forecast staff count for the next 3 years

    # Autocorrelation plot to determine ARIMA model parameters:
    plot_acf(df["Total"])
    # Autocorrelations seem very low, and ARIMA model may not be appropriate, but we try fitting one below:
    model = ARIMA(df["Total"], order=(4,1,0))
    model_fit = model.fit()
    # summary of fit model
    print(model_fit.summary())
    # line plot of residuals
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    # density plot of residuals
    residuals.plot(kind='kde')
    plt.show()
    # summary stats of residuals
    print(residuals.describe())

    # From the residual plot, there does not seem to be any trend information not captured by the model, 
    # as the mean is relevantly stable
    # The density plot shows that the errors are roughly Gaussian and roughly centers around 0, 
    # though the right tail may be a bit thicker and it is slightly biased as the mean is not 0

    # Split the data into training and testing
    X = df["Total"]
    init_year = 2015 # start of the time series
    # 2/3 of the data for training, 1/3 of the data for testing
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    # Define number of periods to predict
    pred_pd = 3
    print("training data: \n", train)
    print("testing data: \n", test)

    warnings.warn("deprecated", DeprecationWarning)
    warnings.simplefilter("ignore")

    # Train the model
    history = [x for x in train]

    # Parameter tuning for ARIMA model
    
    arima_params = ARIMA_func.find_arima_params(5, 3, 2, train, test, pred_pd)
    print(arima_params)

    # Train + Validation
    train_forecast = ARIMA_func.arima_model(arima_params[0], arima_params[1], arima_params[2], train, train, pred_pd, 1)
    arima_train_rmse = train_forecast[0]
    print("Trainining RMSE: ", arima_train_rmse)

    # Test + Validation
    test_forecast = ARIMA_func.arima_model(arima_params[0], arima_params[1], arima_params[2], train, test, pred_pd, 1)
    arima_test_rmse = test_forecast[0]
    print("Testing RMSE: ", arima_test_rmse)

    # Forecast for the next 3 years
    arima_forecasts = ARIMA_func.arima_model(arima_params[0], arima_params[1], arima_params[2], X, X, pred_pd, 0)[1]
    print("Forecasts: ", arima_forecasts)

    # ARIMA predictions
    arima_preds = train_forecast[1] + test_forecast[1] + list(arima_forecasts)

    # Forecast confidence interval for the next 3 years
    forecasts_interval = ARIMA_func.arima_model(arima_params[0], arima_params[1], arima_params[2], X, X, pred_pd, 0)[2]
    print("Forecasts: ", forecasts_interval)

    # Construct interval-based predictions to capture upper and lower bounds for future
    arima_lower = []
    arima_upper = []
    for i in range(len(forecasts_interval)):
        arima_lower.append(forecasts_interval[i][0][0])
        arima_upper.append(forecasts_interval[i][0][1])
    print("Lower quantile: ", arima_lower)
    print("Upper quantile: ", arima_upper)
    # Plot actual vs prediction
    ARIMA_func.plot_time_series_model(X, train, train_forecast, test, test_forecast, arima_forecasts, init_year)
    ARIMA_func.plot_time_series_model_interval(X, train, train_forecast, test, test_forecast, arima_forecasts, arima_lower, arima_upper, init_year)

    # It seems that due to the low autocorrelation of staff count, it is not very effective 
    # to predict staff count only based on itself through ARIMA. 
    # So we add more variables to see if we can have a better prediction

    ##### ---------------------------------------------------------------------------------------------
    #### Incorporate more variables into the time series analysis
    # Redo dataset construction
    df_var = pd.read_csv('./data/Call Volume.csv')
    df_var

    # Clean data to get rid of NA values
    init_year = 2016
    df_var = df_var[["Academic year beginning in Fall of:", "Total UG enrollment", "Total Enrollment", "Total Employees", "Calls (from academic year rpt)"]]
    df_var = df_var[5:]
    df_var = df_var[:-1]
    df_var["Merge"] = range(len(df_var))
    df_var = df_var.dropna()
    df_var
    df["Merge"] = range(len(df))
    df
    merged_df = pd.merge(df, df_var, on='Merge')
    merged_df = merged_df[["Merge", "Volunteer", "Paid", "Total", "Total UG enrollment", "Total Enrollment", "Total Employees", "Calls (from academic year rpt)"]]
    merged_df

    # Plot and visualize the variables over time separately
    fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10,6))
    #print(merged_df.shape[1])
    for i, ax in enumerate(axes.flatten()):
        #print(i)
        if 0 < i < merged_df.shape[1]:
            data = merged_df[merged_df.columns[i]]
            years = [x for x in range(init_year, init_year + len(data))]
            ax.plot(years, data, color='red', linewidth=1)
            # Decorations
            ax.set_title(merged_df.columns[i])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.spines["top"].set_alpha(0)
            ax.tick_params(labelsize=6)

    plt.tight_layout()

    # Construct training and testing datasets
    size = int(0.66*len(merged_df))
    df_train = merged_df.iloc[:size]  # Select the first 'size' rows
    df_test = merged_df.iloc[size:]
    df_train
    df_test

    ##### ---------------------------------------------------------------------------------------------
    ## SARIMAX model
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    sarimax_model = SARIMAX(df_train['Total'], order=(2, 0, 1), exog=df_train[['Total UG enrollment', 'Total Enrollment', 'Total Employees', 'Calls (from academic year rpt)']])
    result = sarimax_model.fit()

    forecast = result.get_forecast(steps=3, exog=df_test[['Total UG enrollment', 'Total Enrollment', 'Total Employees', 'Calls (from academic year rpt)']])
    forecast

    forecast_mean = forecast.predicted_mean

    # Get confidence intervals of predictions
    forecast_ci = forecast.conf_int()

    # Plot the data along with the forecast and the confidence interval
    plt.figure(figsize=(14, 7))
    plt.plot(df_train.index, df_train['Total'], label='Train')
    plt.plot(df_test.index, df_test['Total'], label='Test')
    plt.plot(forecast_mean.index, forecast_mean, label='Forecast')
    plt.fill_between(forecast_ci.index, 
                    forecast_ci.iloc[:, 0], 
                    forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.title('Forecast vs Actuals')
    plt.legend()
    plt.show()




    ##### ---------------------------------------------------------------------------------------------
    ##### VAR model
    ## Relevant source: https://www.analyticsvidhya.com/blog/2021/08/vector-autoregressive-model-in-python/

    # Check correlation between variables to see if VAR is appropriate
    corr_matrix = merged_df.corr()
    plt.figure(figsize=(25, 20))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"fontsize":16})
    # plt.title('Feature Correlation Matrix', fontsize=20, fontweight='bold', color='black')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

    # Construct the model to forecast
    print(var_pred(df_train, pred_pd)) # forecast is for the entire pred_pd period for all variables in the dataframe

    training_error = ARIMA_func.error_calc(df_train, df_train, len(df_train))
    testing_error = ARIMA_func.error_calc(df_train, df_test, pred_pd)
    var_train_rmse = training_error[0]
    var_test_rmse = testing_error[0]
    print("training rmse: ", var_train_rmse, "\n testing error: ", var_test_rmse)

    # Forecast next 3 years
    var_forecasts = var_pred(merged_df, pred_pd)
    print("Forecasts: ", var_forecasts)

    var_preds = training_error[1] + testing_error[1] + list(var_forecasts)

    # Construct interval-based predictions to capture upper and lower bounds for testing dataset
    point_test, lower_quantile_test, upper_quantile_test = var_pred_with_interval(df_train, pred_pd)
    print("Point estimates: ", point_test)
    print("Lower quantile: ", lower_quantile_test)
    print("Upper quantile: ", upper_quantile_test)
    
    # Construct interval-based predictions to capture upper and lower bounds for future
    point, var_lower, var_upper = var_pred_with_interval(merged_df, pred_pd)
    print("Point estimates: ", point)
    print("Lower quantile: ", var_lower)
    print("Upper quantile: ", var_upper)

    # Plot the results
    ARIMA_func.plot_time_series_model(merged_df["Total"], df_train, training_error, df_test, testing_error, var_forecasts, init_year)
    ARIMA_func.plot_time_series_model_interval(merged_df["Total"], df_train, training_error, df_test, testing_error, var_forecasts, var_lower, var_upper, init_year)





    ##### ---------------------------------------------------------------------------------------------
    ##### Polynomial Regression model
    # Extract predictor and target variables, the predictor variable has to be such that we know its values into the future
    X_poly = df_var["Academic year beginning in Fall of:"].to_numpy()
    y_poly = merged_df["Total"].to_numpy()
    size = int(len(X_poly) * 0.66)
    pred_years = [2024, 2025, 2026] # future values of the predictor variable

    # Select optimal parameter for the degree of the polynomial regression
    for degree in range(5):
        print(poly_reg(X_poly, y_poly, pred_years, degree, size)[1:])
    # Combining considerations for training and testing RMSEs, the 1st degree polynomial, or linear regression, performs best
    polynomial_results = poly_reg(X_poly, y_poly, pred_years, 1, size)
    poly_train_rmse = polynomial_results[1]
    poly_test_rmse = polynomial_results[2]
    print("Training RMSE: ", poly_train_rmse)
    print("Testing RMSE: ", poly_test_rmse)
    poly_results_all = poly_reg(X_poly, y_poly, range(init_year, init_year + len(data) + pred_pd), 1, size)
    poly_preds = list(poly_results_all[0])


    #################### Plot all models on the same graph
    #----------------------------------------------------------------------
    

    ## Showing testing error in legend for model
    plt.figure(figsize=(15,8))
    plt.xlabel('Academic Year', fontsize=15)
    plt.ylabel('Number of staff', fontsize=15)
    plt.title("Comparison of time series predictions for the next 3 years", fontsize=20)

    years = range(init_year, init_year + len(data) + pred_pd)
    data = merged_df["Total"]

    # Plot actual data
    plt.plot(years[:len(data)], data, label="Actual", marker='o', linewidth=3)
    plt.ylim(70, 120)
    
    # Plot training predictions
    arima_label = "ARIMA, Train RMSE: " + str(int(arima_train_rmse)) + ", Test RMSE: " + str(int(arima_test_rmse))
    var_label = "VAR, Train RMSE: " + str(int(var_train_rmse)) + ", Test RMSE: " + str(int(var_test_rmse))
    reg_label = "Regression, Train RMSE: " + str(int(poly_train_rmse)) + ", Test RMSE: " + str(int(poly_test_rmse))
    plt.plot(years[:len(data)], arima_preds[1:len(data)+1], color='orange', label=arima_label, marker='o', linewidth=3)
    plt.plot(years[:len(data)], var_preds[:len(data)], color='green', label=var_label, marker='x', linewidth=3)
    plt.plot(years[:len(data)], poly_preds[:len(data)], color='red', label=reg_label, marker='^', linewidth=3)

    # Plot future predictions
    plt.plot(years[len(data):], arima_preds[len(data)+1:], color='orange', marker='o', linewidth=3)
    plt.plot(years[len(data):], var_preds[len(data):], color='green', marker='x', linewidth=3)
    plt.plot(years[len(data):], poly_preds[len(data):], color='red', marker='^', linewidth=3)

    # Plot confidence intervals
    plt.fill_between(years[len(data):], arima_lower, arima_upper, color='orange', alpha=0.3, label='ARIMA 95% Confidence Interval')
    plt.fill_between(years[len(data):], var_lower, var_upper, color='green', alpha=0.3, label='VAR 95% Confidence Interval')
    
    plt.legend()
    plt.show()

    ## Final - showing training error only, as the whole dataset is used for training
    ## Showing testing error in legend for model
    plt.figure(figsize=(15,8))
    plt.xlabel('Academic Year', fontsize=15)
    plt.ylabel('Number of staff', fontsize=15)
    plt.title("Comparison of time series predictions for the next 3 years", fontsize=20)

    years = range(init_year, init_year + len(data) + pred_pd)
    data = merged_df["Total"]

    # Plot actual data
    plt.plot(years[:len(data)], data, label="Actual", marker='o')
    plt.ylim(70, 120)
    
    arima_error = ARIMA_func.error(data, arima_preds[1:-3])
    var_error = ARIMA_func.error(data, var_preds[:-3])
    poly_error = ARIMA_func.error(data, poly_preds[:-3])

    # Plot training predictions
    arima_label_train = "ARIMA, RMSE: " + str(int(arima_error))
    var_label_train = "VAR, RMSE: " + str(int(var_error))
    reg_label_train = "Regression, RMSE: " + str(int(poly_error))
    plt.plot(years[:len(data)], arima_preds[1:len(data)+1], color='orange', label=arima_label_train, marker='o', linewidth=3)
    plt.plot(years[:len(data)], var_preds[:len(data)], color='green', label=var_label_train, marker='x', linewidth=3)
    plt.plot(years[:len(data)], poly_preds[:len(data)], color='red', label=reg_label_train, marker='^', linewidth=3)

    # Plot future predictions
    plt.plot(years[len(data):], arima_preds[len(data)+1:], color='orange', marker='o', linewidth=3)
    plt.plot(years[len(data):], var_preds[len(data):], color='green', marker='x', linewidth=3)
    plt.plot(years[len(data):], poly_preds[len(data):], color='red', marker='^', linewidth=3)

    # Plot confidence intervals
    plt.fill_between(years[len(data):], arima_lower, arima_upper, color='orange', alpha=0.3, label='ARIMA 95% Confidence Interval')
    plt.fill_between(years[len(data):], var_lower, var_upper, color='green', alpha=0.3, label='VAR 95% Confidence Interval')
    
    plt.legend()
    plt.show()


    ## Ultimate Final - fix formats to be consistent across poster - 4/15/2024
    plt.figure(figsize=(15,8))
    plt.xlabel('Academic Year', fontsize=15)
    plt.ylabel('Number of staff', fontsize=15)
    plt.title("Comparison of time series predictions for the next 3 years", fontsize=20)

    years = range(init_year, init_year + len(data) + pred_pd)
    data = merged_df["Total"]
    mksz = 12

    # Plot actual data
    plt.plot(years[:len(data)], data, label="Actual", marker='s', markersize = mksz, color="black", linewidth=3)
    plt.ylim(70, 120)
    
    arima_error = ARIMA_func.error(data, arima_preds[1:-3])
    var_error = ARIMA_func.error(data, var_preds[:-3])
    poly_error = ARIMA_func.error(data, poly_preds[:-3])
 
    arima_label_train = "ARIMA"
    var_label_train = "VAR"
    reg_label_train = "Regression"
    cl_bl = '#5873A3'
    ft_sz = "large"
    # Plot ARIMA
    plt.plot(years, arima_preds[1:], color='red', label=arima_label_train, marker='o',
             markersize=mksz, markerfacecolor='none', linewidth=3)
    plt.fill_between(years[len(data):], arima_lower, arima_upper, color='red', alpha=0.25, label='ARIMA 95% Confidence Interval')
    
    # Plot VAR
    plt.plot(years, var_preds, color=cl_bl, label=var_label_train, marker='x', markersize=mksz, markerfacecolor='none', 
            linewidth=3)
    plt.fill_between(years[len(data):], var_lower, var_upper, color=cl_bl, alpha=0.3, label='VAR 95% Confidence Interval')
    
    # Plot Regression
    plt.plot(years, poly_preds, color='purple', label=reg_label_train, marker='^', markersize=mksz,
             markerfacecolor='none', linewidth=3)
    
    # Plot vertical lines to differentiate training, testing, and prediction
    plt.axvline(x=2021, linestyle="dashed", linewidth=2, color="gray")
    plt.axvline(x=2024, linestyle="dashed", linewidth=2, color="gray")
    
    plt.legend(loc='upper left', fontsize=14)
    plt.show()


    ## Ultimate ultimate Final - remove confidence intervals
    plt.figure(figsize=(9.5,6))
    plt.xlabel('Academic Year', fontsize=ft_sz)
    plt.ylabel('Number of staff', fontsize=ft_sz)
    plt.title("Comparison of time series predictions for the next 3 years", fontsize=ft_sz)

    years = range(init_year, init_year + len(data) + pred_pd)
    data = merged_df["Total"]
    mksz = 12
    cl_bl = '#5873A3'
    ft_sz = "large"
    ln_wd = 1.5

    # Plot actual data
    plt.plot(years[:len(data)], data, label="Actual", marker='s', markersize = mksz, color="black", linewidth=ln_wd)
    plt.ylim(70, 120)
    
    arima_error = ARIMA_func.error(data, arima_preds[1:-3])
    var_error = ARIMA_func.error(data, var_preds[:-3])
    poly_error = ARIMA_func.error(data, poly_preds[:-3])
 
    arima_label_train = "ARIMA"
    var_label_train = "VAR"
    reg_label_train = "Regression"

    # Plot ARIMA
    plt.plot(years, arima_preds[1:], color='red', label=arima_label_train, marker='o',
             markersize=mksz, markerfacecolor='none', linewidth=ln_wd)
    # Plot VAR
    plt.plot(years, var_preds, color=cl_bl, label=var_label_train, marker='x', markersize=mksz, markerfacecolor='none', 
            linewidth=ln_wd)
    # Plot Regression
    plt.plot(years, poly_preds, color='purple', label=reg_label_train, marker='^', markersize=mksz,
             markerfacecolor='none', linewidth=ln_wd)
    
    # Plot vertical lines to differentiate training, testing, and prediction
    plt.axvline(x=2021, linestyle="dashed", linewidth=ln_wd, color="gray")
    plt.axvline(x=2024, linestyle="dashed", linewidth=ln_wd, color="gray")
    
    plt.legend(loc='upper left', fontsize=ft_sz)
    plt.show()