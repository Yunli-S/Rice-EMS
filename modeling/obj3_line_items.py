import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import scipy.stats as stats

def show_save(show, save, fig_name):
    """
    This function will save the current plot to the 'data_visualization' directory with the provided filename if 'save' is True. 

    Parameters:
    show: A boolean variable indicating whether to display the plot.
    save: A boolean variable indicating whether to save the plot to a file.
    fig_name: The filename for the plot if the save option is True.

    Returns:
    None
    """
    if save:
        plt.savefig('./modeling/' + fig_name)
    
    if show:
        plt.show()
        return
    # clears the current axes of the current figure, removing all content from it, such as lines, texts, labels, and other elements.
    plt.cla()
    # removes all axes in the figure and creates a clean figure with no axes at all.
    plt.clf()

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def fit_standard_scaler(df):
    """
    Applies standard scaling to a DataFrame.

    Parameters:
    df : A DataFrame containing the numerical features to be scaled.

    Returns:
    scaled_data : A numpy array where each feature in the input DataFrame 
    has been scaled to have a mean of zero and a standard deviation of one.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data

def plot_arima_arimax(idx, data, train, test, train_forecasts, test_forecasts, forecasts, init_year):
    '''
    Plots the actual vs predicted values from arima and arimax in a well defined line plot
    inputs:
    - data: the whole data set used (training and testing)
    - test: testing dataset
    - test_forecast: the forecasted values for the testing data
    - forecasts: the forecasted values for applying the model to the entire dataset
    - init_year: first year on the x-axis for specific graphing purpose
    output: N/A (the plot)
    '''
    years = range(init_year, init_year + len(data) + 3)
    for i in range(2):
        df_train_forecast = pd.DataFrame(train_forecasts[i], index = train.index)
        df_test_forecast = pd.DataFrame(test_forecasts[i], index=test.index)
        df_forecast = pd.DataFrame(forecasts[i], index=range(test.index[-1]+1, test.index[-1]+len(forecasts[i])+1))
        print(df_forecast)
        df_pred = pd.concat([df_train_forecast, df_test_forecast,df_forecast], ignore_index= True)
        if i == 0:
            axes[idx].plot(years[0:], df_pred, color='red', label='ARIMA', marker='o', markerfacecolor = 'none', linewidth = 1.5)
        #elif i == 1:
        #    plt.plot(years[0:5], df_train_forecast, color='purple', label='ARIMAX Pred on train set', marker='*')
        #    plt.plot(years[5:7], df_test_forecast, color='purple', label='ARIMAX Pred on test set', marker='o')
        #    plt.plot(years[7:], df_forecast, color='purple', label='ARIMAX Forecast', marker='x')
    

if __name__ == '__main__':
    df_budget_general = pd.read_csv("./data/obj3_dataset_knn.csv")
    
    # need future values for features: ['Total Enrollment', 'Total REMS Staff', 'Call Volume']
    future_feature = {
        'Total Enrollment': [8972, 8972, 8972],
        'Total REMS Staff': [95, 99, 106],
        'Call Volume': [791, 838, 888]
    }
    future_feature = pd.DataFrame(future_feature)

    train_size = 5

    model_colors = {
        'KNeighborsRegressor': 'orange'
    }
    
    models = [KNeighborsRegressor(n_neighbors= 2)]
    model_names = ['KNeighborsRegressor']
    
    df_rmses = {}

    num_plots = 5
    fig, axes = plt.subplots(num_plots, 1, figsize = (9, 12))

    i = 0
    for label in ['Vehicle Maintenace', 'IC Housing', 'Insurance', 'Medical Supplies', 'Uniforms']:
        # Prepare the features (X) and target (y)
        X = df_budget_general[['Total Enrollment', 'Total REMS Staff', 'Call Volume']]
        y = df_budget_general[label]
        X_scaled = fit_standard_scaler(X)

        row_with_zero = (y == 0)
        index_with_zero = -1
        index_with_zero = y.index[row_with_zero]
        X = X.drop(X.index[index_with_zero])
        y = y.drop(y.index[index_with_zero])

        # Split the data into training and test sets
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        df_rmse = pd.DataFrame(columns=model_names, index=['Training RMSE', 'Testing RMSE'])
        
        test_rmse_lst = []

        axes[i].plot(df_budget_general[['Academic Year (FY - 1)']], y, color='black', marker = 's', label='Actual', linestyle='-', linewidth = 1.5)

        # Combine the curve of ARIMA
        if label == "Uniforms":
            X_arima = df_budget_general["Uniforms"]
            init_year = 2016 # start of the time series
            size_arima = int(len(X_arima) * 0.75)
            train_arima, test_arima = X_arima[0:size_arima], X_arima[size_arima:len(X)]
            train_forecast_models = [[6833.190578728357, 7229.405440907034, 7911.636411629603, 7141.038685407949, 7184.85719516417], [6618.771385956332, 6072.637978298713, 9289.562066610048, 7714.764612793717, 6825.643098633226]]
            test_forecast_models = [[6833.190578728357, 7229.405440907034], [7526.580142005745, 8478.843323455323]]
            forecasts_models = [[7737.042866996036, 7054.280201303352, 7105.124592674773], [6409.711586533616, 7029.919816051949, 8103.2839043252425]]
            plot_arima_arimax(i, X, train_arima, test_arima, train_forecast_models, test_forecast_models, forecasts_models, init_year)

        if label == "Vehicle Maintenace":
            X_arima = df_budget_general["Vehicle Maintenace"]
            init_year = 2016 # start of the time series
            size_arima = int(len(X_arima) * 0.75)
            train_arima, test_arima = X_arima[0:size_arima], X_arima[size_arima:len(X)]
            train_forecast_models = [[7604.539857267255, 9789.101627442327, 8517.392351121553, 6551.173536250386, 5324.898740516082], [7982.550857170888, 8631.002079111633, 8038.505411531835, 7095.108065363309, 4792.386008149776]]
            test_forecast_models = [[7604.539857267255, 9789.101627442327], [9320.840868998286, 11878.60247966298]]
            forecasts_models = [[6062.831366860177, 6927.502101805478, 8230.943566877315], [2541.150011216913, 4452.757387427067, 6036.233905233554]]
            plot_arima_arimax(i, X, train_arima, test_arima, train_forecast_models, test_forecast_models, forecasts_models, init_year)


        if label == "IC Housing":
            X_arima = df_budget_general["IC Housing"]
            init_year = 2016 # start of the time series
            size_arima = int(len(X_arima) * 0.75)
            train_arima, test_arima = X_arima[0:size_arima], X_arima[size_arima:len(X)]
            train_forecast_models = [[11676.969934006192, 10886.24161452022, 11027.074328946183, 11722.796702569118, 12262.852129510995], [11276.85393835705, 10215.10577293099, 12932.472142766072, 12231.707608250483, 13935.327174226613]]
            test_forecast_models = [[11676.969934006192, 10886.24161452022], [12677.78383518152, 11681.83161218888]]
            forecasts_models = [[11300.04414136757, 9921.699706425325, 13538.697524432073], [10655.575034546071, 8587.26411921449, 16174.789072667367]]
            plot_arima_arimax(i, X, train_arima, test_arima, train_forecast_models, test_forecast_models, forecasts_models, init_year)

        if label == "Insurance":
            X_arima = df_budget_general["Insurance"]
            init_year = 2016 # start of the time series
            size_arima = int(len(X_arima) * 0.75)
            train_arima, test_arima = X_arima[0:size_arima], X_arima[size_arima:len(X)]
            train_forecast_models = [[4556.800629773015, 6923.382934399986, 6424.801379504978, 5291.3359570716475, 7483.000733503259], [5601.101230949103, 6073.835112925367, 5205.15490039592, 6250.657717892026, 7845.686788969297]]
            test_forecast_models = [[4556.800629773015, 6923.382934399986], [6335.372915057037, 5907.469864774129]]
            forecasts_models = [[8701.356913404292, 8268.796122147483, 7953.345511133842], [9156.932840819263, 8819.7615834339, 8288.19952077258]]
            plot_arima_arimax(i, X, train_arima, test_arima, train_forecast_models, test_forecast_models, forecasts_models, init_year)

        if label == "Medical Supplies":
            X_arima = df_budget_general["Medical Supplies"]
            init_year = 2016 # start of the time series
            size_arima = int(len(X_arima) * 0.75)
            train_arima, test_arima = X_arima[0:size_arima], X_arima[size_arima:len(X)]
            train_forecast_models = [[9151.573613714316, 10638.896590906934, 14648.345912361683, 14685.540590822964, 9806.478679761458], [13013.471936296426, 4347.827843127204, 19796.36145357827, 10046.774664076876, 14686.168444519499]]
            test_forecast_models = [[9151.573613714316, 10638.896590906934], [14177.462744842302, 8209.78477042912]]
            forecasts_models = [[10217.48022048734, 9144.821288227438, 10692.875263298587], [11684.6467250987, 5503.512066161322, 15555.87241638033]]
            plot_arima_arimax(i, X, train_arima, test_arima, train_forecast_models, test_forecast_models, forecasts_models, init_year)


        for model in models:
            model_name = model.__class__.__name__
            model_color = model_colors[model_name]

            model.fit(X_train, y_train)
            
            y_train_pred = pd.Series(model.predict(X_train))
            train_rmse_value = root_mean_squared_error(y_train, y_train_pred)

            y_test_pred = pd.Series(model.predict(X_test))
            test_rmse_value = root_mean_squared_error(y_test, y_test_pred)

            test_rmse_lst.append(test_rmse_value)

            df_rmse.loc['Training RMSE', model_name] = train_rmse_value
            df_rmse.loc['Testing RMSE', model_name] = test_rmse_value

            # Calculate the standard deviation of the prediction errors
            train_prediction_errors = y_train - y_train_pred
            train_std_dev = np.std(train_prediction_errors)

            test_prediction_errors = y_test - y_test_pred
            test_std_dev = np.std(test_prediction_errors)

            # Assuming a normal distribution, compute the 95% confidence interval
            # This is an approximation and might not be perfectly accurate for all distributions of errors
            train_ci = 1.96 * train_std_dev
            test_ci = 1.96 * test_std_dev

            # Get the academic years for plotting
            train_time_units = df_budget_general.iloc[0:train_size]['Academic Year (FY - 1)']
            test_time_units = df_budget_general.iloc[train_size:]['Academic Year (FY - 1)']

            model_full_name = model.__class__.__name__
            abbreviated_name = ''.join([letter for letter in model_full_name if letter.isupper()])

            best_model_name = model_names[test_rmse_lst.index(min(test_rmse_lst))]  
            best_model_color = model_colors[best_model_name]
            best_model_name = 'KNN'
            best_model = KNeighborsRegressor(n_neighbors= 2)
            best_model.fit(X, y)
            future_predictions = pd.Series(best_model.predict(future_feature))

            # Adding future years to your plot
            future_years = pd.Series([2023, 2024, 2025])  # Replace with actual future years

            time_unit = pd.concat([train_time_units, test_time_units, future_years], ignore_index= True)
            predictions = pd.concat([y_train_pred, y_test_pred, future_predictions], ignore_index=True)

            axes[i].plot(time_unit, predictions, label=f'{best_model_name}', color=best_model_color, marker='x', linewidth = 1.5)

        best_model_name = model_names[test_rmse_lst.index(min(test_rmse_lst))]  # Selecting the model with the smallest Testing RMSE
        best_model_color = None
       
        axes[i].set_ylabel(f'{label} Cost', fontsize = 'large')
        axes[i].legend(loc='upper left', fancybox=True, framealpha=0.5, fontsize = 'large')

        df_rmses[label] = df_rmse
        print(label)
        print(df_rmse)
        print()

        i += 1
    axes[4].set_xlabel('Academic Year', fontsize = 'large')
    show_save(show = True, save = True, fig_name=f'Objective 3 result')