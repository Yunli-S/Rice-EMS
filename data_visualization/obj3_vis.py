import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def difference(timeseries, interval=1):
    return timeseries - timeseries.shift(interval)

def seasonal_difference(timeseries, seasonal_period):
    return timeseries - timeseries.shift(seasonal_period)

def test_stationarity(timeseries):
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')  # AIC is used to select the best lag length
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

if __name__ == '__main__':
    obj3_df = pd.read_csv('./data/obj3_dataset_knn.csv')
    dates = obj3_df['Academic Year (FY - 1)']
    values = obj3_df['Vehicle Maintenace']

    # Create a DataFrame
    df = pd.DataFrame({'Date': dates, 'Value': values})

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Value'], label='Sample Time Series')

    plt.title('Time Series Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()  # Display the plot

    timeseries_diff = difference(values.squeeze(), interval=1)
    print(timeseries_diff)
    test_stationarity(timeseries_diff.dropna(inplace = False))

    # Plot ACF:
    plt.figure(figsize=(10,6))
    plot_acf(values, ax=plt.gca(), lags=6)
    #plt.title('Autocorrelation Function')
    plt.show()

    # Plot PACF:
    plt.figure(figsize=(10, 6))
    plot_pacf(timeseries_diff, ax=plt.gca(), lags=6)
    #plt.title('Partial Autocorrelation Function')
    plt.show()