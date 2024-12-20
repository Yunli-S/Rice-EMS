import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, DistanceMetric
import matplotlib.pyplot as plt

# Define a function to visualize the correlation matrix of a DataFrame.
def correlation(df, show, save, fig_name):
    """
    Calculates and visualizes the correlation matrix of a DataFrame as a heatmap.

    Parameters:
    df : The DataFrame for which the correlation matrix is to be calculated and visualized.
    show: A boolean variable indicating whether to display the plot.
    save: A boolean variable indicating whether to save the plot to a file.
    fig_name: The filename for the plot if the save option is True.

    Returns:
    None
    """
    # Utilize the euclidean distance metric for potential future extensions.
    dist = DistanceMetric.get_metric('euclidean')
    
    # Calculate the correlation matrix from the DataFrame.
    correlation_matrix = df.corr()
    
    # Initialize the plot with a specified figure size.
    plt.figure(figsize=(8, 7))
    
    # Create a heatmap to visualize the correlation matrix.
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"fontsize":7})
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)

    if save:
        plt.savefig('./data_visualization/' + fig_name)
    
    if show:
        plt.show()
    else:
        plt.cla()
        plt.clf()


# Load and prepare the dataset for the first objective - analyzing call volume correlations.
df_call_volume = pd.read_csv('./data/monthly_data_16_23.csv') # Import the call volume data.

# Clean the data by removing the 'YearMonth' column and renaming relevant columns to
# enhance understandability and to standardize the naming convention.
df_call_volume = df_call_volume.drop(['YearMonth'], axis=1)
new_names = {
    'Call Count': 'Call Volume',
    'Number of Special Events': 'Number of SE',
    'Undergraduate full-time, non-visiting': 'UG Enrollment',
    'Graduate full-time, non-visiting': 'GR Enrollment',
    'Total (including part-time, visiting)': 'Total Enrollment',
    'Number_of_Employee': 'Num of Employee'
}
df_call_volume.rename(columns=new_names, inplace=True)

# Generate and save a correlation plot for the first objective.
print(correlation(df_call_volume,show=True, save=True, fig_name = "objective1_correlation"))
# objetive1_correlation = correlation(df_call_volume)
# print(show_save(False, True, objetive1_correlation))

# Load and prepare the dataset for the third objective.
df_merge = pd.read_csv('./data/obj3_with_0.csv') # Import data for the third objective.

# Data cleaning steps include removing a redundant column, replacing zero values with
# NaN for proper handling, and dropping rows with missing values to ensure the integrity
# of the correlation analysis.
df_merge_wo_0 = df_merge.drop(['Year'], axis=1)
df_merge_wo_0 = df_merge_wo_0.replace(0, np.nan)
df_merge_wo_0 = df_merge_wo_0.dropna()

# Generate and save a correlation plot for the third objective.
print(correlation(df_merge_wo_0,show=True, save=True, fig_name = "objective3_correlation"))