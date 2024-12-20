import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import mean_squared_error, DistanceMetric

# Convert FYXX into 20(XX-1) in the budget table
def convert_fy_to_year(fy_string):
    """
    Converts a fiscal year string (FYXX) to a calendar year.

    Parameters:
    fy_string : A string representing the fiscal year, e.g., 'FY21'.

    Returns:
    The calendar year to which the fiscal year corresponds.
    """
    last_two_digits = int(fy_string[-2:])
    return 2000 + last_two_digits - 1


def convert_academic_to_year(fy_string):
    """
    Converts an academic year string to a calendar year.

    Parameters:
    fy_string : A string representing the academic year, e.g., 'FY21'.

    Returns:
    The calendar year in which the academic year begins.
    """

    first_two_digits = int(fy_string[:2])
    return 2000 + first_two_digits

if __name__ == '__main__':
    df_budget = pd.read_csv("./data/Budget_Forecast.csv")
    
    # delete future prediction from REMS
    df_budget = df_budget[df_budget['Year'] != 'FY24']

    """
    Combine features into the budget table
    """
    
    df_call_volume = pd.read_csv("./data/Call Volume.csv")
    df_call_volume = df_call_volume.rename(columns={'Academic year beginning in Fall of:': 'Year'})

    df_call_volume['Calls (from academic year rpt)'] = []

    # Apply the conversion function to the 'Year' column
    df_budget['Year'] = df_budget['Year'].apply(convert_fy_to_year)

    df_budget.replace(np.nan, 0.0, inplace = True)


    df_staff = pd.read_csv("./data/Staffing.csv")

    ## Convert XX/YY into 20XX in the staff table
    df_staff['Year'] = df_staff['Year'].apply(convert_academic_to_year)

    df_edu = pd.read_csv("./data/Education_Training_Hours.csv")

    # acquire data of graduate enrollment from a montly table

    df_monthly_enrollment = pd.read_csv("./data/Monthly_Enrollment.csv")

    df_monthly_enrollment['YearMonth'] = pd.to_datetime(df_monthly_enrollment['YearMonth'])

    # Filter the rows where the month is August
    august_data = df_monthly_enrollment[df_monthly_enrollment['YearMonth'].dt.month == 8]

    # get a table containing the year and the value for the month of August
    august_yearly_data = august_data.copy()
    august_yearly_data['Year'] = august_yearly_data['YearMonth'].dt.year
    august_yearly_data = august_yearly_data[['Year', 'Graduate full-time, non-visiting']]
    august_yearly_data = august_yearly_data.rename(columns= {'Graduate full-time, non-visiting': 'GR Total Enrollment'}, )
    print(august_yearly_data)

    # Combine total UG enrollment, total enrollment, total employees, and calls into the budget dataset
    df_merge = pd.merge(df_budget, df_call_volume[['Year', 'Total UG enrollment', 'Total Enrollment', 'Total Employees', 'Calls (from academic year rpt)']], on = 'Year', how = 'inner')
    df_merge = df_merge.rename(columns={'Calls (from academic year rpt)': 'Call Volume'})

    # Combine total REMS staff into the budget table
    df_merge = pd.merge(df_merge, df_staff[['Year', 'Total People']], how = 'inner')
    df_merge = df_merge.rename(columns={'Total People': 'Total REMS Staff'})

    # combine education hours and training hours into the budget table
    df_merge = pd.merge(df_merge, df_edu, how = 'inner')

    # combine total graduate enrollment into the budget table
    df_merge = pd.merge(df_merge, august_yearly_data, how = 'inner')

    # df_merge.to_csv('./data/obj3_with_0.csv', index=False)

    # create correlation matrix without considering the rows containing 0's
    df_merge_wo_0 = df_merge.drop(['Year'], axis = 1)
    df_merge_wo_0 = df_merge_wo_0.replace(0, np.nan)
    df_merge_wo_0 = df_merge_wo_0.dropna()

    # Apply KNN imputation on the two 0's
    from sklearn.impute import KNNImputer

    df_merge_knn = df_merge.replace(0, np.nan)

    # exclude 'year' when applying imputation
    col_to_impute = df_merge_knn.columns.difference(['Year'])

    # 2 is a reasonable K value
    imputer = KNNImputer(n_neighbors=2)

    df_merge_knn = pd.DataFrame(imputer.fit_transform(df_merge_knn[col_to_impute]), columns = col_to_impute)

    df_merge_knn = pd.concat([df_merge[['Year']], df_merge_knn], axis = 1)
    df_merge_knn = df_merge_knn.rename(columns = {'Year': 'Academic Year (FY - 1)'})

    print(df_merge_knn)

    # df_merge_knn.to_csv('./data/obj3_dataset_knn.csv', index=False)