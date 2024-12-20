import pandas as pd
import matplotlib.pyplot as plt
import warnings


def Convert_DateType_To_YearMonth(df):
    """
    Converts the 'Date' column in the given DataFrame to a 'YearMonth' format (YYYY-MM).

    Parameters:
    df : The DataFrame containing the 'Date' column to be converted.

    Returns:
    df: The DataFrame with the new 'YearMonth' column added.
    """
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format= True)
    df['YearMonth'] = df['Date'].dt.strftime('%Y-%m')

    return df;
    

def Sum_Up_Monthly_Feature(df, name_feature):
    """
    Sums up the count of occurrences of name_feature for each 'YearMonth' in the DataFrame.

    Parameters:
    df : The DataFrame with 'YearMonth' as one of its columns.
    name_feature : The name to be given to the feature column containing the counts.

    Returns:
    A DataFrame with 'YearMonth' and the newly created `name_feature` columns.
    """
    monthly_calls = df.groupby('YearMonth').size().reset_index(name= name_feature)

    return monthly_calls


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # Compute the sum of calls in each months in 2006/01 - 2018/04
    df_yearly_06_18 = pd.read_csv('./data/EMS Stats Jan 2006 - April 2018.csv')
    # df_yearly_06_18.head()

    # Typos in Date
    df_yearly_06_18.iloc[8240, df_yearly_06_18.columns.get_loc('Date')] = '5/2018'
    df_yearly_06_18.iloc[6101, df_yearly_06_18.columns.get_loc('Date')] = '1/2015'
    df_yearly_06_18.iloc[1291, df_yearly_06_18.columns.get_loc('Date')] = '3/2008'
    df_yearly_06_18.iloc[1572, df_yearly_06_18.columns.get_loc('Date')] = '9/2008'
    df_yearly_06_18.iloc[1573, df_yearly_06_18.columns.get_loc('Date')] = '9/2008'
    df_yearly_06_18.iloc[7711, df_yearly_06_18.columns.get_loc('Date')] = '8/2017'
    df_yearly_06_18.iloc[7713, df_yearly_06_18.columns.get_loc('Date')] = '8/2017'
    df_yearly_06_18.iloc[7714, df_yearly_06_18.columns.get_loc('Date')] = '8/2017'
    df_yearly_06_18.iloc[7715, df_yearly_06_18.columns.get_loc('Date')] = '8/2017'
    df_yearly_06_18.iloc[7694, df_yearly_06_18.columns.get_loc('Date')] = '8/2017'
    df_yearly_06_18.iloc[7695, df_yearly_06_18.columns.get_loc('Date')] = '8/2017'
    df_yearly_06_18.iloc[6909, df_yearly_06_18.columns.get_loc('Date')] = '4/2016'
    df_yearly_06_18.iloc[6910, df_yearly_06_18.columns.get_loc('Date')] = '4/2016'
    df_yearly_06_18.iloc[6911, df_yearly_06_18.columns.get_loc('Date')] = '4/2016'
    df_yearly_06_18.iloc[6912, df_yearly_06_18.columns.get_loc('Date')] = '4/2016'
    df_yearly_06_18.iloc[7610, df_yearly_06_18.columns.get_loc('Date')] = '5/2017'

    monthly_calls = Convert_DateType_To_YearMonth(df_yearly_06_18)
    monthly_calls = Sum_Up_Monthly_Feature(monthly_calls, 'Call Count')

    # Compute the sum of calls in each months in 2018/05 - 2023/12
    df_yearly_18_23 = pd.read_csv('./data/EMS Stats May 2018 - Dec 2023.csv')
    
    # one typo
    df_yearly_18_23.iloc[427, df_yearly_06_18.columns.get_loc('Date')] = '01/2019'
    # the first 48 rows are already recorded in the 2006-2018 csv.
    df_yearly_18_23 = df_yearly_18_23.iloc[48:]
    df_yearly_18_23 = df_yearly_18_23[df_yearly_18_23['Date'] != '01/1900']

    monthly_calls_2 = Convert_DateType_To_YearMonth(df_yearly_18_23)
    monthly_calls_2 = Sum_Up_Monthly_Feature(monthly_calls_2, 'Call Count')

    df_calls_combined = pd.concat([monthly_calls, monthly_calls_2], ignore_index=True)

    df_calls_combined = df_calls_combined.groupby('YearMonth')['Call Count'].sum().reset_index()

    # comment it to prevent repetitive uploads
    # df_calls_combined.to_csv('../data/monthly_calls_dataset.csv', index=False)
    # print("Monthly call summary has been saved to 'monthly_calls_dataset.csv'.")

    # Add the number of special events into the table
    df_se = pd.read_csv('./data/SE_06_18.csv')

    df_se['Date'] = pd.to_datetime(df_se['Date'], format='%b-%y')

    df_se['YearMonth'] = df_se['Date'].dt.strftime('%Y-%m')

    df_se_counts = Sum_Up_Monthly_Feature(df_se, 'Number of Special Events')

    df_calls_se = pd.merge(df_calls_combined, df_se_counts, on='YearMonth', how='left')


    df_en = pd.read_csv('./data/Event_Number.csv')
    df_en = df_en.iloc[0:12]

    df_en_new = pd.melt(df_en, id_vars=['Month x Year'], var_name='Year', value_name='Count')

    month_to_num = {
        'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06',
        'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'
    }
    df_en_new['YearMonth'] = df_en_new['Year'] + '-' + df_en_new['Month x Year'].map(month_to_num)
    df_en_new = df_en_new[['YearMonth', 'Count']]

    df_calls_se = pd.merge(df_calls_se, df_en_new, on='YearMonth', how='outer')

    df_calls_se.loc[180:216, 'Number of Special Events'] = df_calls_se.loc[180:216, 'Count']

    df_calls_se.loc[180:216]

    df_calls_se = df_calls_se[['YearMonth', 'Call Count', 'Number of Special Events']]

    df_calls_se = df_calls_se.drop(df_calls_se.loc[149:179].index)

    df_calls_se = df_calls_se.reset_index(drop = True)

    df_calls_se['Number of Special Events'] = df_calls_se['Number of Special Events'].fillna(0)

    df_calls_se.head(5)


    # Combine student enrollment data into the final dataset
    df_enroll = pd.read_csv('./data/Monthly_Enrollment.csv')

    df_enroll['YearMonth'] = pd.to_datetime(df_enroll['YearMonth'], infer_datetime_format=True)
    df_calls_se['YearMonth'] = pd.to_datetime(df_calls_se['YearMonth'], infer_datetime_format=True)

    # Ensure both 'YearMonth' columns are in the same format for accurate merging
    df_enroll['YearMonth'] = df_enroll['YearMonth'].dt.to_period('M')
    df_calls_se['YearMonth'] = df_calls_se['YearMonth'].dt.to_period('M')

    # Merge the 'number of special events' column from se table into monthly_dataset based on YearMonth
    df_calls_se_enroll = pd.merge(df_calls_se, df_enroll, on='YearMonth', how='inner')

    print("The first five rows in df_calls_se_enroll:")
    print(df_calls_se_enroll.head())
    print(df_calls_se_enroll.shape)

    # df_calls_se_enroll.to_csv('../data/monthly_data_06_23.csv', index=False)



    # Combine employee enrollment data into the final dataset
    df_employee =  pd.read_csv('./data/Monthly Employee Enrollment.csv')

    df_employee['YearMonth'] = pd.to_datetime(df_employee['Year_Month'], infer_datetime_format=True)
    # df_calls_se['YearMonth'] = pd.to_datetime(df_calls_se['YearMonth'], infer_datetime_format=True)

    # Ensure both 'YearMonth' columns are in the same format for accurate merging
    df_employee['YearMonth'] = df_employee['YearMonth'].dt.to_period('M')
    # df_calls_se['YearMonth'] = df_calls_se['YearMonth'].dt.to_period('M')

    # Merge the 'number of special events' column from se table into monthly_dataset based on YearMonth
    df_16_23 = pd.merge(df_calls_se_enroll , df_employee[['YearMonth', 'Number_of_Employee']], on='YearMonth', how='inner')

    print("The first five rows in df_16_23:")
    print(df_16_23.head())
    print(df_16_23.shape)

    # df_16_23.to_csv('../data/monthly_data_16_23.csv', index=False)