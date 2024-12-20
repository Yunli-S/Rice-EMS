import pandas as pd


if __name__ == 'main':
    df = pd.read_csv('./data/monthly_data_06_23.csv')
    # Convert YearMonth to datetime and extract the year
    df['Year'] = pd.to_datetime(df['YearMonth']).dt.year

    # Group by the year and sum the Call Count
    call_volume_per_year = df.groupby('Year')['Call Count'].sum()

    # Display the result
    print(call_volume_per_year)