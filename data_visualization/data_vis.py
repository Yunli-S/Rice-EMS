import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns

def line_graph(feature_1, feature_2, xlabel, ylabel, fig_name, marker = None, markersize = None, show = True, save = False):
    """
    Plot a line graph of two features with the specified features and parameters.
    If 'show' is True, the plot will be displayed. If 'save' is True, the plot will be saved to `fig_name`.

    Parameters:
    feature_1: An array-like variable containing the data for the x-axis. Most calls will have Year/Month as its parameter value in this project
    feature_2: An array-like variable containing the data for the y-axis.
    xlabel: A string variable representing the label for the x-axis.
    ylabel: A string variable representing the label for the y-axis.
    fig_name: A string variable specifying the file name to save the figure.
    marker: A string variable specifying the marker style for the line plot. Default is None.
    markersize: A integer variable specifying the size of the markers. Default is None.
    show: An optional boolean indicating whether to display the plot; default value is True.
    save: An optional boolean indicating whether to save the plot to the specified fig_name; default value is False.

    Returns:
    None
    """
    plt.plot(feature_1, feature_2, marker = marker, markersize = markersize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    show_save(show = show, save = save, fig_name = fig_name)


def histogram(feature, bins, xlabel, ylabel, fig_name, ticklabels = None, color = 'skyblue', edgecolor = 'black', 
              tick_par_axis = 'x', tick_par_size = 8, tick_par_rot = 0, show = True, save = False, weights = None):
    height, bins, patches = plt.hist(feature, bins = bins, color = color, edgecolor=edgecolor, weights = weights)
    """
    This function creates a histogram plot with the specified features and parameters. 
    If 'show' is True, the plot will be displayed. If 'save' is True, the plot will be saved to `fig_name`.

    Parameters:
    feature: An array-like variable containing the data to be binned in the histogram.
    bins: Either an integer specifying the number of bins or a sequence defining the bin edges.
    xlabel: A string variable representing the label for the x-axis.
    ylabel: A string variable representing the label for the y-axis.
    fig_name: A string variable specifying the file name to save the figure.
    ticklabels: An optional list of string variables representing custom labels for x-axis ticks; default value is None.
    color: An optional string variable specifying the fill color for the bars of the histogram; default value is 'skyblue'.
    edgecolor: An optional string variable specifying the color of the bar borders; default value is 'black'.
    tick_par_axis: An optional string variable indicating which axis ('x' or 'y') the tick parameters will be applied to; default value is 'x'.
    tick_par_size: An optional integer specifying the font size of the tick labels; default value is 8.
    tick_par_rot: An optional integer defining the rotation angle of the tick labels; default value is 0.
    show: An optional boolean indicating whether to display the plot; default value is True.
    save: An optional boolean indicating whether to save the plot to the specified fig_name; default value is False.
    weights: An optional array-like variable providing weights for each element in feature; default value is None.

    Returns:
    None
    """

    ticks = [(patch.get_x() + (patch.get_x() + patch.get_width()))/2 for patch in patches] ## or ticklabels

    plt.xticks(ticks, ticklabels)

    plt.xlabel(xlabel)
    plt.tick_params(axis = tick_par_axis,labelsize=tick_par_size, rotation=tick_par_rot)
    plt.ylabel(ylabel)
    plt.tight_layout()

    show_save(show = show, save = save, fig_name = fig_name)

def multi_line_graph(time, feature_1, feature_2, label_1, label_2, xlabel, ylabel, fig_name, legend_loc = None, 
                     show = True, save = False):
    """
    The function plots two lines on the same graph, with the option to display the plot immediately, save it to a file, or both.

    Parameters:
    time: An array-like variable containing the time points at which the data was collected.
    feature_1: An array-like variable representing the data to be plotted as the first line on the y-axis.
    feature_2: An array-like variable representing the data to be plotted as the second line on the y-axis.
    label_1: A string variable that describes the first line plot, which will be used in the legend.
    label_2: A string variable that describes the second line plot, which will be used in the legend.
    xlabel: A string variable representing the label for the x-axis, describing the time variable.
    ylabel: A string variable representing the label for the y-axis, describing the features.
    fig_name: A string variable that determines the name of the file where the plot will be saved if saving is enabled.
    legend_loc: An optional variable that determines the location of the legend on the plot. It defaults to None.
    show: A boolean value that, if set to True, displays the plot. The default is True.
    save: A boolean value that, if set to True, saves the plot to the file path specified by fig_name. The default is False.

    Returns:
    None
    """
    plt.plot(time, feature_1, label=label_1)
    plt.plot(time, feature_2, label=label_2)
    plt.ylabel(ylabel=ylabel)
    plt.xlabel(xlabel=xlabel)
    plt.legend(loc=legend_loc)
    
    show_save(show = show, save = save, fig_name = fig_name)

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
        plt.savefig('./data_visualization/' + fig_name)
    
    if show:
        plt.show()
        return
    # clears the current axes of the current figure, removing all content from it, such as lines, texts, labels, and other elements.
    plt.cla()
    # removes all axes in the figure and creates a clean figure with no axes at all.
    plt.clf()


if __name__ == '__main__':

    """
    Plot a line graph showing:
    REMS Staff by Year
    """
    df = pd.read_csv("./data/Staffing.csv")

    line_graph(feature_1=df['Year'], feature_2=df['Total People'], xlabel='School year', ylabel='Total staff', fig_name='REMS staff by school year', show = False, save = True)

    df1 = pd.read_csv("./data/EMS Stats Jan 2006 - April 2018.csv")
    df2 = pd.read_csv("./data/EMS Stats May 2018 - Dec 2023.csv")

    df_all_stats = pd.concat([df1, df2])

    """ 
    Plot a histogram showing:
    Amount of REMS Calls by days of the week
    """
    unique_days = df_all_stats['Day'].unique()

    # Fix day typos
    df_all_stats['Day'] = df_all_stats['Day'].replace('Tuesdasy', 'Tuesday')
    df_all_stats['Day'] = df_all_stats['Day'].replace('Saturady', 'Saturday')
    df_all_stats['Day'] = df_all_stats['Day'].replace('Satuday', 'Saturday')
    df_all_stats['Day'] = df_all_stats['Day'].replace('Thusday', 'Thursday')
    df_all_stats['Day'] = df_all_stats['Day'].replace('Wedensday', 'Wednesday')
    df_all_stats['Day'] = df_all_stats['Day'].replace('#REF!', 'Unknown')
    df_all_stats['Day'] = df_all_stats['Day'].replace('Saturday ', 'Saturday')
    df_all_stats['Day'] = df_all_stats['Day'].replace(np.nan, 'Unknown')
    df_all_stats['Day'].fillna("Unknown")

    df_all_stats['Call Location'].fillna("Unknown location")

    df_all_stats['Day'] = pd.Categorical(df_all_stats['Day'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday', 'Sunday', 'Unknown'], ordered=True)
    df_all_stats.sort_values(by='Day', ascending=True, inplace=True, ignore_index=True)

    ticklabels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday', 'Sunday', 'Unknown']

    histogram(feature=df_all_stats['Day'], ticklabels = ticklabels, bins= 8, xlabel='Days of the week', ylabel='Frequency', fig_name= 'Amount of REMS calls by days of the week', tick_par_rot= 5, show = False, save = True)

    """
    Plot a histogram showing:
    Top 5 REMS Calls Location
    """
    df_all_stats['Call Location'] = df_all_stats['Call Location'].replace('Undergraduate College 聽聽', 'Undergraduate College')
    df_all_stats['Call Location'] = df_all_stats['Call Location'].replace('Undergraduate College 鑱借伣', 'Undergraduate College')
    df_all_stats['Call Location'] = df_all_stats['Call Location'].replace('Football Stadium', 'Rice Stadium')
    df_all_stats['Call Location'] = df_all_stats['Call Location'].replace('Rice Recreational Center', 'The Rec')
    df_all_stats['Call Location'] = df_all_stats['Call Location'].replace(np.nan, 'Unknown')
    df_all_stats['Call Location'].fillna("Unknown")

    # Get the top 5 values from the column
    top_locs = df_all_stats['Call Location'].value_counts().nlargest(5)
    histogram(top_locs.index, bins=5, ticklabels=top_locs.index, xlabel='Top 5 Locations', ylabel='Frequency', fig_name='Top 5 REMS Call Locations', tick_par_rot = 5, weights=top_locs.values, show=False, save=True)


    """
    Plot a histogram showing:
    Top 5 REMS Call Natures
    """
    top_injuries = df_all_stats['Nature of Injury/Illness'].value_counts().nlargest(5)
    histogram(top_injuries.index, bins = 5, weights = top_injuries.values, xlabel= 'Top 5 Reasons for the Call', ylabel= 'Frequency', fig_name='Top 5 REMS Call Natures', tick_par_size= 6, tick_par_rot=5, show = False, save = True)


    """
    Plot a multiple-line graph showing:
    Curve of paid hours, volunter hours
    """
    unique_ppl = df_all_stats['Affiliation'].unique()
    print(unique_ppl)

    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Visitor/Public ', 'Visitor/Public')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Visitor/ Public', 'Visitor/Public')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Visitor', 'Visitor/Public')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Vitisor/Public', 'Visitor/Public')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('No patients found', 'No patient')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('No Patient', 'No patient')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Staff', 'Faculty/Staff')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Staff', 'Faculty/Staff')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Staff', 'Faculty/Staff')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Staff', 'Faculty/Staff')

    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student-Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Grad Student', 'Graduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student- Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student-Grad', 'Graduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student-GRAD', 'Graduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Stduent-Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student-Graduate', 'Graduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Studen -Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student - Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student - Grad', 'Graduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student-McMurty', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student - Sid', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Sutdent - Grad', 'Graduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student -Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student - Undergraduate ', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student-  Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student - Graduate', 'Graduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student - Grauate', 'Graduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Stidemt - Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student - Graduate ', 'Graduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Stuent - Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student- Graduate', 'Graduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student - Graudate', 'Graduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student -  Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Stuent - Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student  - Undergraduate', 'Undergraduate')
    df_all_stats['Affiliation'] = df_all_stats['Affiliation'].replace('Student  -  Undergraduate', 'Undergraduate')
    
    ## Event hours over 2021-2023
    eventhours = pd.read_csv("./data/Event_Hours.csv")
    print(eventhours)
    print(eventhours.head())
    totalhours = []
    volhours = []
    paidhours = []
    date = []
    for year in ['2021','2022','2023']:
        for i in range(len(eventhours)-3):
            if i % 3 == 2:
                totalhours.append(eventhours[year][i])
            elif i % 3 == 0:
                paidhours.append(eventhours[year][i])
                date.append(str(int((i/3) + 1)) + '/' + year[-2:])
            else:
                volhours.append(eventhours[year][i])
    print(totalhours[-2])
    plt.plot(date, totalhours)
    plt.plot(date, paidhours)
    plt.plot(date, volhours)
    plt.tick_params(axis = 'x',labelsize=7, rotation=-45)
    plt.xlabel('Month')
    plt.ylabel('Hours')
    plt.legend(["Total hours", "Paid Hours", "Volunteer Hours"])
    show_save(show=False, save = True, fig_name= 'Curve of paid hours, volunter hours')
    
    """
    Plot a line graph showing:
    Total enrollment
    """
    callvol = pd.read_csv("./data/Call Volume.csv")
    callvol.drop(callvol[callvol['Academic year beginning in Fall of:'] == 2025].index, inplace = True)
    line_graph(feature_1=callvol['Academic year beginning in Fall of:'], feature_2=callvol['Total Enrollment'], xlabel = 'Academic Year beginning in Fall of', ylabel= 'Total Enrollment', marker = 'o', markersize = 5, fig_name= 'Total Student Enrollment', show = False, save = True)

    """
    Plot a histogram showing:
    Rice faculty/staff data (excludes student employees)
    """
    callvol_wo_na = callvol.dropna(how = 'any')
    histogram(feature=callvol_wo_na['Academic year beginning in Fall of:'], ticklabels= callvol_wo_na['Academic year beginning in Fall of:'], weights=callvol_wo_na['Total Employees'], bins = len(callvol_wo_na['Total Employees']), xlabel= "Year", ylabel='Total Employees', fig_name='Total Rice Employees', show = False, save = True)

    """
    Plot a histogram showing:
    Calls by all months
    """
    month_df = pd.read_csv("./data/monthly_calls_dataset.csv")

    month_df["month"] = month_df["YearMonth"].str[5:]
    
    month_call_count = month_df.groupby('month')
    group_sums = month_call_count.sum()
    group_sums.reset_index(inplace=True)

    histogram(feature=group_sums['month'], weights=group_sums['Call Count'], ticklabels=group_sums['month'], xlabel= 'Months', ylabel='Number of Calls', fig_name='Call Volume by All Months', bins = len(group_sums['Call Count']), show = False, save = True)
    

    """
    Plot a histogram showing:
    Sum of Call Counts per year (exclude months we do not record)
    """
    month_df = pd.read_csv("./data/monthly_data_06_23.csv")

    month_df['Year'] = month_df['YearMonth'].astype(str).str.slice(0, 4)

    # Convert 'Year' to numeric if it's not already
    month_df['Year'] = pd.to_numeric(month_df['Year'])

    # Assuming 'call count' is the name of the column you want to sum up
    # Aggregate 'call count' by 'Year'
    yearly_call_count = month_df.groupby('Year')['Call Count'].sum()

    bins = [year - 0.5 for year in yearly_call_count.index] + [max(yearly_call_count.index) + 0.5]
    histogram(feature= yearly_call_count.index, bins = bins,
              xlabel = 'Year', ylabel='Sum of Call Counts', ticklabels= yearly_call_count.index,
              fig_name= 'Call Volume per Year (exclude months that are not recorded)', tick_par_rot=45, 
              weights= yearly_call_count.values, show = False, save = True)


    """
    Plot a multiple-line graph showing:
    Education vs. Training Hours
    """
    df_edu_train_hours = pd.read_csv("./data/Education_Training_Hours.csv")
    multi_line_graph(time = df_edu_train_hours['Year'], feature_1=df_edu_train_hours['Education Hours'], 
                     feature_2=df_edu_train_hours['Training Hours'], label_1="Education Hours", label_2="Training Hours", 
                     xlabel="School year", ylabel="Hours", fig_name="Education and Training Hours by Year", legend_loc= "upper left", show = False, save = True)

    """
    Plot a multiple-line graph showing:
    The number of Paid Staff vs. The nubmer of Volunteer Staff
    """
    df_staffing = pd.read_csv("./data/Staffing.csv")
    df_staff = df_staffing[['Year', 'Total UG', 'Part Timers', 'Total People']]
    totalppl = df_staffing['Total People']
    other = totalppl - df_staffing['Total UG'] - df_staffing['Part Timers']
    df_staff = df_staff.rename(columns={"Total UG": "Volunteer", "Part Timers": "Paid", "Total People": "Total"})
    df_staff['Other'] = other
    df_staff = df_staff[['Year', 'Volunteer', 'Paid', 'Other', 'Total']]
    df_staff["Paid"] = df_staff["Paid"] + df_staff["Other"]
    df_staff = df_staff.drop(columns=['Other'])
    df_staff

    multi_line_graph(time = df_staff['Year'], feature_1 = df_staff['Volunteer'], feature_2 = df_staff['Paid'], 
                     label_1 = "Volunteer", label_2 = "Paid", xlabel = "School Year", ylabel = "Amount of Staff", 
                     fig_name = "Volunteer and Paid Staff by Year", legend_loc = "upper left", show = False, save = True)


    """
    Plot a line graph showing:
    Off campus room shifts over months
    """
    df_staff_loc = pd.read_csv("./data/REMS off-campus room.csv")
    df_staff_loc = df_staff_loc[:84]
    df_staff_loc = df_staff_loc.rename(columns={"Unnamed: 0": "Date"})
    df_staff_loc = df_staff_loc[['Date', 'Shifts w room (12hr)', 'Hours with Room', 'Rolling Average', 'Month / Year']]
    df_staff_loc["Year"] = df_staff_loc["Date"].str[:4]
    new_row = pd.DataFrame({'Date':['2023/12/1'], 'Hours with Room':[384.0],'Year': ['2024']})
    df_staff_loc = pd.concat([df_staff_loc, new_row], ignore_index=True)
    df_staff_loc

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(12))
    line_graph(feature_1=df_staff_loc['Date'], feature_2=df_staff_loc['Hours with Room'], xlabel = 'Year', ylabel='Hours Off-Campus Room Used', fig_name='Off-Campus Room Usage in Hours', show = False, save=True)
    

