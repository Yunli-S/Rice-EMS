This project was conducted for the Rice Emergency Services Department, utilizing a time series model to assist in forecasting the department's budget.

**Project Description**

Rice Emergency Medical Services (REMS) serves the Rice community and provides them with accessible medical care. After receiving a phone call, REMS dispatches staff to the site of the request. REMS staff then contact the patient and decide the best treatment plan, which could include calling an ambulance for transport to a nearby hospital. Composed mainly of undergraduate volunteers, REMS responds to about 1,000 emergencies each year and provides 5 academic courses in the Department of Kinesiology. 

Since its establishment in 1996, REMS has seen significant increases in student enrollment, call volume (the number of calls), and its number of staff. Due to an expected addition of around 700 students to the student body in the next 3-5 years, Rice plans to build two new residential colleges to accommodate demand. 

REMS would like to predict the expected call volume since this would allow them to be more prepared for the next 3-5 years of future calls. Call volume can be influenced by a number of factors, such as the number of special events, number of students living on campus, etc.

Requests for increased budget occur around May each year for a fiscal year that begins on July 1st. Through predicting call volume, personnel need, and supply requirements, REMS can create an accurate and thorough budget forecast. More efficient financial allotment means that REMS can foster a safer and healthier Rice campus. 

This GitHub repository holds the code and data that our team used in order to predict the following four objectives:
1. Use past data on call volume to predict the future call volume.
2. Use data on personnel to predict future staff growth, as well as future training needs.
3. Use financial information, including the top five line items in primary operations and educational budgets, to predict future expenditures.
4. Collect data on equipment to plan for future equipment needs, including increasing inventory, maintenance costs, storage, and usage.

**Data Description**

For Objective 1, the response variable is Call Volume in each month. The features we use include student enrollment data, employee data, and the number of special events. There are 139 rows (months) in our dataset, starting from 2006. The data we use are in the file "monthly_data_06_23.csv".

For Objective 2, there are three datasets on Staff Growth, Training Needs, and Off-campus Room Usage. The first dataset is on Staff Growth and spanned from 2016 to 2023. The second dataset is on Training Needs and spanned from 2010 to 2023. The last dataset is on Off-campus Room Usage and spanned from 2017 to 2023. The data we use are in the files "staff_data_model.csv", "Education_Training_Hours.csv", and "REMS off-campus room.csv".

For Objective 3, the response variables are five kinds of expenditures REMS spends per year. The features we use include student enrollment data, employee data, and call volume. There are 7 rows (years) in our dataset, starting from 2016. The data we use are in the file "obj3_dataset_knn.csv".

**Code Description**

There are four main folders in this repository. They are as follows: data, data_visualization, data_wrangling, and modeling. More description of their purposes are below.

- data: This folder contains the data used in the following analyses. All data is in csv or excel format and named according to the information it contains. The README in this folder details each dataset’s contents.
- data_visualization: This folder contains code that was used to complete data visualizations for the project. This was done during the data exploration stage.
- data_wrangling: This folder contains code that was used to create and process the given data by the sponsor. 
- modeling: This folder contains code that makes predictions on the data.

**Usage instructions**

To install the libraries we need, first make sure you have installed Python 3.8+ (any version more recent than 3.8) since this is a requirement when installing the latest version of some libraries below. Once you have installed Python 3.8+, you automatically have the package manager called “pip”, a package manager for Python that allows you to install and manage software packages written in Python.

To run our code, you need to install the following libraries:
- numpy
- pandas
- scikit-learn 
- seaborn
- statsmodels
- matplotlib
- tensorflow

To install the following libraries, set Python 3.8+ as the current Python interpreter and run the following command line in Terminal:

python -m pip install <library>

For example, to install numpy, simply run the command:

python -m pip install numpy

Using -m pip instead of just pip can be useful because it ensures that you are using the pip associated with the current Python interpreter you are calling. This is particularly helpful if you have multiple versions of Python installed on your system.

To check the libraries you have installed for the current Python interpreter, run the following command:

_python -m pip list_

Once you have installed all of the libraries above, you can run any .py file under appropriate Python IDE by clicking “Run Python File” or “Run Python File in Dedicated Terminal” or “Run without Debugging” when in the page of the file or clicking “Run Python File in Terminal” when right click on the file name inside the directory bar. For .ipynb file, the results are already saved inside the file after each block of code, but you can run it by clicking “Execute Cell” button near each block or clicking “Run all” in the page of the file.
