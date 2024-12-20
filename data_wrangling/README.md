# README of data_wrangling

- `Dataset_For_Obj1.py`: Extracts features potentially related to Objective 1 (predicting call volumes) from data sources. Generates the dataset `monthly_data_06_23.csv` for modeling.

- `Dataset_For_Obj2.ipynb`: Extracts features potentially related to Objective 2 (predicting the number of staff and the number of hours off-campus rooms are used) from data sources. Generates the dataset `staff_data_model.csv` for modeling.

- `Dataset_For_Obj3.py`: Extracts features potentially related to Objective 3 (predicting five line items) from data sources. Generates the dataset `obj3_dataset_knn.csv` for modeling.

- `Feature_Selection_For_Obj1.py`: Applies PCA and NMF to see whether the number of employees, which do not have data in early years, should be included in the dataset for Objective 1. The result shows we do not need to include this feature in the final dataset for Objective 1.

- `PCA`: Applies PCA for objective 1. Draw the biplot for the 5 variables to show their influences on the major 2 principal components.
