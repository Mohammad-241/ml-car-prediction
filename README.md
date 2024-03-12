[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10479431&assignment_repo_type=AssignmentRepo)

Title: Building a Car Price Prediction Model Using Regression Analysis
==============================
Abstract: 
==============================
The automobile industry is a highly competitive market, and the price of a car is one of the most critical factors in determining the purchase decision of a customer. Accurately predicting the price of a car is essential for car dealerships, manufacturers, and customers. With the vast amount of data available on cars, building a price prediction model using regression analysis can be highly beneficial for its users. The end result of this project is to create a functional and accurate model to give the expected prices of a car given the specifciations it has.

The objective of this project is to create a regression model that can predict the price of a car based on specific features such as make, model, year, mileage, fuel type, and transmission type. The model will be trained on a dataset of cars, which includes both new and used cars and the model will also account for inflation over time to provide a more realisitic and fitting model.

The methodology for creating the model will involve the following steps:

Data Collection: A dataset of cars has been sourced from third party websites that include the data of popular mid size sedans from bmw and toyota rangng from the 20th century to today. This dataset is based on cars sold in Germany. Furthermore, an external dataset named inflation is used to determine the inflation rate on car prices in Europe. By applying the inflation rate to the car prices in our dataset, we can adjust for the impact of inflation and provide a more accurate representation of the real value of the cars at the time of sale. By accounting for inflation, we can make more informed comparisons and analysis of the prices of cars over time.

Data Cleaning and Preprocessing: The dataset will be cleaned and preprocessed to remove any missing or duplicate values, the data will be examined to see if there are any strings in columns where it should only be numbers as well as remive the money symbol to be able to work with the data. additionally data given in a range will be updated to display the mean of the range allowing us to work with the data.

Feature Selection: The most relevant features will be selected given the results when various techniques such as correlation analysis with respect to the target variable, the analysis of pairplots as well as the feature_importance function from sklearn on a random forrest regresor model to obtain the top 10 features. Furthermore, Random forest regressor to determine feature importance, sorts the features by importance, and then selects the top k features using the SelectKBest function and f_regression score function. In this case, we will choose 4 features that are good out of the result we got from the 10 features. 

Model Selection: Various regression models such as Linear Regression and  Random Forest Regression will be evaluated to determine the best model that provides the highest accuracy. additioally, testing with hyperparameters and ridge regularization will be completed  and the highest r squared score will be selected

Model Evaluation: The performance of the selected model will be evaluated using various metrics such as R-squared, MAE and RMSE.

The outcome of this project will be a robust and accurate car price prediction model that can be used by car dealerships, manufacturers, and customers to estimate the price of a car accurately. The model will provide insights into the factors that influence car prices, and it will assist customers in making informed purchase decisions.



Project Instructions
==============================

This repo contains the instructions for a machine learning project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    ├── data
    │   ├── dataset.csv       <- Data from third party sources.
    │   ├── inflation.csv        <- Data from Euro Central Bank providing HICP for all of Europe
    │   
    │   
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation 
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── data           <- Scripts to download or generate data and pre-process the data
       │   └── cardata.py
       │  
       │
       ├── features       <- Scripts to turn raw data into features for modeling
       │   └── build_features.py
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── predict_model.py
       │   └── train_model.py
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py           

