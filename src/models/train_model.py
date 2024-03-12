# %%
# Splitting the data sets
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
preprocessing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(preprocessing_path)
import cardata

print("Splitting Data Sets")

df2 = cardata.df_modified #get the processed dataframe
df2 = df2.iloc[1:]  

# Split the data into features and target
features = ['inflation_rate', 'Insurance group', 'Length (mm)', 'Engine Size (cc)']
target = 'price_inflated'
df2.replace(np.nan,0,inplace=True)
print(df2['inflation_rate'])
X = df2[features]
y = df2[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


print("Linear Regression Model vs Random Forest Regression Model")
# Linear Regression Model vs Random Forest Regression Model

# Create a linear regression model
lin_model = LinearRegression()

# Fit the linear regression model to the training data
lin_model.fit(X_train, y_train)

# Make predictions on the testing data using the linear regression model
lin_y_pred = lin_model.predict(X_test)

# Calculate the mean squared error for the linear regression model
lin_mse = mean_squared_error(y_test, lin_y_pred)
print(f"Linear Regression - Mean squared error: {lin_mse:.2f}")

# Calculate the R-squared score for the linear regression model
lin_r2 = r2_score(y_test, lin_y_pred)
print(f"Linear Regression - R-squared score: {lin_r2:.2f}")

# Create a random forest regression model
rf_model = RandomForestRegressor()

# Fit the random forest regression model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the testing data using the random forest regression model
rf_y_pred = rf_model.predict(X_test)

# Calculate the mean squared error for the random forest regression model
rf_mse = mean_squared_error(y_test, rf_y_pred)
print(f"Random Forest Regression - Mean squared error: {rf_mse:.2f}")

# Calculate the R-squared score for the random forest regression model
rf_r2 = r2_score(y_test, rf_y_pred)
print(f"Random Forest Regression - R-squared score: {rf_r2:.2f}")



print("")
print("Training the Random Forest Regression Model with by tuning hyperparameters using Gridsearch")
print('Testing First Hyperparameter')

# Extract the features and target variable
features = ['inflation_rate', 'Insurance group', 'Length (mm)', 'Engine Size (cc)']
target = ['price_inflated']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model with some hyperparameters
model = RandomForestRegressor(random_state=42)

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best hyperparameters: ", grid_search.best_params_)

# Predict the target values for the test set using the best model
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluate the model's performance using mean squared error and r2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("R-squared score: ", r2)

print("")
print('Testing Second Hyperparameter')
# Testing Second Hyperparameter

# Extract the features and target variable
features = ['inflation_rate', 'Insurance group', 'Length (mm)', 'Engine Size (cc)']
target = ['price_inflated']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [5, 7, 9, 11],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model with some hyperparameters
model = RandomForestRegressor(random_state=42)

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best hyperparameters: ", grid_search.best_params_)

# Predict the target values for the test set using the best model
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluate the model's performance using mean squared error and r2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("R-squared score: ", r2)

print("")
print('Testing Third Hyperparameter')
# Third Hyperparameter
# Extract the features and target variable
features = ['inflation_rate', 'Insurance group', 'Length (mm)', 'Engine Size (cc)']
target = ['price_inflated']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [5, 7, 9, 11, 13],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Initialize the model with some hyperparameters
model = RandomForestRegressor(random_state=42)

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
print("Best hyperparameters: ", grid_search.best_params_)

# Predict the target values for the test set using the best model
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluate the model's performance using mean squared error and r2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("R-squared score: ", r2)

print("")
print("Evaluating the random regression model with the best hyperparameter")
# Evaluating the random regression model with the best hyperparameter 

# Make predictions on the testing set
y_pred = rf_model.predict(X_test)

# Calculate mean squared error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("R-squared score: ", r2)

print("")
print("The training model R square score was much better without tuning the hyperparameters")
print("")
#Finding out performance metrics
print("Performance Metrics")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import UndefinedMetricWarning
import warnings
# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# define the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])

# fit the pipeline on the training data
pipe.fit(X_train, y_train)

# evaluate the model on the testing data
y_pred = pipe.predict(X_test)

# calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = pipe.score(X_test, y_test)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UndefinedMetricWarning)
    precision = precision_score(y_test, y_pred.round(), average='weighted')
    recall = recall_score(y_test, y_pred.round(), average='weighted')
    f1 = f1_score(y_test, y_pred.round(), average='weighted')
# print the results
print("Mean squared error: ", mse)
print("R-squared score: ", r2)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)
# apply regularization techniques
pipe_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=0.1))
])

scores = cross_val_score(pipe_reg, X_train, y_train, cv=5, scoring='r2')
print("R-squared score with Ridge regularization: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# %%
