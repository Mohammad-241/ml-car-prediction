# %%

print("Testing Different Sample sizes")
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

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
print("This is with 0.4 test_size")


print("")
print("Testing Different Sample sizes")
# Splitting the data sets

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

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
print("This is with 0.5 test_size")

print("")

print("Testing Different Sample sizes")
# Splitting the data sets

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
print("This is with 0.2 test_size")

# %%
