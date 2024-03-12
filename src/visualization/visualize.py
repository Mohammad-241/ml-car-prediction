
import numpy as np
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os # Used to interact with the file system

preprocessing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(preprocessing_path)
import cardata
df4 = cardata.df_modified
bin_range = [0,1,2,3,4,5,6,7,10,15,20,25,30,35,40,45,50,55,60,65,70,75]

# Create the histogram using sns.histplot()
sns.histplot(data=df4, x='Fuel consumption (mpg)', bins=bin_range)
# Add axis labels and a title
plt.xlabel('MPG')
plt.ylabel('Count')
plt.title('Data distribution')

# Display the plot
plt.show()

bin_range = [0,20000,25000,30000,35000,40000,45000,50000,60000]

# Create the histogram using sns.histplot()

sns.histplot(data=df4, x='New Price', bins=bin_range)
# Add axis labels and a title
plt.xlabel('price')
plt.ylabel('Count')
plt.title('price')



# display the plot
plt.show()

bin_range = [0,2000,4000,6000,8000,10000,12000,14000,16000,20000,25000,30000,35000,40000]

# Create the histogram using sns.histplot()

sns.histplot(data=df4, x='Used price range', bins=bin_range)
# Add axis labels and a title
plt.xlabel('used price')
plt.ylabel('Count')
plt.title('used price')

# Display the plot
plt.show()

bin_range = [50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450]

# Create the histogram using sns.histplot()
df4['Power (bhp)'] = pd.to_numeric(df4['Power (bhp)'], errors='coerce')

# Apply the 'round()' method to the 'Power (bhp)' column
df4['Power (bhp)'] = df4['Power (bhp)'].round()

sns.histplot(data=df4, x='Power (bhp)', bins=bin_range)
# Add axis labels and a title
plt.xlabel('hp')
plt.ylabel('Count')
plt.title('hp distribution')

# Display the plot
plt.show()
#print("pair plot of the variables")
#sns.pairplot(df4)
#plt.show()
X = df4[['year']]
y = df4['price_inflated']

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)
# Create a scatter plot of the data
plt.scatter(X, y)

# Add the regression line to the plot
plt.plot(X, model.predict(X), color='red')

# Set the plot title and axis labels
plt.title('Linear Regression')
plt.xlabel('Year')
plt.ylabel('Price (inflated)')

# Show the plot
plt.show()

X = df4[['Used price range']]
y = df4['price_inflated']


# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)
# Create a scatter plot of the data
plt.scatter(X, y)

# Add the regression line to the plot
plt.plot(X, model.predict(X), color='red')

# Set the plot title and axis labels
plt.title('Linear Regression')
plt.xlabel('Used price range')
plt.ylabel('Price (inflated)')

# Show the plot
plt.show()

X = df4[['Power (bhp)']]
y = df4['price_inflated']

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)
# Create a scatter plot of the data
plt.scatter(X, y)

# Add the regression line to the plot
plt.plot(X, model.predict(X), color='red')

# Set the plot title and axis labels
plt.title('Linear Regression')
plt.xlabel('Power (bhp)')
plt.ylabel('price_inflated')

# Show the plot
plt.show()

X = df4[['CO2 Emissions (g/km)']]
y = df4['price_inflated']

# Create a linear regression model and fit it to the data
model = LinearRegression()
model.fit(X, y)
# Create a scatter plot of the data
plt.scatter(X, y)

# Add the regression line to the plot
plt.plot(X, model.predict(X), color='red')

# Set the plot title and axis labels
plt.title('Linear Regression')
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('price_inflated')

# Show the plot
plt.show()