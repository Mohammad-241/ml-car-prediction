# %%
import sys
import os

preprocessing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(preprocessing_path)
import cardata

import numpy as np
df3 = cardata.df_modified
print("Correlation Analysis")
# Compute correlation coefficients between features and target variable
correlations = df3.corr()['price_inflated'].sort_values()

# Print correlation coefficients
print(correlations)

# Visualize correlation matrix using a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df3.corr(), cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
print("")


print("Feature Selection, will select top 10 features")
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Split data into features and target
X = df3.drop(['price_inflated'], axis=1)
y = df3['price_inflated']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Get feature importances
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)

# Select top 10 features
selected_features = feature_importances.head(10).index.tolist()

print(selected_features)


print("Top 4 Features")
print("Random forest regressor to determine feature importance, sorts the features by importance, and selects the top k features based on their importance.\n It then selects the top k features using the SelectKBest function and f_regression score function. In this case, we will choose 4 features that are good out of the result we got previously")
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression


# define X (input features) and y (target variable)
X = df3[['inflation_rate', 'Insurance group', 'Length (mm)', 'Engine Size (cc)', 'Used price range', 'New Price', 'year', 'Height (mm)', 'Power (bhp)', 'CO2 Emissions (g/km)']]
y = df3['price_inflated'] = pd.to_numeric(df3['price_inflated'], errors='coerce')

# use random forest to determine feature importance
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)

# sort features by importance and select top k
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
k = 4
top_k_features = X.columns[indices][:k]

# select top k features using SelectKBest and f_regression
skb = SelectKBest(score_func=f_regression, k=k)
X_top_k = skb.fit_transform(X, y)
top_k_indices = np.argsort(skb.scores_)[::-1][:k]
top_k_features = X.columns[top_k_indices]

print("Top {} features: {}".format(k, top_k_features))
# %%
