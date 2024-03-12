
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import os # Used to interact with the file system

dir_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(dir_path))
data_dir = os.path.join(parent_dir, "data")

# Construct the path to the CSV file using the data directory
csv_path = os.path.join(data_dir, "dataset.csv")
csv_path2 = os.path.join(data_dir, "inflation.csv")


'''
                DATA INITLIZATION AND PROCESSING

'''

df = pd.read_csv(csv_path)
pd.DataFrame(df)
df.head
df.head(100)

inflation = pd.read_csv(csv_path2)
print(inflation.head(10))

pd.DataFrame(inflation)
inflation.head(100)


def currency_to_float(s):
    return int(s.replace('£', '').replace(',',''))

def convert_range_to_mean(df_column):
        # Remove the '$' and ',' characters from the input column
        
    df_column = df_column.str.replace('£', '').str.replace(',', '')
    
    # Split the input column into two parts based on the '-' character
    df_column = df_column.str.split('-', expand=True)
    
    # Convert the two parts into floats
    lower = df_column[0].astype(float)
    upper = df_column[1].astype(float)
    
    # Calculate the mean of the two parts
    mean = (lower + upper) / 2
    
    return mean

def convert_range_to_mean_mileage(df_column):
        # Remove the '$' and ',' characters from the input column
    mean_values = []
    
    for value in df_column:
        if isinstance(value, str):
            # If the value is a string, split it into two parts based on the '-' character
            value_parts = value.split('-')
            
            # Convert the two parts into floats
            lower = float(value_parts[0].strip().replace(' mpg', ''))
            if len(value_parts) > 1:
                upper = float(value_parts[1].strip().replace(' mpg', ''))
            else:
                upper = lower
            
            # Calculate the mean of the two parts
            mean_mpg = int((lower + upper) / 2)
            
            # Append the mean value to the list
            mean_values.append((mean_mpg))
        else:
            # If the value is not a string, assume it's a float and append it to the list followed by "mpg"
            mean_values.append((value))
    
    return mean_values

def inflate_prices(df, inflation):
    # Merge car_data and inflation_data on year and month columns
    merged_data = pd.merge(df, inflation, on=["year", "month"])

    # Apply inflation adjustment to price column
    merged_data["price_inflated"] = merged_data["New Price"] *(merged_data["HICP"].astype(int) / 100).round().astype(int)

    # Insert the new column into the original car dataframe
    df["price_inflated"] = merged_data["price_inflated"]

    return df

df.fillna(0,inplace =True)
df.replace('NaN',0,inplace =True)
df.replace('N/a',0,inplace =True)
df.replace('-','0',inplace =True)

inflation.fillna(0,inplace =True)
inflation.replace('NaN',0,inplace =True)
inflation.replace('N/a',0,inplace =True)
inflation.replace('-','0',inplace =True)

#fix year start to be a year and not range
df['Year start'] = df['Year start'].astype(str).str[:4].apply(lambda x: int(x) if len(x) >= 4 else x)
df = df.rename(columns={'Year start': 'year'})
# convert 'Production Start' to datetime format
df['Production start'] = pd.to_datetime(df['Production start'])

# extract month from 'Production Start' and create new 'Month' column
df['month'] = df['Production start'].dt.month



df['0-60 mph (secs)'] = df['0-60 mph (secs)'].astype(float).round().astype(int)
df['Torque (Nm)'] = df['Torque (Nm)'].astype(float).round().astype(int)
df['CO2 Emissions (g/km)'] = df['CO2 Emissions (g/km)'].astype(float).round().astype(int)
df['Cylinders'] = df['Cylinders'].astype(float).round().astype(int)

print(df['Power (bhp)'])
print(df.head())

print('before the functions: \n')
print(df['Fuel consumption (mpg)'])
print('\n')
print(df['Used price range'])

# prepare used price and mileage
df['Fuel consumption (mpg)'] = convert_range_to_mean_mileage(df['Fuel consumption (mpg)'])
df['Used price range'] = convert_range_to_mean(df['Used price range'])

print('after the functions: \n')
print(df['Fuel consumption (mpg)'])
print('\n')
print(df['Used price range'])
print('\n')

#Prepare New Price
df['New Price'] = df['New Price'].astype(str)  # convert column to string type
df['New Price'] = df['New Price'].apply(currency_to_float)
print(df['New Price'])


#split the month and year into workable columns
inflation['year'] = inflation['date'].str[:4].astype(int)
inflation['month'] = pd.to_datetime(inflation['date'], format='%Y%b').dt.month.astype(int)

#make a new price column to account for inflation
inflated_prices = inflate_prices(df, inflation)

# Print the updated car dataframe with the "price_inflated" column
print(inflated_prices.head())

#get inflation rate column
df.replace('nan',0,inplace =True)

df['inflation_rate'] = (df['price_inflated'] - df['New Price']) / df['New Price'].astype(float)
df.replace(np.NaN,0,inplace=True)

'''
            REMOVE COLUMNS WE WILL NOT BE USING
'''
df.columns = df.columns.str.strip()
df_modified = df = df.iloc[1:]  
df_modified = df.drop(['Fuel Type', 'Transmission', 'Gearbox', 'Drivetrain', 'Doors', 'EuroNCAP Rating','Status', 'Series (production years start-end)','Adult Occupant','Child Occupant', 'Make', 'Model', 'Trim / Version', 'Series launch year', 'Image URL', 'Miles per pound (mpp)','Year end','Production end','Annual road tax', 'EuroNCAP Rating description','Country', 'Production start','Pedestrian'], axis=1)


#df_modified.to_csv('newdataset.csv', index=False)
