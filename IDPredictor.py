import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression



DA_Prices_Predictor = pd.read_excel('Data/RegressionIntraday/DayAheadPrices_2021_2024.xlsx')
Demand_Predictor = pd.read_excel('Data/RegressionIntraday/Demand_2021_2024.xlsx')
ID_Prices_Predictor = pd.read_excel('Data/RegressionIntraday/IntradayPrices_2021_2024.xlsx')
OffshoreCapacityFactor_Predictor = pd.read_excel('Data/RegressionIntraday/OffshoreCapacityFactor_2021_2024.xlsx')
OnshoreCapacityFactor_Predictor = pd.read_excel('Data/RegressionIntraday/OnshoreCapacityFactor_2021_2024.xlsx')
SolarCapacityFactor_Predictor = pd.read_excel('Data/RegressionIntraday/SolarCapacityFactor_2021_2024.xlsx')




# Making 15 Min prices hourly
# Ensure the time is in string format before concatenating
DA_Prices_Predictor['time_str'] = DA_Prices_Predictor['(Uhrzeit) von'].astype(str)
ID_Prices_Predictor['time_str'] = ID_Prices_Predictor['(Uhrzeit) von'].astype(str)

# Combine the date and time columns into a single datetime column, with `dayfirst=True`
DA_Prices_Predictor['timestamp'] = pd.to_datetime(DA_Prices_Predictor['Datum von'] + ' ' + DA_Prices_Predictor['time_str'], dayfirst=True)
ID_Prices_Predictor['timestamp'] = pd.to_datetime(ID_Prices_Predictor['Datum von'] + ' ' + ID_Prices_Predictor['time_str'], dayfirst=True)

# Drop the temporary columns
DA_Prices_Predictor.drop(columns=['time_str'], inplace=True)
ID_Prices_Predictor.drop(columns=['time_str'], inplace=True)

# Drop the old date and time columns if you don't need them
DA_Prices_Predictor.drop(columns=['Datum von', '(Uhrzeit) von'], inplace=True)
ID_Prices_Predictor.drop(columns=['Datum von', '(Uhrzeit) von'], inplace=True)

# Set the timestamp as the index
DA_Prices_Predictor.set_index('timestamp', inplace=True)
ID_Prices_Predictor.set_index('timestamp', inplace=True)

# Sort the index to avoid issues during resampling
DA_Prices_Predictor.sort_index(inplace=True)
ID_Prices_Predictor.sort_index(inplace=True)

# Select only numeric columns for resampling
DA_Prices_numeric = DA_Prices_Predictor.select_dtypes(include=['number'])
ID_Prices_numeric = ID_Prices_Predictor.select_dtypes(include=['number'])

# Resampling to hourly average
DA_Prices_hourly = DA_Prices_numeric.resample('h').mean()
ID_Prices_hourly = ID_Prices_numeric.resample('h').mean()

# Remove February 29th if it exists (leap years only)
DA_Prices_hourly = DA_Prices_hourly[~((DA_Prices_hourly.index.month == 2) & (DA_Prices_hourly.index.day == 29))]
ID_Prices_hourly = ID_Prices_hourly[~((ID_Prices_hourly.index.month == 2) & (ID_Prices_hourly.index.day == 29))]


# Creating df containing features and target
# Reset the index of the reference DataFrame
DA_Prices_hourly.reset_index(drop=False, inplace=True)

# Drop the original DateTime column if it exists (it is part of the index now)
if 'DateTime' in DA_Prices_hourly.columns:
    DA_Prices_hourly.drop(columns=['DateTime'], inplace=True)

# Attach (concatenate) the DataFrames horizontally
data = pd.concat([
    DA_Prices_hourly,
    ID_Prices_hourly['ID AEP in €/MWh'].reset_index(drop=True),
    Demand_Predictor['Demand [GWh]'].reset_index(drop=True),
    OffshoreCapacityFactor_Predictor['Capacity Factor'].reset_index(drop=True),
    OnshoreCapacityFactor_Predictor['Capacity Factor'].reset_index(drop=True),
    SolarCapacityFactor_Predictor['Capacity Factor'].reset_index(drop=True)
], axis=1)

# Rename the columns
data.columns = ['DateTime', 'DA_price', 'ID_price', 'Demand', 'Offshore_CF', 'Onshore_CF', 'PV_CF']

data.fillna(0, inplace=True)


# Test Train split
# Features and target variable
X = data[['DA_price', 'Demand', 'Offshore_CF', 'Onshore_CF', 'PV_CF']]
y = data['ID_price']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the model
IDForecaster = LinearRegression()

# Train the model
IDForecaster.fit(X_train, y_train)


