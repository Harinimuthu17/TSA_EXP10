## Name : M.Harini
## Reg No : 212222240035
## Date:

# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
 
## AIM:

To implement SARIMA model using python for Microsoft Stock Prediction.

## ALGORITHM:

1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
   
## PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

file_path = 'Microsoft_Stock (1).csv'
data = pd.read_csv(file_path)
# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Sort data by date to ensure time series continuity
data = data.sort_values('date').set_index('date')

# Plot the 'close' time series
plt.plot(data.index, data['close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Close Price Time Series')
plt.show()

# Check stationarity of the 'close' time series
def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['close'])

# Plot ACF and PACF
plot_acf(data['close'])
plt.show()
plot_pacf(data['close'])
plt.show()

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data['close'][:train_size], data['close'][train_size:]

# Define SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Make predictions
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('SARIMA Model Predictions on Stock Close Price')
plt.legend()
plt.show()
```

## OUTPUT:

### Original Data:
![Screenshot 2024-11-11 110234](https://github.com/user-attachments/assets/a40cdf19-0334-4d65-82ac-b8311d8f6873)


### ACF anf PACF Representation:

![Screenshot 2024-11-11 110527](https://github.com/user-attachments/assets/4ccf9491-b474-4af1-b3a3-a1d11bf326bd)


![Screenshot 2024-11-11 110539](https://github.com/user-attachments/assets/8557648d-9d97-4fcf-bc3b-a0df78038574)

![Screenshot 2024-11-11 110549](https://github.com/user-attachments/assets/cf0c88d9-083c-43e7-9555-188525f48ff2)



### SARIMA Prediction Representation:


![Screenshot 2024-11-11 110601](https://github.com/user-attachments/assets/058d5904-ed5d-4242-91ac-758199e35ae4)



## RESULT:
Thus the program run successfully based on the SARIMA model for Microsoft stock prediction.
