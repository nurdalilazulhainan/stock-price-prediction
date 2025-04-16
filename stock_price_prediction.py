# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
# For this example, we will create a sample dataset
# In a real scenario, you would load data from a CSV file or an API
dates = pd.date_range(start='2020-01-01', periods=100)
prices = np.random.rand(100) * 100  # Random stock prices
data = pd.DataFrame({'Date': dates, 'Price': prices})
data.set_index('Date', inplace=True)

# Visualize the data
plt.figure(figsize=(10, 5))
plt.plot(data['Price'])
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit the ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # (p, d, q) parameters
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))
test['Predicted'] = predictions

# Evaluate the model
mse = mean_squared_error(test['Price'], test['Predicted'])
rmse = np.sqrt(mse)

# Visualize the predictions
plt.figure(figsize=(10, 5))
plt.plot(train, label='Training Data')
plt.plot(test['Price'], label='Actual Prices', color='blue')
plt.plot(test['Predicted'], label='Predicted Prices', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Print RMSE
print(f'Root Mean Squared Error: {rmse}')
