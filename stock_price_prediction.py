# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Set the random seed for reproducibility
np.random.seed(42)

# Create a sample dataset
# In a real scenario, you would load data from a CSV file or an API
# Here, we create a DataFrame with dates and random stock prices
dates = pd.date_range(start='2020-01-01', periods=100)  # 100 days of data
prices = np.random.rand(100) * 100  # Random stock prices between 0 and 100
data = pd.DataFrame({'Date': dates, 'Price': prices})  # Create DataFrame
data.set_index('Date', inplace=True)  # Set the date as the index

# Display the first few rows of the dataset
print("Sample Data:")
print(data.head())

# Visualize the stock prices over time
plt.figure(figsize=(10, 5))
plt.plot(data['Price'], label='Stock Price', color='blue')
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Split the data into training and testing sets
# We will use 80% of the data for training and 20% for testing
train_size = int(len(data) * 0.8)  # 80% for training
train, test = data[:train_size], data[train_size:]  # Split the data

# Fit the ARIMA model
# ARIMA model parameters: (p, d, q)
# p = number of lag observations included in the model
# d = number of times that the raw observations are differenced
# q = size of the moving average window
model = ARIMA(train, order=(5, 1, 0))  # Using (5, 1, 0) as an example
model_fit = model.fit()  # Fit the model

# Make predictions on the test set
predictions = model_fit.forecast(steps=len(test))  # Forecast for the length of the test set
test['Predicted'] = predictions  # Add predictions to the test DataFrame

# Evaluate the model's performance
# Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(test['Price'], test['Predicted'])  # Calculate MSE
rmse = np.sqrt(mse)  # Calculate RMSE

# Print the RMSE
print(f'Root Mean Squared Error: {rmse:.2f}')

# Visualize the actual vs predicted stock prices
plt.figure(figsize=(10, 5))
plt.plot(train, label='Training Data', color='green')  # Training data
plt.plot(test['Price'], label='Actual Prices', color='blue')  # Actual prices
plt.plot(test['Predicted'], label='Predicted Prices', color='red')  # Predicted prices
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
