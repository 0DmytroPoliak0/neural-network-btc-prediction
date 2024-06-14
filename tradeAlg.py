import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
file_path = '' # enter your path to a btc dataset
data = pd.read_csv(file_path)

# Print the first few rows to verify column names
print(data.head())
print(data.columns)

# Convert the date column to datetime
data['date'] = pd.to_datetime(data['date'])  # Adjust column name if needed

# Set the date column as the index
data.set_index('date', inplace=True)  # Adjust column name if needed

# Sort the data by date
data = data.sort_index()

# Use only the 'close' column for prediction
prices = data['close'].values.reshape(-1, 1)  # Adjust column name if needed

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences for training
sequence_length = 60
X, y = [], []
for i in range(len(scaled_prices) - sequence_length):
    X.append(scaled_prices[i:i + sequence_length])
    y.append(scaled_prices[i + sequence_length])
X, y = np.array(X), np.array(y)

# Split the data into training and test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Make predictions
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Prepare data for plotting
data['Predicted'] = np.nan
data.iloc[-len(predicted_prices):, data.columns.get_loc('Predicted')] = predicted_prices.flatten()

# Define buy and sell signals based on predictions
data['Signal'] = 0
data['Signal'][-len(predicted_prices):] = np.where(data['Predicted'][-len(predicted_prices):] > data['Predicted'].shift(1)[-len(predicted_prices):], 1, 0)
data['Position'] = data['Signal'].diff()

# Plot the actual and predicted prices
plt.figure(figsize=(14, 7))
plt.plot(data['close'], label='Close Price', alpha=0.5)
plt.plot(data['Predicted'], label='Predicted Price', alpha=0.75)

# Plot buy signals
plt.plot(data[data['Position'] == 1].index, data['Predicted'][data['Position'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')

# Plot sell signals
plt.plot(data[data['Position'] == -1].index, data['Predicted'][data['Position'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('Bitcoin Price Prediction and Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
