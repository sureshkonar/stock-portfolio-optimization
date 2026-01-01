import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prophet
def predict_prophet(price_series, days=30):
    df = pd.DataFrame({"ds": price_series.index, "y": price_series.values})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    predicted_price = forecast['yhat'].iloc[-1]
    current_price = price_series.iloc[-1]
    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "expected_return_pct": (predicted_price-current_price)/current_price*100
    }

# LSTM
def predict_lstm(price_series, days_ahead=30):
    data = price_series.values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    window = 60
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i,0])
        y.append(scaled_data[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1],1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    last_window = scaled_data[-window:]
    predictions = []
    for _ in range(days_ahead):
        x_input = last_window.reshape((1,window,1))
        pred = model.predict(x_input, verbose=0)
        predictions.append(pred[0,0])
        last_window = np.append(last_window[1:], pred[0,0])

    predicted_price = scaler.inverse_transform(np.array(predictions).reshape(-1,1))[-1,0]
    current_price = price_series.iloc[-1]
    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "expected_return_pct": (predicted_price-current_price)/current_price*100
    }

def classify_price_trend(current_price, predicted_price):
    if predicted_price > current_price:
        return "Up"
    return "Down"
