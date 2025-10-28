import os
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# ============================
# 1. Load Data
# ============================
data_path = "../milestone1/cleaned_sales_data.csv"
df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])

# ============================
# 2. Create folders
# ============================
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("forecasts", exist_ok=True)

# ============================
# 3. Forecasting function
# ============================
def forecast_product(product_name, product_df, forecast_days=30):
    product_df = product_df[["Date", "Units Sold"]].rename(columns={"Date": "ds", "Units Sold": "y"})
    product_df = product_df.groupby("ds").sum().reset_index()

    results = []

    # Prophet
    prophet = Prophet(yearly_seasonality=True, daily_seasonality=False)
    prophet.fit(product_df)
    future = prophet.make_future_dataframe(periods=forecast_days)
    forecast_prophet = prophet.predict(future)
    joblib.dump(prophet, f"models/prophet_{product_name}.pkl")

    y_true = product_df["y"].values
    y_pred_prophet = forecast_prophet["yhat"].iloc[:len(y_true)].values
    results.append(["Prophet", mean_absolute_error(y_true, y_pred_prophet), 
                               np.sqrt(mean_squared_error(y_true, y_pred_prophet))])

    # ARIMA
    arima = ARIMA(product_df["y"], order=(5,1,0))
    arima_fit = arima.fit()
    joblib.dump(arima_fit, f"models/arima_{product_name}.pkl")

    y_pred_arima = arima_fit.fittedvalues
    y_true_arima = product_df["y"].iloc[-len(y_pred_arima):].values
    results.append(["ARIMA", mean_absolute_error(y_true_arima, y_pred_arima),
                             np.sqrt(mean_squared_error(y_true_arima, y_pred_arima))])

    # LSTM
    series = product_df["y"].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    series_scaled = scaler.fit_transform(series)

    X, y = [], []
    look_back = 5
    for i in range(len(series_scaled)-look_back):
        X.append(series_scaled[i:i+look_back,0])
        y.append(series_scaled[i+look_back,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    lstm_model.save(f"models/lstm_{product_name}.h5")

    y_pred_lstm_scaled = lstm_model.predict(X, verbose=0)
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled)
    y_true_lstm = series[look_back:]
    results.append(["LSTM", mean_absolute_error(y_true_lstm, y_pred_lstm),
                             np.sqrt(mean_squared_error(y_true_lstm, y_pred_lstm))])

    # Save individual forecast
    forecast_prophet[["ds", "yhat"]].to_csv(f"forecasts/{product_name}_forecast.csv", index=False)

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(product_df["ds"], product_df["y"], label="Actual Sales")
    plt.plot(forecast_prophet["ds"], forecast_prophet["yhat"], label="Prophet Forecast")
    plt.legend()
    plt.title(f"Forecast for {product_name}")
    plt.savefig(f"plots/{product_name}_forecast.png")
    plt.close()

    # Summary
    summary_df = pd.DataFrame(results, columns=["Model","MAE","RMSE"])
    print(f"\nForecasting summary for {product_name}:\n", summary_df)
    return summary_df

# ============================
# 4. Run for sample products
# ============================
sample_products = ["P0001","P0002"]  # Add more if needed
all_summaries = []

for prod in sample_products:
    prod_df = df[df["Product ID"]==prod]
    if len(prod_df) > 10:
        summary = forecast_product(prod, prod_df)
        all_summaries.append(summary)
    else:
        print(f"Not enough data for {prod}, skipping.")

# Combined summary CSV
combined_summary = pd.concat(all_summaries, keys=sample_products)
combined_summary.to_csv("forecasts/model_performance_summary.csv")
print("\n✅ All forecasts, models, and plots saved.")

# ============================
# 5. Create Combined Forecast CSV for Dashboard
# ============================
combined_forecast_path = "../milestone2/data/forecast_results.csv"
all_forecasts = []

for prod in sample_products:
    prod_forecast = pd.read_csv(f"forecasts/{prod}_forecast.csv")
    prod_forecast["Product ID"] = prod
    prod_forecast.rename(columns={"ds":"date","yhat":"forecast_best"}, inplace=True)
    all_forecasts.append(prod_forecast)

combined_df = pd.concat(all_forecasts, ignore_index=True)
combined_df.to_csv(combined_forecast_path, index=False)
print(f"✅ Combined forecast saved to {combined_forecast_path}")
