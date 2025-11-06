# ========================================================================
# MILESTONE 2 - FORECASTING (Prophet + ARIMA + LSTM)
# ========================================================================

# ✅ Hide all warnings & logs (TensorFlow, Prophet, ARIMA, Matplotlib)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Hide TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # Disable OneDNN logs for TF

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").disabled = True
logging.getLogger("tensorflow").disabled = True
logging.getLogger("statsmodels").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)


# ========================================================================
# Imports
# ========================================================================
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


# ========================================================================
# 1. Load Data
# ========================================================================
# Use absolute path dynamically
import os
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_PATH, "..", "milestone1", "cleaned_sales_data.csv")

df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"])


# ========================================================================
# 2. Create directories
# ========================================================================
# ========================================================================
# 2. Create directories
# ========================================================================
os.makedirs("models", exist_ok=True)
os.makedirs("forecasts", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("../milestone2/data", exist_ok=True)   # ✅ FIX ADDED

print("\n🔄 Running forecasting on all products...\n")


# ========================================================================
# 3. Forecasting function
# ========================================================================
def forecast_product(product_name, product_df, show_print=True, forecast_days=30):

    product_df = product_df[["Date", "Units Sold"]] \
        .rename(columns={"Date": "ds", "Units Sold": "y"})
    product_df = product_df.groupby("ds").sum().reset_index()

    results = []


    # --------------------- ✅ PROPHET ---------------------
    prophet = Prophet(yearly_seasonality=True, daily_seasonality=False)
    prophet.fit(product_df)

    future = prophet.make_future_dataframe(periods=forecast_days)
    forecast_prophet = prophet.predict(future)

    joblib.dump(prophet, f"models/prophet_{product_name}.pkl")

    y_true = product_df["y"].values
    y_pred_prophet = forecast_prophet["yhat"].iloc[:len(y_true)].values

    results.append(["Prophet",
                    mean_absolute_error(y_true, y_pred_prophet),
                    np.sqrt(mean_squared_error(y_true, y_pred_prophet))])


    # --------------------- ✅ ARIMA (NO CRASH MODE) ---------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima = ARIMA(product_df["y"], order=(5, 1, 0))
            arima_fit = arima.fit()

        joblib.dump(arima_fit, f"models/arima_{product_name}.pkl")

        y_pred_arima = arima_fit.fittedvalues
        y_true_arima = product_df["y"].iloc[-len(y_pred_arima):].values

        results.append(["ARIMA",
                        mean_absolute_error(y_true_arima, y_pred_arima),
                        np.sqrt(mean_squared_error(y_true_arima, y_pred_arima))])

    except Exception:
        results.append(["ARIMA", None, None])  # just skip, no crash


    # --------------------- ✅ LSTM ---------------------
    series = product_df["y"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    X, y = [], []
    look_back = 5
    for i in range(len(series_scaled) - look_back):
        X.append(series_scaled[i:i + look_back, 0])
        y.append(series_scaled[i + look_back, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X, y, epochs=3, verbose=0)

    lstm_model.save(f"models/lstm_{product_name}.keras")

    y_pred_lstm = lstm_model.predict(X, verbose=0)
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
    y_true_lstm = series[look_back:]

    results.append(["LSTM",
                    mean_absolute_error(y_true_lstm, y_pred_lstm),
                    np.sqrt(mean_squared_error(y_true_lstm, y_pred_lstm))])


    # Save forecast csv for dashboard
    forecast_prophet[["ds", "yhat"]].to_csv(
        f"forecasts/{product_name}_forecast.csv", index=False
    )

    # Save forecast plot
    plt.figure(figsize=(10, 6))
    plt.plot(product_df["ds"], product_df["y"], label="Actual Sales")
    plt.plot(forecast_prophet["ds"], forecast_prophet["yhat"], label="Prophet Forecast")
    plt.legend()
    plt.title(f"Forecast - {product_name}")
    plt.savefig(f"plots/forecast_{product_name}.png")
    plt.close()

    summary_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE"])
    summary_df["Product"] = product_name

    if show_print:
        print(f"\nForecasting summary for {product_name}:\n")
        print(summary_df)

    return summary_df



# ========================================================================
# 4. Run forecasting for ALL products (terminal shows only first 2)
# ========================================================================
all_results = pd.DataFrame()
product_list = df["Product ID"].unique()

for idx, prod in enumerate(product_list):
    prod_df = df[df["Product ID"] == prod]
    if len(prod_df) > 10:
        result = forecast_product(prod, prod_df, show_print=(idx < 2))
        all_results = pd.concat([all_results, result], ignore_index=True)



# ========================================================================
# 5. MODEL COMPARISON GRAPH (ONLY FIRST PRODUCT)
# ========================================================================
first_product = all_results["Product"].unique()[0]
filtered = all_results[all_results["Product"] == first_product]

plt.figure(figsize=(12, 5))

# MAE subplot
plt.subplot(1, 2, 1)
plt.bar(filtered["Model"], filtered["MAE"], color=["royalblue", "orange", "green"])
plt.title(f"MAE - {first_product}")
plt.ylabel("Mean Absolute Error")

# RMSE subplot
plt.subplot(1, 2, 2)
plt.bar(filtered["Model"], filtered["RMSE"], color=["royalblue", "orange", "green"])
plt.title(f"RMSE - {first_product}")
plt.ylabel("Root Mean Square Error")

plt.suptitle(f"Model Comparison (Prophet vs ARIMA vs LSTM)\n{first_product}", fontsize=14)
plt.savefig("plots/model_comparison_single_product.png")
plt.close()

print(f"\n✅ Clean model comparison generated for {first_product}")
print("   → saved as: plots/model_comparison_single_product.png\n")


# ========================================================================
# 6. Save combined forecast for dashboard (milestone4)
# ========================================================================
combined = []

for prod in product_list:
    f = pd.read_csv(f"forecasts/{prod}_forecast.csv")
    f["Product ID"] = prod
    f.rename(columns={"ds": "date", "yhat": "forecast_best"}, inplace=True)
    combined.append(f)

pd.concat(combined).to_csv("../milestone2/data/forecast_results.csv", index=False)

print("✅ Forecast files saved: milestone2/forecasts/")
print("✅ Model files saved:    milestone2/models/")
print("✅ Plots saved:          milestone2/plots/")
print("\n🎉 Forecasting Completed Successfully!\n")
