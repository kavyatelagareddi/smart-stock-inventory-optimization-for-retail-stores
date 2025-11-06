## inventory.py - Milestone 3: Advanced Inventory Optimization Logic (Unique Version)

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- Streamlit Page Setup ----------------
st.set_page_config(page_title="📦 Milestone 3: Smart Inventory Optimizer", layout="wide")
st.title("📦 Milestone 3: Smart Inventory Optimization Dashboard")
df = pd.read_csv("../milestone2/data/forecast_results.csv")

# ---------------- Data Loading ----------------
try:
    df = pd.read_csv("../milestone2/data/forecast_results.csv")
except FileNotFoundError:
    st.error("❌ Could not find 'data/forecast_results.csv'. Please ensure it exists in the 'data' folder.")
    st.stop()

st.sidebar.header("⚙️ Configuration")

# ✅ Now df is loaded — safe to detect the forecast column
possible_cols = [
    "forecast_prophet",  # ✅ your column name
    "forecast_best",
    "Forecast",
    "forecast",
    "best_forecast",
    "Predicted",
    "predicted",
    "Forecast_ARIMA",
    "Forecast_LSTM",
]
forecast_col = next((col for col in possible_cols if col in df.columns), None)

if forecast_col is None:
    st.error(f"❌ No forecast column found! Available columns: {list(df.columns)}")
    st.stop()

# ---------------- User Inputs ----------------
products = df["Product ID"].unique()
selected_product = st.sidebar.selectbox("Select Product", products)

lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
ordering_cost = st.sidebar.slider("Ordering Cost ($)", 10, 200, 50)
holding_cost = st.sidebar.slider("Holding Cost ($/unit)", 1, 20, 2)
service_levels = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
z = service_levels[st.sidebar.selectbox("Service Level", list(service_levels.keys()), index=1)]

# ---------------- Inventory Logic ----------------
inventory_plan = []

for product in products:
    subset = df[df["Product ID"] == product]

    if subset[forecast_col].isnull().all():
        continue  # skip if forecast missing

    mean_demand = subset[forecast_col].mean() / 30  # avg daily
    total_demand = subset[forecast_col].sum()
    demand_std = subset[forecast_col].std()

    eoq = np.sqrt((2 * total_demand * ordering_cost) / holding_cost)
    safety_stock = z * demand_std * np.sqrt(lead_time)
    reorder_point = (mean_demand * lead_time) + safety_stock

    inventory_plan.append({
        "Product": product,
        "AvgDailySales": round(mean_demand, 2),
        "TotalDemand": round(total_demand, 2),
        "EOQ": round(eoq, 2),
        "SafetyStock": round(safety_stock, 2),
        "ReorderPoint": round(reorder_point, 2)
    })

inv_df = pd.DataFrame(inventory_plan)

if inv_df.empty:
    st.warning("⚠️ No valid inventory data could be computed. Check your forecast values.")
    st.stop()

# ---------------- ABC Classification ----------------
inv_df["Value"] = inv_df["TotalDemand"] * holding_cost
inv_df = inv_df.sort_values(by="Value", ascending=False)
inv_df["Cumulative%"] = inv_df["Value"].cumsum() / inv_df["Value"].sum() * 100
inv_df["ABC_Category"] = inv_df["Cumulative%"].apply(lambda x: "A" if x <= 20 else ("B" if x <= 50 else "C"))

# ---------------- Dashboard Visualization ----------------
row = inv_df[inv_df["Product"] == selected_product].iloc[0]
weeks = np.arange(1, 9)
inv_level = np.linspace(row["ReorderPoint"] + 100, row["SafetyStock"], 8)

fig, ax = plt.subplots()
ax.plot(weeks, inv_level, marker='o', label="Inventory Level", linewidth=2)
ax.axhline(y=row["ReorderPoint"], color="orange", linestyle="--", label="Reorder Point (ROP)")
ax.axhline(y=row["SafetyStock"], color="red", linestyle="-.", label="Safety Stock")
ax.set_xlabel("Weeks")
ax.set_ylabel("Units in Stock")
ax.set_title(f"Inventory Trend for Product {selected_product}")
ax.legend()
st.pyplot(fig)

# ---------------- Metrics Display ----------------
col1, col2, col3 = st.columns(3)
col1.metric("📊 Reorder Point", f"{row['ReorderPoint']:.2f}")
col2.metric("📦 EOQ", f"{row['EOQ']:.2f}")
col3.metric("🛡️ Safety Stock", f"{row['SafetyStock']:.2f}")

# ---------------- Data Download ----------------
csv_data = inv_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Inventory Plan", csv_data, "inventory_plan.csv", "text/csv")

st.success("✅ Inventory optimization completed successfully!")