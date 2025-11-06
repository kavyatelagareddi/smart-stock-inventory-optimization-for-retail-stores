# -------------------------------------------------------------
# Milestone 1: Data Preprocessing & Exploratory Data Analysis
# Generates: cleaned_sales_data.csv
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Hide tkinter emoji font warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tkinter")

# ✅ Fix emoji/font rendering issues
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = "DejaVu Sans"
matplotlib.rcParams['font.sans-serif'] = ["DejaVu Sans"]

sns.set_style("whitegrid")

print("\n🔍 Loading Dataset...")

# -------------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------------
df = pd.read_csv("data_set.csv")   # <- ensure this file is inside milestone1 folder

print("\n✅ Data loaded successfully!")
print(f"📊 Shape: {df.shape}")
print("\n📝 Sample Rows:\n", df.head())

# -------------------------------------------------------------
# 2. Data Cleaning
# -------------------------------------------------------------
print("\n🧹 Cleaning data...")

# Convert date column to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])    # Remove invalid date rows

# Fill missing numerical values
numeric_cols = [
    "Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast",
    "Price", "Discount", "Competitor Pricing", "Supplier Lead Time (days)",
    "Product Rating", "Return Rate", "Stockout Indicator", "Customer Footfall"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

if "Holiday/Promotion" in df.columns:
    df["Holiday/Promotion"] = df["Holiday/Promotion"].fillna(0)

# Remove duplicates
initial_rows = df.shape[0]
df = df.drop_duplicates()
removed = initial_rows - df.shape[0]

print(f"✅ Removed {removed} duplicate rows.")
df = df.sort_values(by=["Product ID", "Date"])

print("\n📌 Data Types After Cleaning:\n")
print(df.dtypes)

# -------------------------------------------------------------
# 3. Dataset Insights
# -------------------------------------------------------------
print("\n📊 Summary Statistics:")
print(df.describe(include="all"))

print("\n📈 Missing Values:")
print(df.isnull().sum())

print("\n✅ Unique Products:", df["Product ID"].nunique())
print("✅ Unique Stores:", df["Store ID"].nunique())
print("✅ Time Range:", df["Date"].min(), "→", df["Date"].max())

# -------------------------------------------------------------
# 4. Feature Engineering
# -------------------------------------------------------------
print("\n🧱 Creating new features...")

df["units_sold_ma7"] = df.groupby("Product ID")["Units Sold"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

df["lag_1_units_sold"] = df.groupby("Product ID")["Units Sold"].shift(1).fillna(0)

print("✅ Features added: units_sold_ma7, lag_1_units_sold")

# -------------------------------------------------------------
# 5. Exploratory Data Analysis (Graphs)
# -------------------------------------------------------------
print("\n📊 Running EDA... (Close graph windows to continue)\n")

# -------- Graph 1: Sales Trend (per product) --------
sample_products = df["Product ID"].unique()[:3]
for p in sample_products:
    temp = df[df["Product ID"] == p]

    plt.figure(figsize=(12, 5))
    plt.plot(temp["Date"], temp["Units Sold"], label="Daily Units Sold", alpha=0.7)
    plt.plot(temp["Date"], temp["units_sold_ma7"], label="7-Day Moving Avg", linewidth=2)
    plt.title(f"Units Sold Trend - {p}")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.show()

# -------- Graph 2: Units Sold Distribution --------
plt.figure(figsize=(8, 5))
sns.histplot(df["Units Sold"], bins=40, kde=True)
plt.title("Units Sold Distribution")
plt.xlabel("Units Sold")
plt.show()

# -------- Graph 3: Monthly Sales --------
df["month"] = df["Date"].dt.to_period("M")
monthly_sales = df.groupby("month")["Units Sold"].sum().reset_index()

plt.figure(figsize=(12, 5))
plt.plot(monthly_sales["month"].astype(str), monthly_sales["Units Sold"], marker="o")
plt.title("Monthly Units Sold Trend")
plt.xticks(rotation=45)
plt.show()

# -------- Graph 4: Correlation Heatmap --------
plt.figure(figsize=(10, 7))
corr = df[numeric_cols + ["units_sold_ma7", "lag_1_units_sold"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------------------------------------------
# 6. Save cleaned dataset
# -------------------------------------------------------------
output_file = "cleaned_sales_data.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Preprocessed dataset saved as → {output_file}")
print("\n🎉 Milestone-1 Completed! Ready for Forecasting (Milestone-2).\n")
