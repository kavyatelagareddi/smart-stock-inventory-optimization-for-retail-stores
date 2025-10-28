import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set plot style
sns.set_style('whitegrid')

# 1. Load Data
print("\n🔍 Loading Dataset...")

df = pd.read_csv("data_set.csv")

print("\n✅ Data loaded successfully!")
print(f"📊 Data Shape: {df.shape}")
print("\n📝 Sample Data:\n", df.head())

# 2. Data Cleaning
print("\n🧹 Data Cleaning in Progress...")

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows with missing 'Date'
df = df.dropna(subset=['Date'])

# Fill missing numerical fields with 0
numeric_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast',
                'Price', 'Discount', 'Competitor Pricing',
                'Supplier Lead Time (days)', 'Product Rating', 'Return Rate',
                'Stockout Indicator', 'Customer Footfall']

for col in numeric_cols:
    df[col] = df[col].fillna(0)

# Fill promotion column if missing
if 'Holiday/Promotion' in df.columns:
    df['Holiday/Promotion'] = df['Holiday/Promotion'].fillna(0)

# Remove duplicates
initial_rows = df.shape[0]
df = df.drop_duplicates()
removed_rows = initial_rows - df.shape[0]

# Sort by Product ID and Date
df = df.sort_values(by=['Product ID', 'Date'])

print(f"✅ Removed {removed_rows} duplicate rows.")
print("\n✅ Data types after cleaning:\n", df.dtypes)


# 3. Data Insights

print("\n📊 Dataset Summary Statistics:")
print(df.describe(include='all'))

print("\n📈 Missing Values Report:")
print(df.isnull().sum())

print("\n✅ Number of Unique Products:", df['Product ID'].nunique())
print("✅ Number of Unique Stores:", df['Store ID'].nunique())
print("✅ Time Period Covered:", df['Date'].min(), "to", df['Date'].max())


# 4. Feature Engineering

print("\n🧱 Feature Engineering in Progress...")

# Moving average (7-day rolling window) of 'Units Sold'
df['units_sold_ma7'] = df.groupby('Product ID')['Units Sold'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

# Lag feature (previous day's Units Sold)
df['lag_1_units_sold'] = df.groupby('Product ID')['Units Sold'].shift(1).fillna(0)

print("✅ Added 'units_sold_ma7' and 'lag_1_units_sold' features.")


# 5. Exploratory Data Analysis (EDA)

print("\n📊 Performing Exploratory Data Analysis...")

# Sales trend of sample products
sample_products = df['Product ID'].unique()[:3]
for p in sample_products:
    temp = df[df['Product ID'] == p]
    plt.figure(figsize=(12, 5))
    plt.plot(temp['Date'], temp['Units Sold'], label="Daily Units Sold", alpha=0.7)
    plt.plot(temp['Date'], temp['units_sold_ma7'], label="7-Day MA of Units Sold", linewidth=2)
    plt.title(f"📈 Units Sold Trend - Product {p}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.show()

# Units Sold Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Units Sold'], bins=50, kde=True, color='purple')
plt.title("🔍 Units Sold Distribution", fontsize=14)
plt.xlabel("Units Sold")
plt.ylabel("Frequency")
plt.show()

# Monthly Units Sold Trend
df['month'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('month')['Units Sold'].sum().reset_index()

plt.figure(figsize=(12, 5))
plt.plot(monthly_sales['month'].astype(str), monthly_sales['Units Sold'], marker='o', linestyle='-')
plt.title("📅 Monthly Units Sold Trend", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Total Units Sold")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df[numeric_cols + ['units_sold_ma7', 'lag_1_units_sold']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("🌐 Correlation Heatmap of Numerical Features", fontsize=14)
plt.show()

# Stockout Events Count
stockouts = df['Stockout Indicator'].sum()
print(f"\n⚠️ Total Stockout Events: {stockouts} instances out of {df.shape[0]} records.")


# ➕ Additional Graph 1: Units Sold vs Price Scatter Plot

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='Units Sold', data=df, alpha=0.5)
plt.title("💡 Units Sold vs Price Scatter Plot", fontsize=14)
plt.xlabel("Price")
plt.ylabel("Units Sold")
plt.grid(True)
plt.show()


# ➕ Additional Graph 2: Product Rating vs Units Sold Boxplot

plt.figure(figsize=(10, 6))
sns.boxplot(x='Product Rating', y='Units Sold', data=df)
plt.title("📊 Product Rating vs Units Sold Boxplot", fontsize=14)
plt.xlabel("Product Rating")
plt.ylabel("Units Sold")
plt.grid(True)
plt.show()


# ➕ Additional Graph 3: Store-wise Total Units Sold Bar Plot

store_sales = df.groupby('Store ID')['Units Sold'].sum().reset_index().sort_values(by='Units Sold', ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x='Store ID', y='Units Sold', data=store_sales.head(20))  # Top 20 stores
plt.title("🏬 Top 20 Stores by Total Units Sold", fontsize=14)
plt.xlabel("Store ID")
plt.ylabel("Total Units Sold")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# ➕ Additional Graph 4: Units Sold during Holiday/Promotion vs Normal Days

plt.figure(figsize=(8, 6))
sns.boxplot(x='Holiday/Promotion', y='Units Sold', data=df)
plt.title("🎉 Units Sold: Holiday/Promotion vs Normal Days", fontsize=14)
plt.xlabel("Holiday/Promotion (0 = No, 1 = Yes)")
plt.ylabel("Units Sold")
plt.grid(True)
plt.show()


# ➕ Additional Graph 5: Average Units Sold by Month

df['month_num'] = df['Date'].dt.month
monthly_avg_sales = df.groupby('month_num')['Units Sold'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='month_num', y='Units Sold', data=monthly_avg_sales, palette='viridis')
plt.title("📅 Average Units Sold by Month", fontsize=14)
plt.xlabel("Month (1 = Jan, 12 = Dec)")
plt.ylabel("Average Units Sold")
plt.grid(True)
plt.show()


# ➕ Additional Graph 6: Average Units Sold by Weekday

df['weekday'] = df['Date'].dt.day_name()
weekday_avg_sales = df.groupby('weekday')['Units Sold'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
]).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='weekday', y='Units Sold', data=weekday_avg_sales, palette='magma')
plt.title("📊 Average Units Sold by Weekday", fontsize=14)
plt.xlabel("Weekday")
plt.ylabel("Average Units Sold")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# 6. Save Preprocessed Data

output_file = "cleaned_sales_data.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Preprocessed and feature-engineered data saved as '{output_file}'")
print("\n🚀 Data Preprocessing & EDA Complete! 🎉 Your dataset is now ready for modeling.")
