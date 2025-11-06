📦 Smart Stock Inventory Optimization for Retail Stores

Forecasting + Inventory Optimization + Streamlit Dashboard

🚀 Overview

Smart Stock Inventory Optimization is an AI-powered inventory management system that predicts product demand and optimizes inventory using:

✅ Time series forecasting (Prophet, ARIMA, LSTM)
✅ EOQ (Economic Order Quantity)
✅ ROP (Reorder Point)
✅ Safety Stock calculation
✅ Streamlit dashboard for live inventory analytics

The project automates the entire workflow from data cleaning → forecasting → optimization → dashboard visualization, and can be executed using:

python run_all.py

📁 Project Structure
📦 smart-stock-inventory-optimization-for-retail-stores
│── milestone1/
│   └── code_1.py (Data cleaning + EDA)
│── milestone2/
│   ├── forecasting.py (Prophet + ARIMA + LSTM)
│   ├── data/forecast_results.csv
│   ├── forecasts/
│   ├── plots/
│   └── models/
│── milestone3/
│   └── inventory.py (EOQ + Reorder Point + Safety Stock + ABC Analysis)
│── milestone4/
│   └── dashboard.py (Streamlit Visualization)
│── run_all.py  ✅ Automates all milestones
│── README.md
│── requirements.txt

⚙️ Tech Stack
Component	Technology Used
Programming	Python
Forecasting Models	Prophet, ARIMA, LSTM
Dashboard	Streamlit
Visualization	Matplotlib, Plotly
Data Handling	Pandas, NumPy
🚀 Features
Milestone	Output
🧹 Milestone 1 – Data Cleaning & EDA	preprocesses data, removes duplicates, handles missing values, extracts date features
📈 Milestone 2 – Forecasting Models	Prophet, ARIMA, LSTM forecasting + Model comparison + Error metrics
📦 Milestone 3 – Inventory Optimization	EOQ, Reorder Point, Safety Stock, ABC Classification
📊 Milestone 4 – Dashboard	Streamlit dashboard with multi-tab insights, charts & alerts
🖼️ Dashboard Preview
📈 Forecast Analysis
📊 Inventory Optimization
🚨 Alerts & Notifications
📋 Export Reports
📊 Insights & KPIs


✅ Shows when stock hits reorder level, and generates alerts.

▶️ How to Run
✅ 1. Install Dependencies
pip install -r requirements.txt

✅ 2. Run Complete Pipeline (ALL milestones automatically)
python run_all.py

✅ 3. To run dashboard manually
streamlit run milestone4/dashboard.py

🧠 Key Concepts Used

Time-series forecasting (30-day future prediction)

Compare MAE & RMSE of Prophet, ARIMA, LSTM → choose best model

Inventory strategies:

EOQ = Optimal order quantity

ROP = When to reorder

Safety Stock = Buffer against uncertainty

📊 Model Comparison Example
Model	MAE (↓ better)	RMSE (↓ better)
Prophet ✅	0.000017	0.000020
ARIMA	50.33	63.92
LSTM	29.79	37.65

Prophet performs best → automatically selected for forecasting.

📥 Output Files Generated
Folder	Contains
forecasts/	30-day forecast CSVs for each product
plots/	Forecast graph images
models/	Trained model files (Prophet, ARIMA, LSTM)
milestone2/data/forecast_results.csv	Used by dashboard
👩‍💻 Author

Kavya Telagareddi

⭐ If you found this helpful, consider giving the repo a star!
