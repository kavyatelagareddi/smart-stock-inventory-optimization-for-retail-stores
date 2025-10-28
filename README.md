
# 🧮 SmartStock – AI-Powered Inventory Optimization for Retail Stores
AI-powered Inventory Forecasting Project
## 📌 Overview
SmartStock is an AI-powered inventory optimization system designed to help retailers forecast product demand, calculate optimal stock levels, and automate restocking decisions using machine learning and analytics.

This project was developed as part of a multi-phase academic milestone project.

---

## 🧱 Project Structure

archive (4)/
│
├── milestone1/
│ ├── cleaned_sales_data.csv
│ ├── data_set.csv
│ └── code_1.py
│
├── milestone2/
│ ├── data/
│ ├── forecasts/
│ ├── models/
│ ├── plots/
│ └── forecasting.py
│
├── milestone3/
│ └── inventory.py
│
├── milestone4/
│ └── dashboard.py
│
└── launch.json

---

## 🎯 Milestone Breakdown

### **Milestone 1 – Data Cleaning & Preparation**
- Cleaned and structured raw sales data.  
- Removed duplicates, handled missing values, and formatted timestamps.  
- Output: `cleaned_sales_data.csv`

### **Milestone 2 – Demand Forecasting**
- Implemented ARIMA, LSTM, and Prophet models to predict future demand.  
- Compared model accuracy and selected the best performing one.  
- Output: Forecast plots and model metrics.

### **Milestone 3 – Inventory Optimization**
- Applied EOQ (Economic Order Quantity), Safety Stock, and Reorder Point formulas.  
- Integrated forecasting results to determine optimal reorder levels.  
- Output: `inventory.py` with optimization logic.

### **Milestone 4 – Interactive Dashboard**
- Built an interactive Streamlit dashboard to visualize demand forecasts, KPIs, and reorder alerts.  
- Enabled decision support for procurement planning.  
- Output: `dashboard.py` (Streamlit app).

---

## 🧰 Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Plotly, Prophet, Streamlit, Scikit-learn  
- **Tools:** VS Code, Git, GitHub

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/kavyatelagareddi/smart-stock-inventory-optimization-for-retail-stores.git
   cd smart-stock-inventory-optimization-for-retail-stores
2.Install dependencies:

pip install -r requirements.txt


3.Run each milestone file:

# Forecasting
python milestone2/forecasting.py

# Inventory optimization
python milestone3/inventory.py

# Dashboard
streamlit run milestone4/dashboard.py

🧑‍💻 Author

Telagareddi Kavya
Department of Computer Science
