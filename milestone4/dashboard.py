import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime
import time

# ------------------- Page Setup -------------------
st.set_page_config(page_title="📦 Smart Inventory Dashboard", layout="wide", page_icon="📊")

# ------------------- Forecast Data Path -------------------
DATA_PATH = "../milestone2/data/forecast_results.csv"

if not os.path.exists(DATA_PATH):
    st.error("⚠️ Run forecasting.py first to generate forecast_results.csv")
    st.stop()

# ------------------- Auto-refresh based on file change -------------------
if "last_modified" not in st.session_state:
    st.session_state.last_modified = os.path.getmtime(DATA_PATH)

current_modified = os.path.getmtime(DATA_PATH)
if current_modified != st.session_state.last_modified:
    st.session_state.last_modified = current_modified
    st.experimental_rerun()  # Automatically refresh dashboard

# ------------------- Load Data -------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
if "ds" in df.columns:
    df.rename(columns={"ds":"date"}, inplace=True)
if "yhat" in df.columns:
    df.rename(columns={"yhat":"forecast_best"}, inplace=True)

required_cols = ["Product ID","date","forecast_best"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Missing columns in data: {missing}")
    st.stop()

df["date"] = pd.to_datetime(df["date"])

# ------------------- Sidebar Controls -------------------
st.sidebar.header("🔧 Configuration Panel")
lead = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
oc = st.sidebar.slider("Ordering Cost ($)", 10, 300, 50)
hc = st.sidebar.slider("Holding Cost ($/unit)", 1, 25, 2)
service_level = st.sidebar.selectbox("Service Level", ["90%","95%","99%"], index=1)
z = {"90%":1.28,"95%":1.65,"99%":2.33}[service_level]

# Show last update timestamp
st.sidebar.markdown(f"**Last Forecast Update:** {datetime.fromtimestamp(current_modified).strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------- Tabs -------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast Analysis",
    "📊 Inventory Optimization",
    "🚨 Stock Alerts",
    "📋 Reports & Export",
    "💡 Insights & KPIs"
])

# ------------------- Tab 1: Forecast -------------------
with tab1:
    st.subheader("📈 Product Demand Forecasts")
    prod = st.selectbox("Select Product", df["Product ID"].unique())
    subset = df[df["Product ID"]==prod]
    fig = px.line(subset, x="date", y="forecast_best", title=f"Forecast Trend for {prod}", markers=True, color_discrete_sequence=["#007BFF"])
    fig.update_layout(xaxis_title="Date", yaxis_title="Forecasted Demand")
    st.plotly_chart(fig, use_container_width=True)

# ------------------- Tab 2: Inventory -------------------
with tab2:
    st.subheader("📊 EOQ & Reorder Planning")
    plan=[]
    for p in df["Product ID"].unique():
        sub = df[df["Product ID"]==p]
        avg=sub["forecast_best"].mean()/30
        dem=sub["forecast_best"].sum()
        std=sub["forecast_best"].std()
        eoq = np.sqrt((2*dem*oc)/hc)
        ss = z*std*np.sqrt(lead)
        rop = (avg*lead)+ss
        plan.append({"Product":p,"Avg Daily Demand":round(avg,2),"Total Demand":round(dem,2),
                     "EOQ":round(eoq,2),"Safety Stock":round(ss,2),"Reorder Point":round(rop,2)})
    inv_df=pd.DataFrame(plan)
    st.dataframe(inv_df, use_container_width=True)

# ------------------- Tab 3: Stock Alerts -------------------
with tab3:
    st.subheader("🚨 Live Stock Monitoring")
    inv_df["Current Stock"]=np.random.randint(10,120,len(inv_df))
    inv_df["Status"]=np.where(inv_df["Current Stock"]<inv_df["Reorder Point"],"Reorder 🚨","OK ✅")
    col1,col2=st.columns(2)
    with col1:
        st.dataframe(inv_df[["Product","Current Stock","Reorder Point","Status"]])
    with col2:
        fig2=px.bar(inv_df, x="Product", y=["Current Stock","Reorder Point"], barmode="group", title="Stock vs Reorder Point")
        st.plotly_chart(fig2,use_container_width=True)

# ------------------- Tab 4: Reports -------------------
with tab4:
    st.subheader("📋 Export & Reporting Tools")
    csv=inv_df.to_csv(index=False)
    st.download_button("📥 Download Inventory Report (CSV)", csv, "SmartStock_Report.csv")

# ------------------- Tab 5: KPIs -------------------
with tab5:
    st.subheader("💡 Business Insights")
    total_products=len(inv_df)
    reorder_count=(inv_df["Status"]=="Reorder 🚨").sum()
    avg_eoq=round(inv_df["EOQ"].mean(),2)
    avg_ss=round(inv_df["Safety Stock"].mean(),2)
    kpi1,kpi2,kpi3,kpi4=st.columns(4)
    kpi1.metric("📦 Total Products", total_products)
    kpi2.metric("🚨 Reorder Alerts", reorder_count)
    kpi3.metric("⚙️ Avg EOQ", avg_eoq)
    kpi4.metric("🧠 Avg Safety Stock", avg_ss)

