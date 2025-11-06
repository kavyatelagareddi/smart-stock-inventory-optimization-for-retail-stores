import subprocess
import os
import sys
import time

print("\n🚀 SMART INVENTORY FULL PIPELINE STARTING...\n")

# Resolve absolute project path
BASE = os.path.dirname(os.path.abspath(__file__))

# --------------------------
# Helper to run Python files
# --------------------------
def run_script(script_path):
    print(f"\n🔧 Running: {script_path}")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"❌ Error running {script_path}")
        sys.exit(1)

# --------------------------
# 1️⃣ Milestone 1: Data Cleaning + EDA
# --------------------------
run_script(os.path.join(BASE, "milestone1", "code_1.py"))

# --------------------------
# 2️⃣ Milestone 2: Forecasting (Prophet + ARIMA + LSTM)
# --------------------------
run_script(os.path.join(BASE, "milestone2", "forecasting.py"))

# --------------------------
# 3️⃣ Milestone 3: Inventory Optimization (EOQ, ROP, SS, ABC)
# --------------------------
run_script(os.path.join(BASE, "milestone3", "inventory.py"))

# --------------------------
# 4️⃣ Milestone 4: Dashboard (streamlit)
# --------------------------
print("\n🌐 Launching Streamlit Dashboard...")
dashboard_path = os.path.join(BASE, "milestone4", "dashboard.py")

# Open streamlit on browser
subprocess.Popen(["streamlit", "run", dashboard_path])

time.sleep(2)
print("\n✅ FULL PIPELINE EXECUTED SUCCESSFULLY!")
print("📊 Dashboard Running at: http://localhost:8501\n")
