import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import pandas as pd


st.set_page_config(page_title="Retail Forecasting", layout="wide")

st.title("📊 Retail Sales Forecasting & Inventory Optimization")

st.sidebar.header("Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build correct path
data_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# Load data
data = pd.read_csv(data_path)
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, parse_dates=['date'])
else:
    data = pd.read_csv(data_path)
# Sidebar
st.sidebar.header("Filters")

store = st.sidebar.selectbox("Select Store", data['store_id'].unique())
item = st.sidebar.selectbox("Select Item", data['item_id'].unique())

filtered_data = data[(data['store_id']==store) & (data['item_id']==item)]

st.subheader("📈 Sales Trend")

fig, ax = plt.subplots()
ax.plot(filtered_data['date'], filtered_data['sales'])
ax.set_title("Sales Over Time")
st.pyplot(fig)

# Load model
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
model = joblib.load(model_path)

# Prepare latest data for prediction
latest = filtered_data.tail(1)

features = ['lag_1','lag_7','rolling_mean_7','day_of_week','month']
X = latest[features]

prediction = model.predict(X)[0]

st.subheader("🔮 Forecast")

st.success(f"Predicted Sales: {round(prediction,2)}")

# Inventory logic
lead_time = 7
safety_stock = prediction * 0.2
reorder_point = prediction * lead_time + safety_stock
current_stock = 100
order_qty = max(0, reorder_point - current_stock)

st.subheader("📦 Inventory Recommendation")

st.write(f"Safety Stock: {round(safety_stock,2)}")
st.write(f"Reorder Point: {round(reorder_point,2)}")
st.write(f"Recommended Order Quantity: {round(order_qty,2)}")
st.subheader("📄 Data Preview")
st.dataframe(data.head())