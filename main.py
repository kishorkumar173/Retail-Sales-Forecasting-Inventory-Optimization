from src.forecasting import train_model, plot_results
from src.inventory import calculate_inventory
import pandas as pd

# Train + Predict
model, results = train_model("data/processed/cleaned_data.csv")

plot_results(results)

# Prepare forecast for inventory (simple grouping)
forecast_series = results['Predicted']

# Convert to DataFrame format
forecast_df = pd.DataFrame({
    "Item_1": forecast_series[:30],
    "Item_2": forecast_series[30:60],
    "Item_3": forecast_series[60:90]
}).T

# Inventory Optimization
inventory = calculate_inventory(forecast_df)

print(inventory)