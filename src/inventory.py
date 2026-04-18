import pandas as pd
import numpy as np
from scipy.stats import norm

def calculate_inventory(forecast_df):

    # Assume some business values (real-world assumptions)
    lead_time = 7   # days
    service_level = 0.95
    z = norm.ppf(service_level)

    results = []

    for item in forecast_df.index:

        forecast = forecast_df.loc[item]

        # Mean demand during lead time
        mu_L = np.mean(forecast) * lead_time

        # Std deviation (uncertainty)
        sigma = np.std(forecast)
        sigma_L = sigma * np.sqrt(lead_time)

        # Safety Stock
        safety_stock = z * sigma_L

        # Reorder Point
        reorder_point = mu_L + safety_stock

        # Assume current stock
        current_stock = np.random.randint(50, 150)

        # Order Quantity
        order_qty = max(0, reorder_point - current_stock)

        results.append([
            item,
            round(mu_L,2),
            round(safety_stock,2),
            round(reorder_point,2),
            current_stock,
            round(order_qty,2)
        ])

    inventory_df = pd.DataFrame(results, columns=[
        "Item",
        "Demand_LeadTime",
        "Safety_Stock",
        "Reorder_Point",
        "Current_Stock",
        "Order_Quantity"
    ])

    inventory_df.to_csv("outputs/inventory_recommendations.csv", index=False)

    return inventory_df