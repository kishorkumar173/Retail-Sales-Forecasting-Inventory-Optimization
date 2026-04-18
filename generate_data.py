import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range(start="2023-01-01", end="2023-12-31")

data = []

stores = [1, 2, 3]
items = ['A', 'B', 'C', 'D']

for store in stores:
    for item in items:
        base_demand = np.random.randint(20, 50)

        for date in dates:
            if date.weekday() >= 5:
                seasonality = 1.3
            else:
                seasonality = 1.0

            trend = 1 + (date.dayofyear / 365) * 0.2

            promo = np.random.choice([0,1], p=[0.8,0.2])
            promo_effect = 1.5 if promo == 1 else 1

            holiday = 1 if date.strftime('%m-%d') in ['01-01','12-25'] else 0
            holiday_effect = 2 if holiday else 1

            sales = base_demand * seasonality * trend * promo_effect * holiday_effect
            sales = int(np.random.normal(sales, 5))

            data.append([
                date, store, item, max(0, sales), promo, holiday
            ])

df = pd.DataFrame(data, columns=[
    "date", "store_id", "item_id", "sales", "promotion", "holiday"
])

df.to_csv("data/raw/retail_sales.csv", index=False)

print("Dataset created successfully!")