import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

def train_model(data_path):

    df = pd.read_csv(data_path, parse_dates=['date'])

    # Features & Target
    features = ['lag_1', 'lag_7', 'rolling_mean_7', 'day_of_week', 'month']
    target = 'sales'

    X = df[features]
    y = df[target]

    # Train-Test Split (time-based)
    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    # Save model
    joblib.dump(model, "models/model.pkl")

    # Save predictions
    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions
    })

    results.to_csv("outputs/forecasts.csv", index=False)

    return model, results
import matplotlib.pyplot as plt

def plot_results(results):

    plt.figure(figsize=(12,5))
    plt.plot(results['Actual'].values, label="Actual")
    plt.plot(results['Predicted'].values, label="Predicted")

    plt.legend()
    plt.title("Actual vs Predicted Sales")

    plt.savefig("outputs/plots/forecast_plot.png")