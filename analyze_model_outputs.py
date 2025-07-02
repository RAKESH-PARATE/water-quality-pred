# analyze_model_outputs.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Configuration
pollutants = ['O2', 'NO3', 'SO4', 'PO4', 'CL', 'NO2']
log_targets = ['NO2', 'CL']
data_file = "data/PB_All_2000_2021.csv"
scaler_file = "best_scaler.pkl"
columns_file = "best_columns.pkl"

# Load raw data
raw = pd.read_csv(data_file, sep=';')
raw['date'] = pd.to_datetime(raw['date'], dayfirst=True)
raw['year'] = raw['date'].dt.year
raw['month'] = raw['date'].dt.month
raw['year_month'] = raw['year'] * raw['month']
raw['month_sin'] = np.sin(2 * np.pi * raw['month'] / 12)
raw['month_cos'] = np.cos(2 * np.pi * raw['month'] / 12)

def get_season(month):
    if month in [12, 1, 2]: return 'winter'
    elif month in [3, 4, 5]: return 'spring'
    elif month in [6, 7, 8]: return 'summer'
    else: return 'autumn'

raw['season'] = raw['month'].apply(get_season)
raw = pd.get_dummies(raw, columns=['season'])

# Align to model features
scaler = joblib.load(scaler_file)
columns = joblib.load(columns_file)

raw = pd.get_dummies(raw, columns=['id'])
for col in columns:
    if col not in raw.columns:
        raw[col] = 0
X = raw[columns]
X_scaled = scaler.transform(X)

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Analyze each model
for p in pollutants:
    model_file = f"model_{p.lower()}.pkl"
    if not os.path.exists(model_file):
        continue

    model = joblib.load(model_file)
    y_true = raw[p].values
    y_pred = model.predict(X_scaled)

    # Skip if NaNs exist
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if p in log_targets:
        y_pred_clean = np.expm1(y_pred_clean)

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_clean, y_pred_clean, alpha=0.3, edgecolor='k')
    plt.xlabel(f"Actual {p} (mg/L)")
    plt.ylabel(f"Predicted {p} (mg/L)")
    plt.title(f"{p} - Predicted vs Actual\nR²: {r2_score(y_true_clean, y_pred_clean):.2f} | MSE: {mean_squared_error(y_true_clean, y_pred_clean):.2f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/pred_vs_actual_{p}.png")
    plt.close()

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-15:][::-1]
        plt.figure(figsize=(8, 5))
        plt.barh([columns[i] for i in top_idx], importances[top_idx], color='steelblue')
        plt.gca().invert_yaxis()
        plt.title(f"{p} - Top Feature Importances")
        plt.tight_layout()
        plt.savefig(f"outputs/feature_importance_{p}.png")
        plt.close()

print("✅ Charts saved in /outputs folder")
