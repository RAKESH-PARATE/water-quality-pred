# water_quality_pred.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import zscore

# Load data
df = pd.read_csv("data/PB_All_2000_2021.csv", sep=';')
print("✅ Loaded columns:", df.columns.tolist())

# Rename if needed
if 'Date' in df.columns:
    df.rename(columns={'Date': 'date'}, inplace=True)
if 'station_id' in df.columns:
    df.rename(columns={'station_id': 'id'}, inplace=True)

assert 'date' in df.columns, "Missing 'date' column"
assert 'id' in df.columns, "Missing 'id' column"

df.dropna(inplace=True)

# Date features
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['year_month'] = df['year'] * df['month']
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Season
def get_season(month):
    if month in [12, 1, 2]: return 'winter'
    elif month in [3, 4, 5]: return 'spring'
    elif month in [6, 7, 8]: return 'summer'
    else: return 'autumn'

df['season'] = df['month'].apply(get_season)
df = pd.get_dummies(df, columns=['season'])

# Remove outliers globally
pollutants = ['O2', 'NO3', 'SO4', 'PO4', 'CL', 'NO2']
df = df[(np.abs(zscore(df[pollutants])) < 3).all(axis=1)]

# Sort for lag
df.sort_values(by=['id', 'date'], inplace=True)

# Save original ID for lag
df['id_for_lag'] = df['id']

# Create lag and rolling features
use_lags = ['NO3', 'PO4', 'CL']
for p in use_lags:
    df[f'{p}_lag1'] = df.groupby('id_for_lag')[p].shift(1)
    df[f'{p}_roll3'] = df.groupby('id_for_lag')[p].rolling(3).mean().shift(1).reset_index(level=0, drop=True)

# Drop rows with missing lag/rolling values
df.dropna(inplace=True)

# Station clustering
station_means = df.groupby('id')[pollutants].mean()
kmeans = KMeans(n_clusters=5, random_state=42).fit(station_means)
df['station_cluster'] = df['id'].map(dict(zip(station_means.index, kmeans.labels_)))

# One-hot encode id and cluster
df = pd.get_dummies(df, columns=['id', 'station_cluster'])

# Feature columns
feature_cols = ['year', 'month', 'year_month', 'month_sin', 'month_cos'] + \
               [f'season_{s}' for s in ['autumn', 'spring', 'summer', 'winter']]
lag_features = [f'{p}_lag1' for p in use_lags] + [f'{p}_roll3' for p in use_lags]
dummy_features = [col for col in df.columns if col.startswith('id_') or col.startswith('station_cluster_')]
all_features = feature_cols + lag_features + dummy_features

# Final feature matrix
X = df[all_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'best_scaler.pkl')
joblib.dump(X.columns.tolist(), 'best_columns.pkl')

# Train one model per pollutant
for p in pollutants:
    y = df[[p]].copy()
    log_transform = p in ['NO2', 'CL']
    if log_transform:
        y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)
    y_pred = np.expm1(y_pred) if log_transform else y_pred
    y_true = np.expm1(y_test) if log_transform else y_test

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"\n📊 {p} Model - R²: {r2:.4f}, MSE: {mse:.4f}")

    joblib.dump(model, f"model_{p.lower()}.pkl")

print("\n✅ Hybrid models trained and saved successfully.")
