# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and metadata
pollutants = ['O2', 'NO3', 'SO4', 'PO4', 'CL', 'NO2']
models = {p: joblib.load(f"model_{p.lower()}.pkl") for p in pollutants}
columns = joblib.load("best_columns.pkl")
scaler = joblib.load("best_scaler.pkl")

st.set_page_config(page_title="Water Quality Predictor")
st.title("💧 Water Quality Prediction App")

# User input
station_id = st.selectbox("Select Station ID", [str(i) for i in range(1, 60)])
year = st.number_input("Enter Year", min_value=2000, max_value=2025, step=1)
month = st.selectbox("Select Month", list(range(1, 13)))

if st.button("Predict Pollutants"):
    # Prepare input
    year_month = year * month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Generate seasonal dummy variables
    season = 'winter' if month in [12, 1, 2] else \
             'spring' if month in [3, 4, 5] else \
             'summer' if month in [6, 7, 8] else 'autumn'
    season_dummies = {'season_autumn': 0, 'season_spring': 0, 'season_summer': 0, 'season_winter': 0}
    season_dummies[f'season_{season}'] = 1

    input_df = pd.DataFrame([[station_id, year, month, year_month, month_sin, month_cos] + list(season_dummies.values())],
        columns=["id", "year", "month", "year_month", "month_sin", "month_cos"] + list(season_dummies.keys()))
    input_df = pd.get_dummies(input_df, columns=["id"])

    # Align with training columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict each pollutant
    predictions = []
    for p in pollutants:
        pred = models[p].predict(input_scaled)[0]
        if p in ['NO2', 'CL']:
            pred = np.expm1(pred)  # inverse of log1p
        predictions.append(pred)

    # Display results
    st.subheader("📊 Predicted Pollutant Levels (mg/L):")
    result_df = pd.DataFrame({"Pollutant": pollutants, "Predicted Value": predictions})
    st.dataframe(result_df)
