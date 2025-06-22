# water_quality_pred.py

# 📌 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 📌 2. Load the Dataset (semicolon-separated)
file_path = os.path.join("data", "PB_All_2000_2021.csv")  # Make sure this matches your actual filename
df = pd.read_csv(file_path, sep=';')
print("✅ Dataset Loaded. Shape:", df.shape)

# 📌 3. Display Basic Info
print("\n📌 First 5 rows:\n", df.head())
print("\n📌 Data Info:\n")
print(df.info())
print("\n📌 Missing values:\n", df.isnull().sum())

# 📌 4. Drop rows with missing values
df = df.dropna()
print("✅ After dropping missing rows. New shape:", df.shape)

# 📌 5. Define Features and Targets
target_cols = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
X = df.drop(columns=target_cols + ['id', 'date'])  # drop id and date if present
y = df[target_cols]

# 📌 6. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
print("📊 Saved: correlation_heatmap.png")

# 📌 7. Histograms of Target Variables
df[target_cols].hist(figsize=(12, 8), bins=30)
plt.suptitle("Target Parameter Distributions")
plt.tight_layout()
plt.savefig("target_distributions.png")
print("📊 Saved: target_distributions.png")

# 📌 8. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("📊 Train and test data split complete.")

# 📌 9. Train the Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(rf)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# 📌 10. Predictions
y_pred = model.predict(X_test)

# 📌 11. Evaluation Metrics
print("\n📈 Model Evaluation:")
print("R² Score (overall):", r2_score(y_test, y_pred))
print("MSE (overall):", mean_squared_error(y_test, y_pred))

print("\n📈 R² Score per Parameter:")
for i, col in enumerate(target_cols):
    print(f"{col}: {r2_score(y_test[col], y_pred[:, i]):.4f}")

# 📌 12. Feature Importance (First Target Model)
importances = model.estimators_[0].feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("Feature Importance (First Estimator)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("📊 Saved: feature_importance.png")

# 📌 13. Show One Sample Prediction
example_df = pd.DataFrame({
    "Parameter": target_cols,
    "Actual": y_test.iloc[0].values,
    "Predicted": y_pred[0]
})
print("\n📋 Sample Prediction vs Actual:\n", example_df)

print("\n🎉 Script executed successfully!")
