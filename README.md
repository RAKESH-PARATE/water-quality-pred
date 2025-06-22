# 💧 Water Quality Prediction — Improvised Version

This project is an **enhanced version** of a machine learning pipeline originally developed as part of an AICTE Virtual Internship sponsored by Shell (June 2025). The objective is to predict multiple water quality parameters using regression models.

---

## 🎯 Task Objective

I was provided with:
- A partially working `.ipynb` notebook using `MultiOutputRegressor + RandomForestRegressor`
- A semicolon-separated water quality dataset (2000–2021)
- A basic README and evaluation goal

My role was **not to complete it from scratch**, but to **improvise and improve the existing approach** while keeping its structure intact.

---

## 🔧 Improvements Made

| Area | Mentor’s Version | My Improvisation |
|------|------------------|------------------|
| ✅ Data Reading | Used incorrect delimiter | Fixed using `sep=';'` for correct CSV parsing |
| ✅ Missing Data | Not handled | Removed all rows with missing values using `dropna()` |
| ✅ Feature Engineering | No extra features | Extracted `month` and `year` from the `date` column |
| ✅ Model Inputs | Possibly full or unclear inputs | Used only valid inputs (`id`, `month`, `year`) to avoid data leakage |
| ✅ Visualization | None | Added:  
  - Correlation heatmap  
  - Histogram of target variables  
  - Feature importance chart |
| ✅ Evaluation | Only basic R²/MSE | Added:  
  - Per-parameter R²  
  - Actual vs predicted sample display |
| ✅ Presentation | Limited | Added saved plots & better printed insights |

---

## 📊 Model Overview

- **Model Used**: `MultiOutputRegressor` with `RandomForestRegressor`
- **Targets Predicted**:  
  - NH4, BSK5, Suspended solids, O2, NO3, NO2, SO4, PO4, CL

---

## 📈 Final Output Metrics (R² Score per Parameter)

| Parameter   | R² Score |
|-------------|----------|
| NH4         | 0.8160   |
| CL          | 0.8861   |
| SO4         | 0.7450   |
| NO3         | 0.6636   |
| PO4         | 0.5993   |
| O2          | 0.4605   |
| BSK5        | 0.4419   |
| Suspended   | -0.7248  |
| NO2         | -1.9403  |

> ⚠️ Some scores are low or negative due to limited features (e.g., no sensor values or location data), which is expected.

---

## 🖼️ Visualizations Generated

- `correlation_heatmap.png`  
- `target_distributions.png`  
- `feature_importance.png`

---

## 📁 Folder Structure

