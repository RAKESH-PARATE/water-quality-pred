💧 Water Quality Prediction using Hybrid Machine Learning

Hi there! 👋
Welcome to my water quality prediction project. I built this out of curiosity — to explore how machine learning can help forecast environmental factors, especially something as vital as water quality.

Using over 20 years of real-world data, this project predicts the concentration of pollutants at various monitoring stations using a hybrid ML pipeline and an interactive Streamlit web app.

🔍 What’s This Project About?

Water pollution often gets noticed after it’s too late. What if we could predict it ahead of time?

This project brings together:

Historical water quality data from 2000 to 2021

Smart pollutant-specific ML models

A simple Streamlit app for real-time prediction

It predicts levels of six key pollutants: O₂, NO₃, SO₄, PO₄, Cl⁻, and NO₂.

🎯 Goals

Clean and preprocess long-term water quality data

Engineer seasonal, temporal, and spatial features

Build one machine learning model per pollutant

Visualize and interpret predictions

Deploy a user-friendly prediction interface

⚙️ Tech & Tools Used

Purpose	Tools / Libraries
Data Processing	Python, Pandas, NumPy
Machine Learning	Scikit-learn, HistGradientBoostingRegressor
Feature Engineering	Lag features, Rolling averages, KMeans clustering
Visualization	Matplotlib
App Development	Streamlit
Model Management	Joblib

🧪 How It Works

Data Preprocessing

Loaded water quality data from Punjab (2000–2021)

Removed missing values and outliers (z-score method)

Extracted date-based features: month, season, month_sin, etc.

Hybrid Feature Engineering

Lag and rolling features for NO₃, PO₄, Cl⁻

KMeans clustering based on pollutant means per station

One-hot encoding of seasons, stations, and clusters

Model Training

Trained a separate HistGradientBoostingRegressor for each pollutant

Applied log transformation for skewed data (NO₂ and Cl⁻)

Evaluated using R² and MSE

Streamlit App

Input: Year, Month, Station ID

Output: Predicted pollutant concentrations

Models are pre-trained and loaded dynamically

📊 Model Performance

Achieved R² values up to ~0.90 for some pollutants

Predicted vs Actual plots validate the performance

Feature importance charts provide interpretability

🖼️ Sample Output


This chart shows the model's accuracy in predicting Chloride (Cl⁻) levels.

🌐 Try It Out

GitHub Repository:
https://github.com/RAKESH-PARATE/water-quality-pred

You can clone the repo and run the app locally using:

arduino
Copy
Edit
streamlit run app.py
Make sure to install all dependencies listed in requirements.txt.

🔮 What’s Next

Deploy the app using Streamlit Cloud

Connect live sensor data for real-time forecasting

Expand to datasets from other states or countries

Add support for alerts or pollution threshold warnings

👋 About Me

I'm Rakesh Parate, a Computer Science Engineering student who loves combining data, logic, and creativity to solve real-world problems.

Let’s connect:

LinkedIn: https://www.linkedin.com/in/rakesh-parate

GitHub: https://github.com/RAKESH-PARATE

📄 License

This project is released under the MIT License. You’re welcome to use, modify, and build upon it.
