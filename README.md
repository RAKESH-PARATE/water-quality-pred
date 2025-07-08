💧 Water Quality Prediction - Project Code

This project predicts the concentration of six major water pollutants using hybrid machine learning techniques. It includes a complete data preprocessing pipeline, model training, performance analysis, and a Streamlit web app for real-time prediction.

---

📁 Project Structure

├── app.py                      → Streamlit app for user input and predictions  
├── analyze_model_outputs.py    → Evaluates models and generates performance charts  
├── water_quality_pred.py       → Main pipeline: data cleaning, feature engineering, model training  
├── best_scaler.pkl             → Saved StandardScaler used to transform input features  
├── best_columns.pkl            → List of input features used during training  
├── model_cl.pkl                → Trained model for Chloride (CL)  
├── model_no2.pkl               → Trained model for Nitrite (NO2)  
├── model_no3.pkl               → Trained model for Nitrate (NO3)  
├── model_o2.pkl                → Trained model for Dissolved Oxygen (O2)  
├── model_po4.pkl               → Trained model for Phosphate (PO4)  
├── model_so4.pkl               → Trained model for Sulfate (SO4)  
├── data/                       → Folder containing dataset (PB_All_2000_2021.csv)  
├── outputs/                    → Folder containing prediction and feature importance plots  
└── README.md                   → This file

---

📌 How to Run the Project

1. **Install Dependencies**  
   Ensure Python is installed, then run:

   ```
   pip install -r requirements.txt
   ```

2. **Train the Models**  
   If you want to regenerate models from scratch:

   ```
   python water_quality_pred.py
   ```

3. **Visualize Predictions and Importance**  
   Run this to generate `.png` output charts:

   ```
   python analyze_model_outputs.py
   ```

4. **Launch Streamlit App**  
   Use this command to open the web interface:

   ```
   streamlit run app.py
   ```

---

📊 Project Highlights

- Forecasts pollutant levels: O₂, NO₃, SO₄, PO₄, CL⁻, and NO₂  
- Built using HistGradientBoostingRegressor for each pollutant  
- Includes seasonal, lag-based, and station-clustered features  
- Scaler and column configuration stored in `.pkl` files  
- Streamlit app allows prediction up to the year **2100**

---

⚠️ Notes

- Trained on data from 2000 to 2021  
- Future predictions (beyond 2025) are extrapolations  
- Accuracy may vary outside the trained year range

---

👨‍💻 Developed by

**Rakesh Parate**  
🔗 LinkedIn: https://www.linkedin.com/in/rakesh-parate  
🔗 GitHub: https://github.com/RAKESH-PARATE

---

📄 License

This project is licensed under the MIT License.