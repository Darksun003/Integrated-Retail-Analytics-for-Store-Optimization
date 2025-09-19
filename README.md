# 🛍️ Integrated-Retail-Analytics-for-Store-Optimization
This project focuses on retail sales forecasting and store optimization using machine learning models and an API service for real-time predictions. It integrates multiple datasets — sales, store attributes, and additional features — to train models like Random Forest and XGBoost for predicting weekly sales across stores and departments.

The system is deployed as a FastAPI application that exposes prediction endpoints, enabling seamless integration with retail dashboards, decision-support tools, or operational workflows.

🎯 Objectives

Forecast weekly sales at store and department level.
Identify impact of holidays, markdowns, and external factors (e.g., fuel price, CPI, unemployment).
Optimize inventory management and promotions.
Provide real-time prediction services via API.
Enable data-driven store optimization strategies.

🛠️ Tech Stack
Languages: Python 3
Frameworks: FastAPI, XGBoost, Scikit-learn
Data Processing: Pandas, NumPy
Model Persistence: Joblib, JSON for XGBoost Booster
API Testing: Swagger UI / Postman

📂 Project Structure
├── app.py                          # FastAPI app for sales forecasting
├── models/                         # Trained ML models
│   ├── best_rf_model.pkl
│   └── best_sales_model.json
├── utils/                          # Utility scripts
│   └── preprocessing.py            # Data preprocessing pipeline
├── data/                           # Raw & processed datasets
│   ├── sales data-set.csv
│   ├── stores data-set.csv
│   ├── Features data set.csv
│   └── final_merged_dataset.csv
├── notebooks/                      # Jupyter notebooks for experiments
│   └── Sample_ML_Submission_Template.ipynb
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file

📊 Datasets Used
Sales Data – Historical weekly sales at store & department level.
Store Data – Store size, type (A, B, C).
Features Data – Temperature, Fuel Price, MarkDowns, CPI, Unemployment.
Final Merged Dataset – Combined dataset used for training.

🤖 Machine Learning Models

Random Forest (RF) – Used for robust predictions with tabular data.
XGBoost (XGB) – Gradient boosting model optimized for accuracy.
Both models were trained on the merged dataset with engineered features.

📈 Results

Models achieved strong predictive performance on weekly sales data.
Feature importance analysis highlighted holiday events, markdowns, and fuel prices as key drivers of sales variation.
XGBoost outperformed RF in accuracy but RF offered more interpretability.

🔮 Future Enhancements
Deploy as a cloud-based service (AWS/GCP/Azure).
Add real-time streaming data ingestion (e.g., Kafka).
Extend predictions to multi-channel retail (online + offline).
Integrate recommendation engine for promotions.

👨‍💻 Author -- GV Jayanth
Developed as part of the Integrated Retail Analytics for Store Optimization project to demonstrate the power of ML-driven decision-making in retail
