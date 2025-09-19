# app.py

from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import joblib
import xgboost as xgb

from utils.preprocessing import preprocess  # your preprocessing function

# ----------------------------------------------------
# Initialize FastAPI app
# ----------------------------------------------------
app = FastAPI(title="Sales Forecasting API",
              description="API for predicting weekly sales using RF or XGBoost",
              version="1.0")

# ----------------------------------------------------
# Load Models Once at Startup
# ----------------------------------------------------
rf_model = joblib.load("models/best_rf_model.pkl")

xgb_model = xgb.Booster()
xgb_model.load_model("models/best_sales_model.json")

# ----------------------------------------------------
# Define Input Schema (matches your notebook features)
# ----------------------------------------------------
class PredictionRequest(BaseModel):
    Store: int
    Dept: int
    Date: str
    IsHoliday: int
    Weekly_Sales: float
    Temperature: float
    Fuel_Price: float
    MarkDown1: float
    MarkDown2: float
    MarkDown3: float
    MarkDown4: float
    MarkDown5: float
    CPI: float
    Unemployment: float
    Size: float
    Type: str   # A, B, or C

# ----------------------------------------------------
# Prediction Endpoint
# ----------------------------------------------------
@app.post("/predict")
def predict(request: PredictionRequest,
            model: str = Query("rf", description="Choose 'rf' for RandomForest or 'xgb' for XGBoost")):
    """
    Predict weekly sales.
    - Request body = PredictionRequest schema
    - Query param `model` = rf or xgb
    """

    try:
        # Convert request to DataFrame
        df = pd.DataFrame([request.dict()])

        # Preprocess input
        processed = preprocess(df)

        # Choose model
        if model == "rf":
            result = rf_model.predict(processed)
        elif model == "xgb":
            dmatrix = xgb.DMatrix(processed)
            result = xgb_model.predict(dmatrix)
        else:
            return {"error": "Invalid model choice. Use 'rf' or 'xgb'."}

        return {"prediction": float(result[0]), "model_used": model}

    except Exception as e:
        return {"error": str(e)}
