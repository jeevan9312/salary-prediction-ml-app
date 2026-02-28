# src/api.py

import os
import pickle
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "salary_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.get("/")
def home():
    return {"message": "Salary Prediction API is running"}


@app.post("/predict")
def predict_salary(data: dict):

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    return {
        "predicted_salary": float(prediction)
    }