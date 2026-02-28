# ============================================
# Employee Salary Prediction - Correct Prediction Script
# ============================================

import os
import pickle
import pandas as pd


print("Loading trained model...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "salary_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")


# ============================================
# IMPORTANT: USE EXACT SAME COLUMN NAMES AS DATASET
# ============================================

sample_data = {
    "ID": 101,
    "Age": 30,
    "Gender": "Male",
    "Education_Level": "Master",
    "Job_Title": "Data Scientist",
    "Experience_Years": 5,
    "Location": "Tier1"
}


input_df = pd.DataFrame([sample_data])

print("\nInput data:")
print(input_df)


# Predict
predicted_salary = model.predict(input_df)[0]

print("\nPredicted Salary:")
print(f"₹ {predicted_salary:,.2f}")