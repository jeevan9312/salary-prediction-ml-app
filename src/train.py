# ============================================
# Employee Salary Prediction - Training Pipeline
# Fully Corrected Production Version
# ============================================

import pandas as pd
import numpy as np
import os
import pickle

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================
# 0. Confirm Script Execution
# ============================================

print("Training pipeline started successfully")


# ============================================
# 1. Define Paths Safely (Production Safe)
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Salary_Prediction",
    "data",
    "Employee_Salary_Dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "salary_model.pkl")


# ============================================
# 2. Load Dataset
# ============================================

print("\nLoading dataset from:")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully")
print("Shape:", df.shape)

print("\nColumns in dataset:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================
# 3. Identify Target Column Automatically
# ============================================

# Try common salary column names
possible_targets = ["salary", "Salary", "annual_salary", "AnnualSalary"]

target_column = None

for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    raise Exception("Salary column not found. Check dataset column names.")

print(f"\nTarget column identified as: {target_column}")


# ============================================
# 4. Separate Features and Target
# ============================================

X = df.drop(target_column, axis=1)
y = df[target_column]


# ============================================
# 5. Identify Column Types
# ============================================

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)


# ============================================
# 6. Preprocessing Pipeline
# ============================================

numerical_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)


# ============================================
# 7. Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================
# 8. Define Models
# ============================================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}


# ============================================
# 9. Train and Evaluate Models
# ============================================

best_model = None
best_score = -np.inf
best_model_name = ""

results = []

print("\nStarting model training...")

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{name} Results:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    results.append((name, mae, rmse, r2))

    if r2 > best_score:
        best_score = r2
        best_model = pipeline
        best_model_name = name


# ============================================
# 10. Save Best Model
# ============================================

os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest model selected: {best_model_name}")
print(f"Model saved successfully at:")
print(MODEL_PATH)


# ============================================
# 11. Model Comparison Summary
# ============================================

print("\nFinal Model Comparison:")

for name, mae, rmse, r2 in results:
    print(f"{name}")
    print(f"  R2 Score : {r2:.4f}")
    print(f"  RMSE     : {rmse:.2f}")
    print(f"  MAE      : {mae:.2f}")
    print("----------------------------------")


print("\nTraining pipeline completed successfully")