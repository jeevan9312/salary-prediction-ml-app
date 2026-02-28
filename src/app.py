# ============================================
# Salary Prediction Frontend - Streamlit App
# ============================================

import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Employee Salary Prediction")
st.write("Enter employee details to predict salary")


# ============================================
# Input Fields
# ============================================

id_input = st.number_input("Employee ID", min_value=1, value=101)

age = st.slider("Age", 18, 65, 30)

gender = st.selectbox("Gender", ["Male", "Female"])

education = st.selectbox(
    "Education Level",
    ["Bachelor", "Master", "PhD"]
)

job_title = st.selectbox(
    "Job Title",
    ["Data Analyst", "Data Scientist", "ML Engineer", "Manager"]
)

experience = st.slider("Experience (Years)", 0, 40, 5)

location = st.selectbox(
    "Location",
    ["Tier1", "Tier2", "Tier3"]
)


# ============================================
# Prediction Button
# ============================================

if st.button("Predict Salary"):

    input_data = {
        "ID": id_input,
        "Age": age,
        "Gender": gender,
        "Education_Level": education,
        "Job_Title": job_title,
        "Experience_Years": experience,
        "Location": location
    }

    try:

        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:

            result = response.json()

            salary = result["predicted_salary"]

            st.success(f"Predicted Salary: ₹ {salary:,.2f}")

        else:
            st.error("API error. Check FastAPI server.")

    except:
        st.error("Cannot connect to API. Make sure FastAPI is running.")