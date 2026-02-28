# 💼 Employee Salary Prediction ML Web App

A full-stack Machine Learning web application that predicts employee salaries based on experience, education, job role, and other attributes. This project demonstrates end-to-end ML pipeline development, API deployment, and frontend integration.

---

# 📌 Project Overview

This project implements a complete machine learning workflow:

* Data preprocessing and feature engineering
* Model training and evaluation
* Model serialization using Pickle
* FastAPI backend for real-time predictions
* HTML and Streamlit frontend interfaces
* Cloud deployment using Render
* Version control using Git and GitHub

---

# 🧠 Machine Learning Pipeline

### Models evaluated:

* Linear Regression
* Random Forest Regressor ✅ (Best Model)
* Gradient Boosting Regressor

### Evaluation Metrics:

* R² Score
* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)

The best model is automatically selected and saved.

---

# 🏗️ Project Structure

```
salary-prediction-ml-app/
│
├── models/
│   └── salary_model.pkl        # Trained ML model
│
├── src/
│   ├── api.py                  # FastAPI backend
│   ├── train.py                # Training pipeline
│   ├── predict.py              # Local prediction script
│   ├── preprocess.py           # Preprocessing logic
│   ├── index.html              # Frontend interface
│   └── app.py                  # Streamlit frontend
│
├── Salary_Prediction/data/
│   └── Employee_Salary_Dataset.csv
│
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# ⚙️ Technologies Used

### Machine Learning

* Python-3.10.13
* Scikit-learn
* Pandas
* NumPy

### Backend

* FastAPI
* Uvicorn

### Frontend

* HTML, CSS, JavaScript
* Streamlit

### Deployment

* Render (Cloud Hosting)
* GitHub (Version Control)

---

# 📊 Features

✔ Train ML models automatically
✔ Select the best model based on performance
✔ Save trained model (.pkl)
✔ Real-time prediction API
✔ Web interface for user input
✔ Cloud deployment ready
✔ Scalable architecture

---

# 🧪 Example Input

```
{
  "ID": 101,
  "Age": 30,
  "Gender": "Male",
  "Education_Level": "Master",
  "Job_Title": "Data Scientist",
  "Experience_Years": 5,
  "Location": "Tier1"
}
```

### Example Output

```
Predicted Salary: ₹ 8,225,833.43
```

---

# ▶️ How to Run Locally

## 1. Clone repository

```
git clone https://github.com/yourusername/salary-prediction-ml-app.git
cd salary-prediction-ml-app
```

---

## 2. Install dependencies

```
pip install -r requirements.txt
```

---

## 3. Train model

```
cd src
python train.py
```

---

## 4. Start FastAPI server

```
uvicorn api:app --reload
```

Open browser:

```
http://127.0.0.1:8000/docs
```

---

## 5. Run frontend

Open:

```
src/index.html
```

---

# 🌐 Deployment

This project is deployed using Render cloud platform.

Deployment includes:

* FastAPI backend
* Pretrained ML model
* REST API endpoints

---

# 📡 API Endpoint

### POST /predict

Predict employee salary.

Input: JSON
Output: Predicted salary

---

# 📈 Future Improvements

* Add database integration
* Add authentication
* Improve UI design
* Add Docker support
* Deploy frontend separately
* Add CI/CD pipeline

---

# 👨‍💻 Author

Jeevan G
Aspiring Data Scientist & ML Engineer

GitHub:
https://github.com/jeevan9312

---
