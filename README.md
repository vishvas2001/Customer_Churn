<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/4149/4149678.png" width="120">
</p>

<h1 align="center">📊 Customer Churn Prediction</h1>

<p align="center">
  <b>End-to-End Machine Learning Project | EDA • ML Pipeline • Streamlit App</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Project-blueviolet">
  <img src="https://img.shields.io/badge/Python-3.10-blue">
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Model-yellow">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen">
</p>

---

## 🚀 Overview

This project predicts **customer churn for a telecom company** based on customer demographics, services subscribed, billing information, and account details.

Includes:

✔ Full **Exploratory Data Analysis (EDA)**  
✔ Business insights  
✔ Preprocessing pipeline  
✔ Logistic Regression ML model  
✔ 84% ROC-AUC  
✔ Saved model (`churn_model.pkl`)  
✔ Interactive **Streamlit Web App**  
✔ Production-level folder structure  
✔ Modular Python code in `src/`

---

## 🧠 Problem Statement

> Predict whether a customer will churn (leave the service) based on their account information and service usage.  
>  
> The goal is to help telecom companies **identify at-risk customers** and reduce churn through early intervention.

---

## 📂 Folder Structure

customer_churn/
│
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│ └── 01_eda_and_model.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── train_model.py
│ ├── predict.py
│ └── init.py
│
├── models/
│ └── churn_model.pkl
│
├── app.py
├── requirements.txt
└── README.md


---

## 🔍 Exploratory Data Analysis (EDA) — Highlights

### **📌 Key Findings**
- **Month-to-month** contract customers churn the most  
- Customers with **high monthly charges** have higher churn probability  
- **Short tenure** customers are the most likely to leave  
- **Electronic check** payment method is strongly associated with churn  
- **Fiber optic** customers show noticeably higher churn  

<details>
<summary>📊 Click to view sample EDA plots</summary>

#### Churn Distribution  
![Churn](/assets/churn_distribution.png)

#### Monthly Charges vs Churn  
![Monthly](/assets/monthly_churn-vs-churn.png)

#### Contract Type vs Churn  
![Contract](/assets/contract_churn.png)

</details>

---

## 🤖 Machine Learning Model

### **Algorithm Used:**  
`Logistic Regression`

### **Why Logistic Regression?**
- Outperformed RandomForest in:
  - F1-score  
  - Recall  
  - ROC-AUC  
- Easier to interpret  
- Works well with categorical-heavy data  

---

## 📈 Model Performance

| Metric | Score |
|--------|--------|
| **Accuracy** | 0.81 |
| **Recall (Churn)** | 0.56 |
| **F1-Score (Churn)** | 0.60 |
| **ROC-AUC** | **0.842** |

---

## 🧱 ML Pipeline

This project uses a full preprocessing + model pipeline with:

- OneHotEncoder for categorical features  
- Pass-through for numeric features  
- Logistic Regression classifier  
- Saved using `joblib`  

### Example Pipeline Code

```python
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]
)
```
---
### 🌐 Streamlit Web App

#### Run the app locally:
```
streamlit run app.py
```

#### The app:

- Collects customer details

- Predicts churn probability

- Shows color-coded risk

- Gives business recommendations

<details> <summary>📷 App Screenshot</summary>

</details>

---

### 🏃 Run the Project Locally (Full Steps)

#### 1. Clone the repo
```
git clone https://github.com/your-username/customer_churn.git
cd customer_churn
```
#### 2. Create virtual environment
```
python -m venv venv
venv\Scripts\activate  # Windows
```
#### 3. Install requirements
```
pip install -r requirements.txt
```
#### 4. Train the model (optional)
```
python src/train_model.py
```
#### 5. Run Streamlit app
```
streamlit run app.py
```
---
### 💡 Technologies Used

- Python

- Pandas

- NumPy

- Matplotlib / Seaborn

- scikit-learn

- Streamlit

- joblib

---

## 👤 Author

**Vishvas Parmar**
Final-year Computer Engineering Student
Aspiring Data Scientist & ML Enthusiast
Passionate about AI, ML, and real-world applications

---

<p align="center"> ⭐ If you like this project, consider giving it a star on GitHub! ⭐ </p> 
