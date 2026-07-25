# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project detects fraudulent credit card transactions using Machine Learning classification algorithms.

The dataset is highly imbalanced, so SMOTE is used to balance the training data before model training.

---

## Features

- Data Cleaning
- Exploratory Data Analysis
- SMOTE for Class Imbalance
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Model Comparison
- Feature Importance
- Confusion Matrix
- ROC Curve
- Fraud Prediction
- Streamlit Web Application

---

## Dataset

Credit Card Fraud Detection Dataset (Kaggle)

Rows: 284,807

Features: 31

Target Column:

Class

0 = Normal Transaction

1 = Fraud Transaction

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Streamlit

---

## Project Structure

```
CREDIT_CARD_FRAUD_DETECTION/
│
├── data/
├── models/
├── outputs/
├── src/
├── app.py
├── requirements.txt
└── README.md
```

---

## Machine Learning Workflow

Data Cleaning

↓

EDA

↓

Train-Test Split

↓

Feature Scaling

↓

SMOTE

↓

Model Training

↓

Model Comparison

↓

Evaluation

↓

Prediction

↓

Streamlit Deployment

---

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

## Run Project

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python src/train.py
```

### Evaluate

```bash
python src/evaluate.py
```

### Predict

```bash
python src/predict.py
```

### Streamlit App

```bash
streamlit run app.py
```

---

## Author

Saurabh Mohite