# Fraud Detection Project

**Project Name**: Fraud Detection for E-commerce and Bank Transactions  
**Company**: Adey Innovations Inc.  
**Date**: December 20, 2025

## Project Goals & Objectives

The goal of this project is to build accurate, interpretable machine learning models to detect fraudulent transactions in two datasets:

- **E-commerce transactions** (`Fraud_Data.csv` + `IpAddress_to_Country.csv`)
- **Credit card transactions** (`creditcard.csv`)

### Key Objectives

- Handle severe class imbalance (fraud is rare)
- Achieve high PR-AUC and F1-score while minimizing false positives (customer friction)
- Provide interpretable insights using SHAP
- Deliver actionable business recommendations to reduce fraud effectively

## Project Tasks

1. **Task 1**: Data Analysis & Preprocessing

   - Cleaning, EDA, geolocation mapping, feature engineering, scaling/encoding, SMOTE imbalance handling

2. **Task 2**: Model Building & Training

   - Stratified split, Logistic Regression baseline, XGBoost ensemble, cross-validation, model comparison

3. **Task 3**: Model Explainability
   - Built-in feature importance, SHAP summary & force plots, top fraud drivers, business recommendations

## Folder Structure

fraud-detection/
├── data/
│ ├── raw/ # Original CSV files (not tracked in git)
│ │ ├── Fraud_Data.csv
│ │ ├── IpAddress_to_Country.csv
│ │ └── creditcard.csv
│ └── processed/ # Processed pickle files from Task 1
├── src/
│ ├── config.py # Centralized paths, constants, and helpers
│ └── data_preprocessor.py # Full preprocessing pipeline (Task 1)
├── notebooks/
│ ├── eda-fraud-data.ipynb
│ ├── eda-creditcard.ipynb
│ ├── feature-engineering.ipynb
│ ├── modeling.ipynb
│ └── shap-explainability.ipynb
├── models/ # Saved XGBoost models (.pkl)
│ ├── xgb_ecommerce_best.pkl
│ └── xgb_creditcard_best.pkl
├── scripts/
│ └── run_task1.py # Optional: run preprocessing pipeline
├── requirements.txt
└── README.md # This file

## Setup Instructions

### 1. Prerequisites

- Python 3.8+ (tested with 3.11/3.12)
- Jupyter Notebook or JupyterLab

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
shap
joblib
