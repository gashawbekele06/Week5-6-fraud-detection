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

### Prerequisites

- Python 3.9+
- Git
- Virtual environment (venv or conda)

### Installation Steps

**Clone the repository**

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

**Create and activate virtual environment**

```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate          # Linux/Mac
   # OR Windows
   venv\Scripts\activate

   # Using conda (alternative)
   conda create -n fraud-detection python=3.11
   conda activate fraud-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Prepare data**

```bash
 Place the three CSV files into data/raw/:
   Fraud_Data.csv
   creditcard.csv
   IpAddress_to_Country.csv
```

**Requirements**

```bash
 # requirements.txt
   pandas>=2.0.0
   numpy>=1.24.0
   joblib==1.5.2
   scikit-learn>=1.3.0
   xgboost>=2.0.0
   imbalanced-learn>=0.11.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   jupyter>=1.0.0
   notebook>=7.0.0
   shap>=0.45.0
   tqdm>=4.66.0
   pyarrow>=14.0.0
```
