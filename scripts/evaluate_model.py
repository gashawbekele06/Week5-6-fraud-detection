# scripts/evaluate_model.py
import pandas as pd
from src.explainability import FraudExplainer
from src.config import DATA_PROCESSED

if __name__ == "__main__":
    X_test = pd.read_parquet(DATA_PROCESSED / "X_test_ecommerce.parquet")

    explainer = FraudExplainer(dataset_name="ecommerce_fraud")
    explainer.baseline_importance()
    explainer.shap_global(X_test.sample(1000, random_state=42))