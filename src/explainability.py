# src/explainability.py
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
from .config import MODELS_DIR


class FraudExplainer:
    def __init__(self, dataset_name="ecommerce_fraud"):
        path = MODELS_DIR / f"{dataset_name}_best_model.pkl"
        loaded = joblib.load(path)
        self.model = loaded['model']
        self.feature_names = loaded['feature_names']
        print(f"Loaded {dataset_name} model with {len(self.feature_names)} features")

    def baseline_importance(self, top_n=10):
        imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        plt.barh(imp['feature'][::-1], imp['importance'][::-1])
        plt.title(f'Top {top_n} Features - XGBoost Gain ({self.feature_names[0].split("_")[0]})')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

        return imp

    def shap_global(self, X_sample):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title("SHAP Global Importance")
        plt.tight_layout()
        plt.show()

        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("SHAP Beeswarm Plot")
        plt.tight_layout()
        plt.show()