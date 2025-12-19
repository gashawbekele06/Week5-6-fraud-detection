# src/modeling.py
import joblib
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import xgboost as xgb
from .import import MODELS_DIR


class FraudModelTrainer:
    def __init__(self, dataset_name="ecommerce_fraud"):
        self.dataset_name = dataset_name
        self.best_model = None

    def _pr_auc(self, y_true, y_prob):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        return auc(recall, precision)

    def train_baseline(self, X_train, y_train, X_test, y_test):
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga', n_jobs=-1)
        lr.fit(X_train, y_train)
        self._evaluate(lr, X_test, y_test, "Logistic Regression")
        return lr

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        xgb_base = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(xgb_base, param_grid, scoring='average_precision', cv=cv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)

        print(f"Best XGBoost params: {grid.best_params_}")
        self.best_model = grid.best_estimator_

        self._evaluate(self.best_model, X_test, y_test, "XGBoost")
        return self.best_model

    def _evaluate(self, model, X, y, name):
        prob = model.predict_proba(X)[:, 1]
        pred = model.predict(X)
        print(f"\n{name} Performance:")
        print(f"PR-AUC: {self._pr_auc(y, prob):.4f}")
        print(f"F1-Score: {f1_score(y, pred):.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y, pred))

    def save_best_model(self, feature_names):
        path = MODELS_DIR / f"{self.dataset_name}_best_model.pkl"
        joblib.dump({
            'model': self.best_model,
            'feature_names': feature_names,
            'dataset': self.dataset_name
        }, path)
        print(f"Best model saved to {path}")