# scripts/train_ecommerce.py
from src.preprocessing import ECommerceFraudPreprocessor
from src.modeling import FraudModelTrainer

if __name__ == "__main__":
    print("=== E-Commerce Fraud Detection Pipeline ===")

    prep = ECommerceFraudPreprocessor()
    prep.load_and_merge()
    prep.clean_and_engineer()
    X_train, X_test, y_train, y_test = prep.transform_and_split()

    trainer = FraudModelTrainer(dataset_name="ecommerce_fraud")
    trainer.train_baseline(X_train, y_train, X_test, y_test)
    trainer.train_xgboost(X_train, y_train, X_test, y_test)
    trainer.save_best_model(prep.feature_names)

    print("Pipeline completed successfully!")