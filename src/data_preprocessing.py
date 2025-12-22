# src/data_preprocessor.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    FRAUD_DATA_FULL_PATH,
    IP_MAPPING_FULL_PATH,
    CREDITCARD_FULL_PATH,
    RANDOM_STATE,
    ECOMMERCE_CATEGORICAL_COLS,
    ECOMMERCE_NUMERICAL_COLS,
    ip_to_int,
    MIN_COUNTRY_TXNS_FOR_FRAUD_RATE
)


class FraudDataPreprocessor:
    def __init__(self):
        self.raw_path = RAW_DATA_PATH
        self.processed_path = PROCESSED_DATA_PATH
        try:
            os.makedirs(self.processed_path, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(f"Cannot create processed directory '{self.processed_path}': {e}")
        
        self.preprocessor = None  # ColumnTransformer for e-commerce
        self.fraud_df = None
        self.ip_df = None
        self.credit_df = None

    def load_data(self):
        """Load raw datasets with error handling."""
        print("Loading datasets...\n")
        try:
            self.fraud_df = pd.read_csv(FRAUD_DATA_FULL_PATH)
            self.ip_df = pd.read_csv(IP_MAPPING_FULL_PATH)
            self.credit_df = pd.read_csv(CREDITCARD_FULL_PATH)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Dataset file not found: {e.filename}")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV parsing error: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error loading data: {str(e)}")

        print(f"Fraud_Data loaded: {self.fraud_df.shape}")
        print(f"IP Mapping loaded: {self.ip_df.shape}")
        print(f"Credit Card loaded: {self.credit_df.shape}\n")

    def clean_fraud_data(self):
        """Clean e-commerce fraud dataset with validation."""
        print("Cleaning Fraud_Data...\n")
        if self.fraud_df is None:
            raise ValueError("Fraud data not loaded. Call load_data() first.")

        try:
            print(f"Missing values:\n{self.fraud_df.isnull().sum()}\n")
            print(f"Duplicates before: {self.fraud_df.duplicated().sum()}\n")

            # Convert timestamps
            self.fraud_df['signup_time'] = pd.to_datetime(self.fraud_df['signup_time'], errors='raise')
            self.fraud_df['purchase_time'] = pd.to_datetime(self.fraud_df['purchase_time'], errors='raise')

            # Convert IP to integer
            self.fraud_df['ip_address_int'] = self.fraud_df['ip_address'].apply(ip_to_int)
            if self.fraud_df['ip_address_int'].isnull().any():
                print("Warning: Some IP addresses could not be converted to integers.")

            # Drop duplicates
            self.fraud_df.drop_duplicates(inplace=True)

            print(f"Duplicates after: {self.fraud_df.duplicated().sum()}")
            print("Cleaning complete.\n")
        except Exception as e:
            raise RuntimeError(f"Error in clean_fraud_data: {str(e)}")

    def merge_geolocation(self):
        """Merge IP to country with error handling."""
        print("Merging geolocation...\n")
        if self.fraud_df is None or self.ip_df is None:
            raise ValueError("Required dataframes not loaded.")

        try:
            ip_sorted = self.ip_df.sort_values('lower_bound_ip_address').reset_index(drop=True)
            bounds = ip_sorted['lower_bound_ip_address'].values

            # Handle invalid IP integers
            valid_ips = self.fraud_df['ip_address_int'].dropna()
            if len(valid_ips) != len(self.fraud_df):
                print(f"Warning: {len(self.fraud_df) - len(valid_ips)} invalid IP addresses found.")

            indices = np.searchsorted(bounds, valid_ips.values) - 1
            indices = np.clip(indices, 0, len(ip_sorted) - 1)

            # Assign countries (handle NaN IPs)
            self.fraud_df['country'] = pd.NA
            self.fraud_df.loc[self.fraud_df['ip_address_int'].notna(), 'country'] = \
                ip_sorted.iloc[indices]['country'].values
            self.fraud_df['country'].fillna('Unknown', inplace=True)

            print(f"Unique countries: {self.fraud_df['country'].nunique()}")
            print("Geolocation merge complete.\n")
        except Exception as e:
            raise RuntimeError(f"Error in merge_geolocation: {str(e)}")

    def feature_engineering(self):
        """Create new features with validation."""
        print("Feature Engineering...\n")
        if self.fraud_df is None:
            raise ValueError("Fraud data not loaded.")

        try:
            df = self.fraud_df
            df['hour_of_day'] = df['purchase_time'].dt.hour
            df['day_of_week'] = df['purchase_time'].dt.dayofweek
            df['time_since_signup_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600.0

            # Handle possible negative or infinite values
            df['time_since_signup_hours'] = df['time_since_signup_hours'].clip(lower=0)

            df['user_transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
            df['device_transaction_count'] = df.groupby('device_id')['device_id'].transform('count')

            print("New features created successfully.\n")
        except Exception as e:
            raise RuntimeError(f"Error in feature_engineering: {str(e)}")

    def analyze_fraud_by_country(self):
        """Analyze fraud by country with validation."""
        print(f"Top 10 countries by fraud rate (min {MIN_COUNTRY_TXNS_FOR_FRAUD_RATE} txns):\n")
        try:
            if 'country' not in self.fraud_df.columns or 'class' not in self.fraud_df.columns:
                raise ValueError("Required columns 'country' or 'class' missing.")

            country_stats = (
                self.fraud_df.groupby('country')['class']
                .agg(['mean', 'count'])
                .rename(columns={'mean': 'fraud_rate'})
            )
            filtered = country_stats[country_stats['count'] >= MIN_COUNTRY_TXNS_FOR_FRAUD_RATE]
            top_risky = filtered.sort_values('fraud_rate', ascending=False).head(10)
            print(top_risky.round(4))
            print()
        except Exception as e:
            print(f"Warning: Analysis failed - {str(e)}")

    def preprocess_for_modeling(self):
        """Scale and encode features with validation."""
        print("Preprocessing e-commerce features...\n")
        try:
            feature_cols = ECOMMERCE_CATEGORICAL_COLS + ECOMMERCE_NUMERICAL_COLS
            missing_cols = [col for col in feature_cols if col not in self.fraud_df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns for preprocessing: {missing_cols}")

            X = self.fraud_df[feature_cols]
            y = self.fraud_df['class']

            self.preprocessor = ColumnTransformer([
                ('num', StandardScaler(), ECOMMERCE_NUMERICAL_COLS),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ECOMMERCE_CATEGORICAL_COLS)
            ])

            X_transformed = self.preprocessor.fit_transform(X)
            print(f"Transformed shape: {X_transformed.shape}")

            features_path = os.path.join(self.processed_path, "ecommerce_features.pkl")
            target_path = os.path.join(self.processed_path, "ecommerce_target.pkl")

            pd.DataFrame(X_transformed).to_pickle(features_path)
            pd.Series(y).to_pickle(target_path)

            print(f"Saved to: {features_path} and {target_path}\n")
        except Exception as e:
            raise RuntimeError(f"Error in preprocess_for_modeling: {str(e)}")

    def process_creditcard(self):
        """Preprocess credit card data with validation."""
        print("Processing creditcard.csv...\n")
        try:
            if self.credit_df is None:
                raise ValueError("Credit card data not loaded.")

            df = self.credit_df.copy()
            print(f"Duplicates before: {df.duplicated().sum()}")
            df.drop_duplicates(inplace=True)
            print(f"Shape after: {df.shape}")

            X = df.drop('Class', axis=1)
            y = df['Class']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            features_path = os.path.join(self.processed_path, "creditcard_features.pkl")
            target_path = os.path.join(self.processed_path, "creditcard_target.pkl")

            pd.DataFrame(X_scaled, columns=X.columns).to_pickle(features_path)
            pd.Series(y).to_pickle(target_path)

            print(f"Saved to: {features_path} and {target_path}\n")
        except Exception as e:
            raise RuntimeError(f"Error in process_creditcard: {str(e)}")

    def handle_imbalance_smote(self, X, y):
        """Apply SMOTE with validation."""
        print("Applying SMOTE...\n")
        try:
            if len(np.unique(y)) < 2:
                raise ValueError("Target variable has only one class - cannot apply SMOTE.")

            print(f"Before SMOTE: {np.bincount(y)}")
            smote = SMOTE(random_state=RANDOM_STATE)
            X_res, y_res = smote.fit_resample(X, y)
            print(f"After SMOTE: {np.bincount(y_res)}")
            return X_res, y_res
        except Exception as e:
            raise RuntimeError(f"Error in handle_imbalance_smote: {str(e)}")

    def run_full_pipeline(self):
        """Run the entire pipeline with try-except wrapper."""
        print("Starting Full Fraud Detection Preprocessing Pipeline\n" + "="*60)
        try:
            self.load_data()
            self.clean_fraud_data()
            self.merge_geolocation()
            self.feature_engineering()
            self.analyze_fraud_by_country()
            self.preprocess_for_modeling()
            self.process_creditcard()
            print("="*60)
            print("🎉 Task 1 Pipeline Completed Successfully!")
            print(f"Processed files saved in: {self.processed_path}")
        except Exception as e:
            print("="*60)
            print(f"Pipeline failed: {str(e)}")
            raise


# Direct execution with error handling
if __name__ == "__main__":
    try:
        preprocessor = FraudDataPreprocessor()
        preprocessor.run_full_pipeline()
    except Exception as e:
        print("\nCRITICAL ERROR: Pipeline execution failed")
        print(f"Details: {str(e)}")
        raise