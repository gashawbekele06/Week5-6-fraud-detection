# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from .config import DATA_RAW, DATA_PROCESSED


class ECommerceFraudPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

    def _ip_to_int(self, ip: str) -> int:
        try:
            return int(''.join(f'{int(x):08b}' for x in ip.split('.')), 2)
        except:
            return np.nan

    def load_and_merge(self):
        fraud_path = DATA_RAW / "Fraud_Data.csv"
        ip_path = DATA_RAW / "IpAddress_to_Country.csv"

        df = pd.read_csv(fraud_path)
        ip_df = pd.read_csv(ip_path)

        df['ip_address'] = df['ip_address'].apply(self._ip_to_int)

        ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype('int64')
        ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype('int64')

        df_sorted = df.sort_values('ip_address')
        ip_sorted = ip_df.sort_values('lower_bound_ip_address')

        merged = pd.merge_asof(
            df_sorted,
            ip_sorted,
            left_on='ip_address',
            right_on='lower_bound_ip_address',
            direction='backward'
        )

        merged = merged[merged['ip_address'] <= merged['upper_bound_ip_address']].copy()
        merged['country'].fillna('Unknown', inplace=True)

        self.df = merged.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], errors='ignore')
        return self.df

    def clean_and_engineer(self):
        df = self.df.copy()

        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600

        for group in ['user_id', 'device_id']:
            df = df.sort_values(['purchase_time'])
            for window in ['1H', '1D', '7D']:
                df[f'trans_{group}_{window}'] = (
                    df.groupby(group)['purchase_time']
                    .transform(lambda s: s.rolling(window, closed='left').count())
                    .fillna(0)
                )

        self.df = df
        return df

    def transform_and_split(self, test_size=0.2, random_state=42):
        numerical = [
            'purchase_value', 'age', 'time_since_signup_hours',
            'hour_of_day', 'day_of_week',
            'trans_user_id_1H', 'trans_user_id_1D', 'trans_user_id_7D',
            'trans_device_id_1H', 'trans_device_id_1D', 'trans_device_id_7D'
        ]
        categorical = ['source', 'browser', 'sex', 'country']

        df = self.df.copy()
        df[numerical] = self.scaler.fit_transform(df[numerical])
        df = pd.get_dummies(df, columns=categorical, drop_first=True)

        X = df.drop('class', axis=1)
        y = df['class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        self.feature_names = X.columns.tolist()

        # Save processed data
        X_train_res.to_parquet(DATA_PROCESSED / "X_train_ecommerce.parquet")
        X_test.to_parquet(DATA_PROCESSED / "X_test_ecommerce.parquet")
        y_train_res.to_series().to_parquet(DATA_PROCESSED / "y_train_ecommerce.parquet")
        y_test.to_series().to_parquet(DATA_PROCESSED / "y_test_ecommerce.parquet")

        return X_train_res, X_test, y_train_res, y_test