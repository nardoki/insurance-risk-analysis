import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import os
import matplotlib.pyplot as plt

class ClaimSeverityModel:
    def __init__(self, data):
        self.data = data.copy()
        self.models = {}
        self.results = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

    def prepare_data(self):
        # Filter claims > 0
        df = self.data[self.data['totalclaims'] > 0].copy()
        print(f"Data shape after filtering claims > 0: {df.shape}")

        # Print dtypes for debugging
        print("Column dtypes:")
        print(df.dtypes)

        # Select features
        features = [
            'province', 'postalcode', 'gender', 'vehicletype', 'cubiccapacity',
            'registrationyear', 'covertype', 'maritalstatus', 'suminsured'
        ]
        numeric_features = ['cubiccapacity', 'registrationyear', 'suminsured', 'postalcode']
        string_categorical_features = ['province', 'gender', 'vehicletype', 'covertype', 'maritalstatus']
        target = 'totalclaims'
        exclude_cols = ['transactionmonth', 'vehicleintrodate', 'citizenship', 'legaltype', 'title',
                        'language', 'bank', 'accounttype', 'country', 'maincrestazone', 'subcrestazone',
                        'itemtype', 'make', 'model', 'bodytype', 'alarmimmobiliser', 'trackingdevice',
                        'capitaloutstanding', 'newvehicle', 'termfrequency', 'excessselected',
                        'covercategory', 'covergroup', 'section', 'product', 'statutoryclass',
                        'statutoryrisktype']

        # Feature engineering
        df['vehicle_age'] = 2015 - df['registrationyear']
        df['postcode_claim_freq'] = df.groupby('postalcode')['totalclaims'].transform(lambda x: (x > 0).mean())

        # Handle missing values and whitespace
        for col in features + ['vehicle_age', 'postcode_claim_freq']:
            if col in df.columns:
                if col in string_categorical_features:
                    df.loc[:, col] = df[col].str.strip().replace('', df[col].mode()[0]).fillna(df[col].mode()[0])
                else:
                    df.loc[:, col] = df[col].fillna(df[col].median())

        # Encode categoricals
        # Target encoding for postalcode
        postcode_means = df.groupby('postalcode')[target].mean()
        df.loc[:, 'postalcode_encoded'] = df['postalcode'].map(postcode_means)
        df.loc[:, 'postalcode_encoded'] = df['postalcode_encoded'].fillna(postcode_means.mean())

        # One-hot encoding for string categoricals
        df = pd.get_dummies(df, columns=string_categorical_features, drop_first=True)

        # Select final features
        feature_cols = numeric_features + ['vehicle_age', 'postcode_claim_freq', 'postalcode_encoded'] + \
                       [col for col in df.columns if col.startswith(tuple(string_categorical_features))]
        X = df[feature_cols]
        y = np.log1p(df[target])  # Log-transform target

        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"Error: Non-numeric column {col} detected")
                X.loc[:, col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].mode()[0])
            if X[col].dtype == 'bool':
                X.loc[:, col] = X[col].astype(np.int64)  # Fix FutureWarning

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        print(f"Features used: {self.X_train.columns.tolist()}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        # Initialize models (simplified for small dataset)
        self.models['Linear Regression'] = LinearRegression()
        self.models['Random Forest'] = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.models['XGBoost'] = XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        # Train models
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            print(f"Trained {name}")

    def evaluate_models(self):
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred = np.expm1(y_pred)  # Reverse log-transform
            y_true = np.expm1(self.y_test)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            self.results.append({
                'Model': name,
                'RMSE': rmse,
                'R-squared': r2
            })
            print(f"{name} - RMSE: {rmse:.2f}, R-squared: {r2:.4f}")
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('reports/model_results.csv', index=False)
        print("Results saved to reports/model_results.csv")

    def interpret_model(self, model_name='XGBoost'):
        model = self.models[model_name]
        explainer = shap.Explainer(model, self.X_train)
        shap_values = explainer(self.X_test.sample(20, random_state=42))  # Sample for speed
        shap.summary_plot(shap_values, self.X_test.sample(20, random_state=42), show=False)
        plt.savefig('plots/shap_summary.png')
        plt.close()
        print("SHAP summary plot saved to plots/shap_summary.png")
        # Extract top features
        shap_df = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Mean_SHAP': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('Mean_SHAP', ascending=False).head(10)
        shap_df.to_csv('reports/shap_importance.csv', index=False)
        print("Top 10 features saved to reports/shap_importance.csv")
