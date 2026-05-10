import joblib
import pandas as pd

model_path = "xgboost_stock_delta_model.pkl"
model = joblib.load(model_path)

if hasattr(model, 'feature_names_in_'):
    print(f"Features expected by model: {model.feature_names_in_}")
else:
    print("Model does not have feature_names_in_ attribute.")

try:
    print(f"Number of features: {model.n_features_in_}")
except AttributeError:
    print("Model does not have n_features_in_ attribute.")
