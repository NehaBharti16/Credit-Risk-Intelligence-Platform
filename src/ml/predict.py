import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from src.data.preprocessor import preprocess

def load_artifacts():
    model = joblib.load("./models/lgbm_model.joblib")
    encoders = joblib.load("./models/encoders.joblib")
    threshold = joblib.load("./models/threshold.joblib")
    return model, encoders, threshold

def get_risk_band(probability):
    if probability < 0.2:
        return "Low Risk"
    elif probability < 0.5:
        return "Medium Risk"
    else:
        return "High Risk"

def predict_single(input_dict):
    """Predict risk for a single applicant"""
    model, encoders, threshold = load_artifacts()
    df_input = pd.DataFrame([input_dict])
    df_processed, _, _ = preprocess(
        df_input, fit=False,
        encoders=encoders,
        save_encoders=False
    )
    probability = model.predict_proba(df_processed)[:, 1][0]
    prediction = int(probability >= threshold)
    risk_band = get_risk_band(probability)
    return {
        "probability": round(float(probability), 4),
        "risk_score": round(float(probability) * 100, 2),
        "risk_band": risk_band,
        "prediction": prediction,
        "decision": "REJECT" if prediction == 1 else "APPROVE",
        "processed_df": df_processed
    }

def explain_prediction(df_processed, max_features=10):
    """Generate SHAP explanation for a prediction"""
    try:
        model, _, _ = load_artifacts()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_processed)

        # For binary classification, take class 1 shap values
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        feature_names = df_processed.columns.tolist()
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': sv,
            'Abs SHAP': abs(sv)
        }).sort_values('Abs SHAP', ascending=False).head(max_features)

        return shap_df
    except Exception as e:
        print(f"SHAP error: {e}")
        return None

def plot_shap_bar(shap_df, save_path='./data/shap_explanation.png'):
    """Plot SHAP values as horizontal bar chart"""
    try:
        colors = ['#e74c3c' if v > 0 else '#2ecc71' 
                  for v in shap_df['SHAP Value']]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'], 
                      color=colors, alpha=0.8)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('SHAP Value (Impact on Default Probability)')
        ax.set_title('Feature Impact on Risk Prediction\n(Red = Increases Risk, Green = Decreases Risk)')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        return fig
    except Exception as e:
        print(f"Plot error: {e}")
        return None