import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report,
                             average_precision_score)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib
import os

def get_risk_band(probability):
    """Convert probability score to risk band"""
    if probability < 0.2:
        return "Low Risk"
    elif probability < 0.5:
        return "Medium Risk"
    else:
        return "High Risk"

def train_model(X, y):
    print("Starting model training...")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Default rate in train: {y_train.mean():.3f}")

    # Apply SMOTE
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Train shape: {X_train_sm.shape}")
    print(f"After SMOTE - Default rate: {y_train_sm.mean():.3f}")

    # LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # Train
    model.fit(
        X_train_sm, y_train_sm,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100)
        ]
    )

    # Evaluate with threshold 0.2
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.2).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    ap = average_precision_score(y_val, y_pred_proba)

    print(f"\n{'='*40}")
    print(f"ROC-AUC Score:  {auc:.4f}")
    print(f"Avg Precision:  {ap:.4f}")
    print(f"\nClassification Report (threshold=0.2):")
    print(classification_report(y_val, y_pred))
    print('='*40)

    # Save model
    os.makedirs("./models", exist_ok=True)
    joblib.dump(model, "./models/lgbm_model.joblib")

    # Save threshold
    joblib.dump(0.2, "./models/threshold.joblib")
    print("Model saved to models/lgbm_model.joblib")
    print("Threshold saved to models/threshold.joblib")

    return model, auc, ap

def load_model():
    """Load saved model and threshold"""
    model = joblib.load("./models/lgbm_model.joblib")
    threshold = joblib.load("./models/threshold.joblib")
    return model, threshold