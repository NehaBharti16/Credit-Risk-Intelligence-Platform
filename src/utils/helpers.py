import pandas as pd
import numpy as np

def format_currency(amount):
    """Format number as currency"""
    return f"₹{amount:,.2f}"

def format_percentage(value):
    """Format as percentage"""
    return f"{value:.2f}%"

def get_risk_color(risk_band):
    """Get color for risk band"""
    if "Low" in risk_band:
        return "#27ae60"
    elif "Medium" in risk_band:
        return "#f39c12"
    else:
        return "#e74c3c"

def validate_input(input_dict):
    """Validate prediction input"""
    required = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CODE_GENDER']
    missing = [f for f in required if f not in input_dict]
    if missing:
        return False, f"Missing fields: {missing}"
    if input_dict['AMT_INCOME_TOTAL'] <= 0:
        return False, "Income must be positive"
    if input_dict['AMT_CREDIT'] <= 0:
        return False, "Loan amount must be positive"
    return True, "Valid"

def summarize_dataframe(df):
    """Get quick summary of dataframe"""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "numeric_cols": len(df.select_dtypes(include='number').columns),
        "categorical_cols": len(df.select_dtypes(include='object').columns)
    }