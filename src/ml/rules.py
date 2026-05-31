import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
import os

def derive_rules(X, y, max_depth=4):
    """Derive business-readable rules using Decision Tree"""
    print("Deriving business rules from data...")
    
    # Use simple Decision Tree for interpretable rules
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=1000,
        random_state=42,
        class_weight='balanced'
    )
    dt.fit(X, y)
    
    # Export rules as text
    rules_text = export_text(dt, feature_names=list(X.columns))
    
    # Save rules
    os.makedirs("./models", exist_ok=True)
    with open("./models/business_rules.txt", "w") as f:
        f.write("=== CREDIT RISK BUSINESS RULES ===\n\n")
        f.write(rules_text)
    
    print("✅ Business rules saved to models/business_rules.txt")
    joblib.dump(dt, "./models/rules_model.joblib")
    
    return dt, rules_text

def get_readable_rules():
    """Load and return saved business rules"""
    try:
        with open("./models/business_rules.txt", "r") as f:
            return f.read()
    except:
        return "Rules not generated yet. Run training first."

def get_rule_based_decision(input_dict):
    """Get rule-based decision for an applicant"""
    rules = []
    
    # Simple interpretable rules derived from EDA insights
    income = input_dict.get('AMT_INCOME_TOTAL', 0)
    credit = input_dict.get('AMT_CREDIT', 0)
    age_days = input_dict.get('DAYS_BIRTH', -10000)
    employed_days = input_dict.get('DAYS_EMPLOYED', -1000)
    children = input_dict.get('CNT_CHILDREN', 0)
    education = input_dict.get('NAME_EDUCATION_TYPE', '')
    gender = input_dict.get('CODE_GENDER', '')
    
    age_years = abs(age_days) / 365
    employed_years = abs(employed_days) / 365
    loan_income_ratio = credit / income if income > 0 else 999
    
    risk_flags = 0
    
    # Rule 1: Age
    if age_years < 30:
        rules.append("⚠️ Young applicant (age < 30) — Higher default risk")
        risk_flags += 1
    else:
        rules.append("✅ Applicant age is favorable")
    
    # Rule 2: Loan to income ratio
    if loan_income_ratio > 5:
        rules.append("⚠️ High loan-to-income ratio (>5x) — Risky")
        risk_flags += 1
    else:
        rules.append("✅ Loan-to-income ratio is acceptable")
    
    # Rule 3: Employment
    if employed_years < 2:
        rules.append("⚠️ Short employment history (<2 years) — Higher risk")
        risk_flags += 1
    else:
        rules.append("✅ Employment history is stable")
    
    # Rule 4: Education
    if education in ['Lower secondary', 'Secondary / secondary special']:
        rules.append("⚠️ Lower education level — Slightly higher risk")
        risk_flags += 1
    else:
        rules.append("✅ Education level is favorable")
    
    # Rule 5: Children
    if children > 3:
        rules.append("⚠️ Many dependents (>3 children) — Financial stress risk")
        risk_flags += 1
    else:
        rules.append("✅ Number of dependents is manageable")
    
    # Final rule decision
    if risk_flags == 0:
        decision = "✅ RULE-BASED: APPROVE"
    elif risk_flags <= 2:
        decision = "⚠️ RULE-BASED: REVIEW CAREFULLY"
    else:
        decision = "❌ RULE-BASED: REJECT"
    
    return {
        "rules_applied": rules,
        "risk_flags": risk_flags,
        "rule_decision": decision
    }