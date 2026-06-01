import os
import sys
import subprocess

def run_step(description, func):
    print(f"\n{'='*50}")
    print(f"{description}")
    print('='*50)
    try:
        func()
        print(f"{description} - DONE!")
    except Exception as e:
        print(f"{description} - FAILED: {e}")
        raise

def step1_train_model():
    from src.data.loader import load_train_data
    from src.data.preprocessor import preprocess
    from src.ml.train import train_model
    df = load_train_data()
    X, y, encoders = preprocess(df)
    train_model(X, y)

def step2_generate_rules():
    from src.data.loader import load_train_data
    from src.data.preprocessor import preprocess
    from src.ml.rules import derive_rules
    df = load_train_data()
    X, y, encoders = preprocess(df)
    derive_rules(X, y)

def step3_load_database():
    from src.talk_to_data.query_runner import load_data_to_db
    load_data_to_db()

def step4_generate_presentation():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_presentation", "./generate_presentation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.generate_pdf()

def step5_verify_setup():
    """Verify everything is in place"""
    checks = [
        ("./models/lgbm_model.joblib", "ML Model"),
        ("./models/encoders.joblib", "Encoders"),
        ("./models/threshold.joblib", "Threshold"),
        ("./models/rules_model.joblib", "Rules Model"),
        ("./models/business_rules.txt", "Business Rules"),
        ("./sql/credit_risk.db", "SQLite Database"),
        ("./documents/project_presentation.pdf", "Presentation PDF"),
    ]
    
    print("\nVERIFICATION CHECKLIST:")
    all_good = True
    for path, name in checks:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {name} ({size:,} bytes)")
        else:
            print(f"  {name} - MISSING!")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("\n🏦 CREDIT RISK PLATFORM - AUTO SETUP")
    print("="*50)
    print("This script will:")
    print("  1. Train ML model")
    print("  2. Generate business rules")
    print("  3. Load data into SQLite")
    print("  4. Generate presentation PDF")
    print("  5. Verify everything")
    print("="*50)

    # Check dataset exists
    if not os.path.exists("./data/application_train.csv"):
        print("\nERROR: Dataset not found!")
        print("Please download from Kaggle and place in data/ folder:")
        print("https://www.kaggle.com/competitions/home-credit-default-risk/data")
        sys.exit(1)

    # Run all steps
    run_step("Training ML Model (this takes 8-10 mins)", step1_train_model)
    run_step("Generating Business Rules", step2_generate_rules)
    run_step("Loading Data into SQLite", step3_load_database)
    run_step("Generating Presentation PDF", step4_generate_presentation)
    
    # Verify
    print(f"\n{'='*50}")
    print("Verifying Setup...")
    all_good = step5_verify_setup()
    
    if all_good:
        print("\nSETUP COMPLETE!")
        print("="*50)
        print("Run the app with:")
        print("  streamlit run app.py")
        print("="*50)
    else:
        print("\nSome files are missing. Check errors above.")