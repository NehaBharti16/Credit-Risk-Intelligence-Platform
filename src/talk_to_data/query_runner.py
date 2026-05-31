import sqlite3
import pandas as pd
import os
from src.utils.config import DB_PATH

def get_connection():
    """Get database connection"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn

def run_query(sql):
    """Execute SQL and return results as DataFrame"""
    try:
        conn = get_connection()
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

def load_data_to_db():
    """Load CSV data into SQLite database"""
    print("Loading data into SQLite database...")
    
    df = pd.read_csv("./data/application_train.csv")
    
    # Add derived columns
    df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).round(1)
    df['LOAN_INCOME_RATIO'] = (df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']).round(2)
    
    # Select only needed columns
    cols = [
        'SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
        'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
        'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'OCCUPATION_TYPE',
        'AGE_YEARS', 'LOAN_INCOME_RATIO'
    ]
    
    # Keep only columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Save to SQLite
    conn = get_connection()
    df.to_sql('applications', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✅ {len(df):,} records loaded into SQLite at {DB_PATH}")
    return len(df)

def get_db_stats():
    """Get basic database statistics"""
    sql = "SELECT COUNT(*) as total, SUM(TARGET) as defaults FROM applications"
    df, err = run_query(sql)
    if err:
        return None
    return df