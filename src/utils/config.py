import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Database
DB_PATH = os.getenv("DB_PATH", "./sql/credit_risk.db")

# App Settings
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8501"))

# Data paths
DATA_DIR = "./data"
TRAIN_DATA_PATH = f"{DATA_DIR}/application_train.csv"
TEST_DATA_PATH = f"{DATA_DIR}/application_test.csv"
MODEL_DIR = "./models"