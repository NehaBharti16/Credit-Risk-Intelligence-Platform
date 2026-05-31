# 🏦 Credit Risk Intelligence Platform

An AI-powered credit risk platform built on the Home Credit Default Risk dataset from Kaggle.

## 🎯 Overview
Design and build a lightweight AI-powered platform on a structured dataset. This platform tests ability to work across the full AI engineering stack from data analysis to a deployed application.

## ✨ Features
- 📊 **EDA Dashboard** — Interactive charts and 7 business insights
- 🤖 **Risk Prediction** — LightGBM model with SMOTE (AUC: 0.754)
- 🔍 **Explainable AI** — SHAP values showing why each prediction was made
- 📜 **Business Rules** — Interpretable decision rules derived from ML
- 💬 **Talk to Data** — Natural language to SQL using Groq LLM
- 🐳 **Dockerized** — One command deployment

## 🏗️ Architecture
User → Streamlit UI (5 sections)
↓
┌──────────────────────────────┐
│  EDA Dashboard               │ → Plotly Charts + Insights
│  Risk Prediction             │ → LightGBM + SHAP
│  Business Rules              │ → Decision Tree Rules
│  Talk to Data                │ → Groq LLM + SQLite
└──────────────────────────────┘
↓
Data Layer: CSV + SQLite DB
Model Layer: LightGBM + SMOTE
LLM Layer: Groq (llama-3.3-70b)

## 📁 Project Structure
credit_risk_platform/
├── data/                          # Dataset files (not in git)
│   └── application_train.csv      # Home Credit dataset
├── documents/                     # Project presentation PDF
├── models/                        # Saved ML artifacts
│   ├── lgbm_model.joblib
│   ├── encoders.joblib
│   ├── threshold.joblib
│   ├── rules_model.joblib
│   └── business_rules.txt
├── notebooks/
│   ├── eda.ipynb                  # Exploratory Data Analysis
│   └── eda.py                     # Converted notebook
├── src/
│   ├── data/
│   │   ├── loader.py              # Load dataset
│   │   └── preprocessor.py        # Clean + encode data
│   ├── ml/
│   │   ├── train.py               # Model training pipeline
│   │   ├── predict.py             # Inference + SHAP
│   │   ├── evaluate.py            # Metrics + ROC curve
│   │   └── rules.py               # Business rule derivation
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py           # NL → SQL with memory
│   │   ├── query_runner.py        # Execute SQL on SQLite
│   │   └── prompt_templates.py    # Prompts + validation
│   └── utils/
│       ├── config.py              # Settings + API keys
│       ├── logger.py              # Logging setup
│       ├── helpers.py             # Utility functions
│       └── docker_utils.py        # Docker path utilities
├── sql/
│   ├── schema.sql                 # DB schema
│   └── credit_risk.db             # SQLite database
├── app.py                         # Main Streamlit app
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker + Docker Compose
- Groq API Key (free at console.groq.com)
- Dataset from Kaggle (link below)

### Dataset
Download from: https://www.kaggle.com/competitions/home-credit-default-risk/data
- Download `application_train.csv` and `application_test.csv`
- Place both files in the `data/` folder

### Option 1: Docker (Recommended)
```bash
# Clone repository
git clone https://github.com/yourusername/credit_risk_platform.git
cd credit_risk_platform

# Setup environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Add dataset to data/ folder

# Run with Docker
docker-compose up --build
```
Open http://localhost:8501

### Option 2: Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment  
cp .env.example .env
# Add your GROQ_API_KEY to .env

# Add dataset files to data/ folder

# One command setup (trains model, loads DB, generates PDF)
python setup.py

# Run app
streamlit run app.py
```

## 🔑 Environment Variables
GROQ_API_KEY=your_groq_api_key_here
DB_PATH=./sql/credit_risk.db
APP_HOST=0.0.0.0
APP_PORT=8501

## 📊 Model Details

### Algorithm: LightGBM Classifier
LightGBM was chosen because:
- Handles large datasets (307K rows) efficiently
- Built-in support for imbalanced data
- Outputs probability scores for risk ranking
- Industry standard for credit risk modeling

### Class Imbalance Strategy: SMOTE
- Original default rate: 8.1% (highly imbalanced)
- SMOTE sampling strategy: 0.5
- After SMOTE default rate: 33.3%
- Decision threshold: 0.2 (optimized for recall)

### Evaluation Metrics
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.7620 |
| Avg Precision | 0.2458 |
| Accuracy | 0.88 |
| Default Recall | 0.33 |
| Default Precision | 0.27 |
| Default F1 | 0.30 |
| Features Used | 127 |

## 🔍 Explainable AI (SHAP)
Every prediction includes SHAP values showing:
- Which features increased the risk score (red)
- Which features decreased the risk score (green)
- Magnitude of each feature's impact

## 📜 Business Rules
Interpretable rules derived from Decision Tree + EDA:

| Rule | Condition | Risk Impact |
|------|-----------|-------------|
| Age | Age < 30 years | Higher default risk |
| Loan Ratio | Loan > 5x income | High risk |
| Employment | < 2 years employed | Higher risk |
| Education | Lower secondary | Slightly higher risk |
| Dependents | > 3 children | Financial stress |

Key finding: **EXT_SOURCE_2 and EXT_SOURCE_3** (external credit scores) are the strongest predictors of default.

## 💬 Prompt Engineering & Hallucination Control
- System prompt with strict schema definition
- SQL validation before execution (SELECT only)
- Conversation memory (last 3 exchanges)
- Temperature set to 0.1 for consistent outputs
- Column whitelist to prevent hallucinations
- Model: llama-3.3-70b-versatile (Groq)

## 📈 EDA Business Insights
1. Younger applicants (< 30) default more
2. Lower income applicants default more  
3. Higher loan-to-income ratio = higher risk
4. Cash loans (8.3%) default more than revolving (5.5%)
5. Males (10.1%) default more than females (7.0%)
6. Shorter employment history = higher default risk
7. Lower education level = higher default rate

## 🛠️ Tech Stack
| Component | Technology |
|-----------|-----------|
| ML Model | LightGBM + SMOTE |
| Explainability | SHAP |
| LLM | Groq (llama-3.3-70b-versatile) |
| Database | SQLite |
| UI | Streamlit + Plotly |
| Deployment | Docker + Docker Compose |
| Language | Python 3.10 |

## ⚠️ Known Limitations
- Default recall (0.31) can be improved with more feature engineering
- Dataset not included in repo (must download from Kaggle)
- LLM responses depend on Groq API availability
- SHAP computation adds ~2-3 seconds to prediction time
- Building-related features have ~70% missing values

## 🔧 Possible Improvements
- Add more dataset files (bureau.csv, previous_application.csv)
- Implement XGBoost ensemble for better recall
- Add authentication to the UI
- Deploy to cloud (AWS/GCP/Azure)
- Add batch prediction from CSV upload