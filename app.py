import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
import os

# Page config
st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low { color: #27ae60; font-size: 1.5rem; font-weight: bold; }
    .risk-medium { color: #f39c12; font-size: 1.5rem; font-weight: bold; }
    .risk-high { color: #e74c3c; font-size: 1.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/bank.png", width=80)
st.sidebar.title("Credit Risk Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Home",
    "EDA Dashboard", 
    "Risk Prediction",
    "Business Rules",
    "Talk to Data"
])

# ─── HOME PAGE ───────────────────────────────────────────
if page == "Home":
    st.markdown('<p class="main-header">🏦 Credit Risk Intelligence Platform</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applications", "307,511")
    with col2:
        st.metric("Default Rate", "8.1%")
    with col3:
        st.metric("Model AUC", "0.7536")
    with col4:
        st.metric("Features Used", "120")
    
    st.markdown("---")
    st.subheader("Platform Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**EDA Dashboard**\nExplore loan data patterns, distributions, and business insights through interactive charts.")
        st.info("**Risk Prediction**\nInput applicant details and get instant risk score, band classification, and approval decision.")
    with col2:
        st.success("**Talk to Data**\nAsk questions in plain English and get SQL-powered answers from the loan database.")
        st.success("**Model Performance**\nROC-AUC: 0.75 | Trained on 307K applications | SMOTE for class imbalance")

# ─── EDA DASHBOARD ───────────────────────────────────────
elif page == "EDA Dashboard":
    st.title("Exploratory Data Analysis")
    
    @st.cache_data
    def load_data():
        df = pd.read_csv("./data/application_train.csv")
        df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).astype(int)
        df['LOAN_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED'].clip(upper=0) / 365)
        return df
    
    df = load_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Default Count", f"{df['TARGET'].sum():,}")
    col3.metric("Default Rate", f"{df['TARGET'].mean()*100:.1f}%")
    col4.metric("Avg Income", f"₹{df['AMT_INCOME_TOTAL'].mean():,.0f}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Default Analysis", "Data Quality"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='AGE_YEARS', color='TARGET',
                             title='Age Distribution by Default Status',
                             color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                             labels={'TARGET': 'Default', 'AGE_YEARS': 'Age (Years)'},
                             barmode='overlay', opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            income_cap = df['AMT_INCOME_TOTAL'].quantile(0.95)
            fig = px.histogram(df[df['AMT_INCOME_TOTAL'] <= income_cap],
                             x='AMT_INCOME_TOTAL', color='TARGET',
                             title='Income Distribution by Default Status',
                             color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                             barmode='overlay', opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            gender_default = df[df['CODE_GENDER'].isin(['M','F'])].groupby('CODE_GENDER')['TARGET'].mean() * 100
            fig = px.bar(gender_default.reset_index(), x='CODE_GENDER', y='TARGET',
                        title='Default Rate by Gender',
                        color='CODE_GENDER',
                        color_discrete_map={'M': '#3498db', 'F': '#e74c3c'},
                        labels={'TARGET': 'Default Rate (%)', 'CODE_GENDER': 'Gender'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            edu_default = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean() * 100
            fig = px.bar(edu_default.reset_index().sort_values('TARGET', ascending=True),
                        x='TARGET', y='NAME_EDUCATION_TYPE',
                        title='Default Rate by Education',
                        orientation='h',
                        labels={'TARGET': 'Default Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing %': missing_pct.values
        }).query('`Missing %` > 0').sort_values('Missing %', ascending=False).head(20)
        
        fig = px.bar(missing_df, x='Column', y='Missing %',
                    title='Top 20 Columns with Missing Values',
                    color='Missing %', color_continuous_scale='Reds')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

# ─── RISK PREDICTION ─────────────────────────────────────
elif page == "Risk Prediction":
    st.title("Credit Risk Prediction")
    st.markdown("Fill in applicant details to get instant risk assessment.")
    
    @st.cache_resource
    def load_model_artifacts():
        model = joblib.load("./models/lgbm_model.joblib")
        encoders = joblib.load("./models/encoders.joblib")
        threshold = joblib.load("./models/threshold.joblib")
        return model, encoders, threshold
    
    model, encoders, threshold = load_model_artifacts()
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Info")
            gender = st.selectbox("Gender", ["M", "F"])
            age = st.slider("Age (years)", 20, 70, 35)
            children = st.number_input("Number of Children", 0, 10, 0)
            family_status = st.selectbox("Family Status", 
                ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"])
            education = st.selectbox("Education", 
                ["Secondary / secondary special", "Higher education", 
                 "Incomplete higher", "Lower secondary", "Academic degree"])
        
        with col2:
            st.subheader("Financial Info")
            income = st.number_input("Annual Income (₹)", 50000, 10000000, 200000, step=10000)
            credit = st.number_input("Loan Amount (₹)", 50000, 5000000, 500000, step=10000)
            annuity = st.number_input("Monthly Payment (₹)", 5000, 200000, 25000, step=1000)
            own_car = st.selectbox("Owns Car?", ["N", "Y"])
            own_realty = st.selectbox("Owns Realty?", ["Y", "N"])
        
        with col3:
            st.subheader("Employment Info")
            income_type = st.selectbox("Income Type",
                ["Working", "Commercial associate", "Pensioner", 
                 "State servant", "Unemployed"])
            occupation = st.selectbox("Occupation",
                ["Laborers", "Core staff", "Accountants", "Managers",
                 "Drivers", "Sales staff", "Cleaning staff", "Cooking staff"])
            employment_years = st.slider("Years Employed", 0, 40, 5)
            housing = st.selectbox("Housing Type",
                ["House / apartment", "With parents", "Municipal apartment",
                 "Rented apartment", "Office apartment"])
            region_rating = st.selectbox("Region Rating", [1, 2, 3])
        
        submitted = st.form_submit_button("Assess Risk", use_container_width=True)
    
    if submitted:
        # Build input
        input_data = {
            'NAME_CONTRACT_TYPE': 'Cash loans',
            'CODE_GENDER': gender,
            'FLAG_OWN_CAR': own_car,
            'FLAG_OWN_REALTY': own_realty,
            'CNT_CHILDREN': children,
            'AMT_INCOME_TOTAL': income,
            'AMT_CREDIT': credit,
            'AMT_ANNUITY': annuity,
            'AMT_GOODS_PRICE': credit,
            'NAME_INCOME_TYPE': income_type,
            'NAME_EDUCATION_TYPE': education,
            'NAME_FAMILY_STATUS': family_status,
            'NAME_HOUSING_TYPE': housing,
            'DAYS_BIRTH': int(-age * 365),
            'DAYS_EMPLOYED': int(-employment_years * 365),
            'FLAG_MOBIL': 1,
            'FLAG_EMP_PHONE': 1,
            'CNT_FAM_MEMBERS': children + 1,
            'REGION_RATING_CLIENT': region_rating,
            'OCCUPATION_TYPE': occupation
        }
        
        # Fill remaining features with defaults
        df_sample = pd.read_csv("./data/application_train.csv").drop(
            columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
        defaults = df_sample.median(numeric_only=True).to_dict()
        
        for col in df_sample.columns:
            if col not in input_data:
                if df_sample[col].dtype == 'object':
                    input_data[col] = df_sample[col].mode()[0]
                else:
                    input_data[col] = defaults.get(col, 0)
        
        # Preprocess and predict
        from src.data.preprocessor import preprocess
        df_input = pd.DataFrame([input_data])
        df_processed, _, _ = preprocess(df_input, fit=False, 
                                         encoders=encoders, save_encoders=False)
        
        probability = model.predict_proba(df_processed)[:, 1][0]
        risk_score = round(probability * 100, 2)
        
        if probability < 0.2:
            risk_band = "LOW RISK"
            decision = "APPROVE"
            color = "green"
        elif probability < 0.5:
            risk_band = "MEDIUM RISK"
            decision = "REVIEW"
            color = "orange"
        else:
            risk_band = "HIGH RISK"
            decision = "REJECT"
            color = "red"
        
        st.markdown("---")
        st.subheader("Risk Assessment Result")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", f"{risk_score}/100")
        col2.metric("Risk Band", risk_band)
        col3.metric("Decision", decision)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 20], 'color': '#d5f5e3'},
                    {'range': [20, 50], 'color': '#fef9e7'},
                    {'range': [50, 100], 'color': '#fdedec'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# SHAP Explanation
        st.markdown("---")
        st.subheader("Why this prediction? (SHAP Explanation)")
        
        with st.spinner("Generating SHAP explanation..."):
            from src.ml.predict import explain_prediction
            shap_df = explain_prediction(df_processed)
        
        if shap_df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top factors increasing risk 🔴**")
                risk_up = shap_df[shap_df['SHAP Value'] > 0][['Feature','SHAP Value']].head(5)
                for _, row in risk_up.iterrows():
                    st.markdown(f"🔴 **{row['Feature']}** → +{row['SHAP Value']:.4f}")
            
            with col2:
                st.markdown("**Top factors decreasing risk 🟢**")
                risk_down = shap_df[shap_df['SHAP Value'] < 0][['Feature','SHAP Value']].head(5)
                for _, row in risk_down.iterrows():
                    st.markdown(f"🟢 **{row['Feature']}** → {row['SHAP Value']:.4f}")
            
            # SHAP Bar Chart
            import plotly.express as px
            shap_df['Color'] = shap_df['SHAP Value'].apply(
                lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
            fig_shap = px.bar(
                shap_df.sort_values('SHAP Value'),
                x='SHAP Value', y='Feature',
                color='Color',
                color_discrete_map={
                    'Increases Risk': '#e74c3c',
                    'Decreases Risk': '#2ecc71'
                },
                title='Feature Impact on Risk Score (SHAP Values)',
                orientation='h'
            )
            fig_shap.update_layout(height=400)
            st.plotly_chart(fig_shap, use_container_width=True)
            
            st.info("**How to read this:** Red bars mean the feature INCREASES default risk. Green bars mean it DECREASES risk. Longer bars = stronger impact.")

# ─── TALK TO DATA ─────────────────────────────────────────
elif page == "Talk to Data":
    st.title("Talk to Data")
    st.markdown("Ask questions about the loan data in plain English!")

    col_mem1, col_mem2 = st.columns([4,1])
    with col_mem2:
        if st.button("Clear Memory"):
            from src.talk_to_data.nl_to_sql import clear_memory
            clear_memory()
            st.success("Memory cleared!")
    
    # Show conversation history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sample questions
    st.subheader("Try these questions:")
    sample_questions = [
        "How many people defaulted on their loans?",
        "What is the average income by gender?",
        "Show default rate by education type",
        "Top 5 occupations with highest default rate",
        "How many applicants own a car?"
    ]
    
    cols = st.columns(len(sample_questions))
    for i, q in enumerate(sample_questions):
        if cols[i].button(q, key=f"q{i}"):
            st.session_state['question'] = q
    
    question = st.text_input("Or type your own question:", 
                             value=st.session_state.get('question', ''),
                             placeholder="e.g. What is the average loan amount by income type?")
    
    if st.button("Ask", type="primary") and question:
        from src.talk_to_data.nl_to_sql import ask
        
        with st.spinner("Generating SQL and fetching results..."):
            result = ask(question)
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success("Query executed successfully!")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Generated SQL")
                st.code(result['sql'], language='sql')
            with col2:
                st.subheader(f"Results ({result['row_count']} rows)")
                st.dataframe(result['results'], use_container_width=True)
            
            # Auto chart if numeric results
            if result['row_count'] > 1 and len(result['results'].columns) >= 2:
                try:
                    df_res = result['results']
                    num_cols = df_res.select_dtypes(include='number').columns
                    cat_cols = df_res.select_dtypes(include='object').columns
                    
                    if len(num_cols) >= 1 and len(cat_cols) >= 1:
                        fig = px.bar(df_res, x=cat_cols[0], y=num_cols[0],
                                    title=f"Chart: {question}")
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

    # ─── BUSINESS RULES ──────────────────────────────────────
elif page == "Business Rules":
    st.title("Business Decision Rules")
    st.markdown("Interpretable rules derived from ML model and EDA insights.")
    
    tab1, tab2 = st.tabs(["Check Applicant Rules", "ML Decision Tree Rules"])
    
    with tab1:
        st.subheader("Check Rule-Based Decision for an Applicant")
        
        col1, col2 = st.columns(2)
        with col1:
            r_income = st.number_input("Annual Income", 50000, 5000000, 200000)
            r_credit = st.number_input("Loan Amount", 50000, 5000000, 500000)
            r_age = st.slider("Age", 20, 70, 35)
            r_employed = st.slider("Years Employed", 0, 40, 5)
        with col2:
            r_children = st.number_input("Children", 0, 10, 0)
            r_education = st.selectbox("Education", [
                "Higher education", "Secondary / secondary special",
                "Lower secondary", "Incomplete higher", "Academic degree"
            ])
        
        if st.button("Check Rules", type="primary"):
            from src.ml.rules import get_rule_based_decision
            input_dict = {
                'AMT_INCOME_TOTAL': r_income,
                'AMT_CREDIT': r_credit,
                'DAYS_BIRTH': int(-r_age * 365),
                'DAYS_EMPLOYED': int(-r_employed * 365),
                'CNT_CHILDREN': r_children,
                'NAME_EDUCATION_TYPE': r_education
            }
            result = get_rule_based_decision(input_dict)
            
            st.markdown("---")
            st.subheader(result['rule_decision'])
            st.metric("Risk Flags", f"{result['risk_flags']}/5")
            
            st.subheader("Rules Applied:")
            for rule in result['rules_applied']:
                st.markdown(rule)
    
    with tab2:
        st.subheader("ML Decision Tree Rules")
        from src.ml.rules import get_readable_rules
        rules_text = get_readable_rules()
        st.code(rules_text[:3000], language='text')