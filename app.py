import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

st.set_page_config(
    page_title="E-Commerce Churn & Retention Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0d0e15;
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] {
        background-color: #131525;
        border-right: 1px solid #2d3250;
    }
    
    div[data-testid="stMetric"] {
        background: rgba(25, 28, 48, 0.6);
        border: 1px solid #7c3aed;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.25);
        border-radius: 15px;
        padding: 18px;
        backdrop-filter: blur(10px);
    }

    .stButton>button {
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.8);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #191c30;
        border-radius: 8px;
        color: #a0aec0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7c3aed !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = XGBClassifier()
    model.load_model('xgb_churn_model.json')
    features = joblib.load('model_features.pkl')
    return model, features

model, model_features = load_assets()

st.title("📊 E-Commerce Customer Churn & Retention Dashboard")
st.markdown("Predict customer churn risks and identify retention opportunities using Machine Learning.")

st.sidebar.header("User Input: Single Customer Metrics")

frequency = st.sidebar.number_input("Order Frequency (Total Orders)", min_value=1, value=5)
monetary = st.sidebar.number_input("Total Monetary Value ($)", min_value=0.0, value=250.0)
total_profit = st.sidebar.number_input("Total Profit Generated ($)", min_value=-100.0, value=50.0)
avg_rating = st.sidebar.slider("Average Customer Rating", 1.0, 5.0, 4.0)
avg_delivery_days = st.sidebar.number_input("Average Delivery Days", min_value=1.0, value=3.5)
total_returns = st.sidebar.number_input("Total Returned Orders", min_value=0, value=0)
customer_age = st.sidebar.number_input("Customer Age", min_value=18, max_value=100, value=30)

gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Non-Binary"])
region = st.sidebar.selectbox("Region", ["North", "South", "East", "West", "Central"])
customer_segment = st.sidebar.selectbox("Customer Segment", ["Consumer", "Premium", "Corporate"])

tab1, tab2 = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction (CSV)"])

with tab1:
    st.subheader("Individual Customer Risk Assessment")
    
    if st.button("Predict Churn Risk"):
        input_data = {feat: 0 for feat in model_features}
        
        input_data['frequency'] = frequency
        input_data['monetary'] = monetary
        input_data['total_profit'] = total_profit
        input_data['avg_rating'] = avg_rating
        input_data['avg_delivery_days'] = avg_delivery_days
        input_data['total_returns'] = total_returns
        input_data['customer_age'] = customer_age
        
        if f"gender_{gender}" in input_data:
            input_data[f"gender_{gender}"] = 1
        if f"region_{region}" in input_data:
            input_data[f"region_{region}"] = 1
        if f"customer_segment_{customer_segment}" in input_data:
            input_data[f"customer_segment_{customer_segment}"] = 1

        input_df = pd.DataFrame([input_data])
        
        churn_prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Churn Probability", value=f"{churn_prob * 100:.1f}%")
            
        with col2:
            if prediction == 1:
                st.error("Risk Status: HIGH CHURN RISK")
            else:
                st.success("Risk Status: RETAINED / LOW RISK")

        st.markdown("### 💡 Recommended Business Action")
        if churn_prob > 0.6:
            st.warning("Action Required: High risk customer. Send an exclusive retention discount code via email immediately.")
        else:
            st.info("Action Required: Customer is active. Target with cross-sell and loyalty rewards.")

with tab2:
    st.subheader("Batch Customer Risk Assessment")
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset Preview:", batch_df.head())
        st.info("Batch scoring feature ready for deployment.")