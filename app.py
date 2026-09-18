import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="OmniControl | E-Commerce Churn Intelligence",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Cyber Cyan Theme & Grid CSS (Healthcare & Tech Style)
st.markdown("""
    <style>
    /* Main Theme Background */
    .stApp { background-color: #060c12; color: #d0e7f7; }
    [data-testid="stSidebar"] { background-color: #0a141d; border-right: 1px solid #162c3d; }
    
    /* Cyber Cyan KPI Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #0d1e2d 0%, #081420 100%);
        border: 1px solid #00f2fe;
        box-shadow: 0 0 12px rgba(0, 242, 254, 0.15);
        border-radius: 10px;
        padding: 12px;
    }
    div[data-testid="stMetric"] label { color: #7eb0d5 !important; font-size: 0.85rem; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #00f2fe !important; font-weight: 800; }

    /* Custom Buttons & Accent Elements */
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white; border: none; border-radius: 6px;
        padding: 10px 20px; font-weight: bold;
        box-shadow: 0 0 15px rgba(0, 198, 255, 0.4);
    }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(0, 242, 254, 0.8); }

    /* Panel Card Containers */
    .cyber-card {
        background: #0d1a26;
        border: 1px solid #1b384f;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load ML Engine
@st.cache_resource
def load_assets():
    model = XGBClassifier()
    model.load_model('xgb_churn_model.json')
    features = joblib.load('model_features.pkl')
    return model, features

model, model_features = load_assets()

# --- HEADER SECTION (Grid Style) ---
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown("<h1 style='color: #00f2fe; margin-bottom:0;'>🌐 OMNI-RETAIL | CHURN RISK MATRIX</h1>", unsafe_allow_html=True)
    st.caption("Enterprise Customer Retention Predictive Analytics Engine")

with head_col2:
    st.markdown("<div style='text-align:right; color:#7eb0d5; padding-top:15px;'><b>Data Source:</b> Live Production Pipeline<br><b>Model:</b> XGBoost v2.1</div>", unsafe_allow_html=True)

st.divider()

# --- TOP EXECUTIVE KPI ROW ---
k1, k2, k3, k4 = st.columns(4)
k1.metric(label="Active Account Pool", value="138,500", delta="+12.4% vs PY")
k2.metric(label="Total Portfolio Value", value="$286.6M", delta="+$14.2M")
k3.metric(label="System Risk Index", value="18.4%", delta="-2.8% Risk Drop", delta_color="inverse")
k4.metric(label="Predictive Precision", value="62.4%", delta="ROC-AUC: 0.67")

st.write("")

# --- MAIN DASHBOARD LAYOUT (2 UNEQUAL COLUMNS LIKE THE HEALTHCARE DASHBOARD) ---
main_left, main_right = st.columns([2.2, 1])

# LEFT SIDE: VISUAL ANALYTICS MATRIX
with main_left:
    st.markdown("<h3 style='color:#00f2fe;'>📊 Behavioral Risk Breakdown</h3>", unsafe_allow_html=True)
    
    # Row 1: Donut Chart & Feature Importance Side by Side
    chart_c1, chart_c2 = st.columns(2)
    
    with chart_c1:
        labels = ['Low Risk (Active)', 'Critical Churn Risk']
        values = [81.6, 18.4]
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=.65,
            marker=dict(colors=['#0072ff', '#00f2fe']),
            textinfo='percent+label', textfont_size=11
        )])
        fig_donut.update_layout(
            title="Portfolio Distribution", title_font_color="#7eb0d5",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#d0e7f7'), showlegend=False,
            margin=dict(t=30, b=10, l=10, r=10), height=230
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with chart_c2:
        drivers_df = pd.DataFrame({
            'Factor': ['Order Freq', 'Monetary', 'Profit Margin', 'Delivery Time', 'Rating'],
            'Score': [37.9, 22.4, 18.1, 12.3, 9.3]
        }).sort_values('Score', ascending=True)
        
        fig_bar = px.bar(
            drivers_df, x='Score', y='Factor', orientation='h',
            color='Score', color_continuous_scale=['#0052d4', '#00f2fe']
        )
        fig_bar.update_layout(
            title="Key Churn Drivers (%)", title_font_color="#7eb0d5",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#d0e7f7'), coloraxis_showscale=False,
            margin=dict(t=30, b=10, l=10, r=10), height=230
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Row 2: Trend / Area Chart
    st.markdown("<h3 style='color:#00f2fe; margin-top:15px;'>📈 Churn Probability Trend by Segment</h3>", unsafe_allow_html=True)
    
    trend_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
        'Consumer Risk': [22, 24, 21, 19, 18, 20, 17, 18],
        'Corporate Risk': [12, 11, 14, 10, 9, 11, 8, 9]
    })
    fig_trend = px.area(
        trend_data, x='Month', y=['Consumer Risk', 'Corporate Risk'],
        color_discrete_sequence=['#00f2fe', '#0072ff']
    )
    fig_trend.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#d0e7f7'), legend_title_text='',
        margin=dict(t=10, b=10, l=10, r=10), height=220
    )
    st.plotly_chart(fig_trend, use_container_width=True)


# RIGHT SIDE: REAL-TIME INDIVIDUAL CALCULATOR PANEL (LIKE TOP DOCTOR / SIDE PANEL)
with main_right:
    st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#00f2fe; margin-top:0;'>⚡ Real-Time Diagnostic</h3>", unsafe_allow_html=True)
    st.caption("Adjust parameters to score individual account risk.")
    
    frequency = st.number_input("Order Frequency", min_value=1, value=5)
    monetary = st.number_input("Monetary Value ($)", min_value=0.0, value=250.0)
    total_profit = st.number_input("Total Profit ($)", min_value=-100.0, value=50.0)
    avg_rating = st.slider("Customer Rating", 1.0, 5.0, 3.8)
    avg_delivery_days = st.number_input("Avg Delivery Days", min_value=1.0, value=4.0)
    
    with st.expander("Advanced Demographics"):
        total_returns = st.number_input("Returned Orders", min_value=0, value=0)
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=32)
        gender = st.selectbox("Gender", ["Female", "Male", "Non-Binary"])
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        customer_segment = st.selectbox("Segment", ["Consumer", "Premium", "Corporate"])

    st.write("")
    if st.button("RUN CHURN PREDICTION", use_container_width=True):
        input_data = {feat: 0 for feat in model_features}
        input_data['frequency'] = frequency
        input_data['monetary'] = monetary
        input_data['total_profit'] = total_profit
        input_data['avg_rating'] = avg_rating
        input_data['avg_delivery_days'] = avg_delivery_days
        input_data['total_returns'] = total_returns
        input_data['customer_age'] = customer_age
        
        if f"gender_{gender}" in input_data: input_data[f"gender_{gender}"] = 1
        if f"region_{region}" in input_data: input_data[f"region_{region}"] = 1
        if f"customer_segment_{customer_segment}" in input_data: input_data[f"customer_segment_{customer_segment}"] = 1

        input_df = pd.DataFrame([input_data])
        churn_prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        st.divider()
        st.markdown(f"#### Score Result: **{churn_prob * 100:.1f}%**")
        
        if prediction == 1:
            st.error("🚨 HIGH CHURN RISK DETECTED")
            st.info("💡 **Recommended Action:** Deploy automated 15% VIP retention discount code immediately.")
        else:
            st.success("✅ LOW RISK / RETAINED ACCOUNT")
            st.info("💡 **Recommended Action:** Account is stable. Target with standard cross-sell campaigns.")
            
    st.markdown("</div>", unsafe_allow_html=True)