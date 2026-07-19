import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

/* Button */
div.stButton > button:first-child{
    background: linear-gradient(90deg,#2563EB,#06B6D4);
    color:white;
    font-size:20px;
    font-weight:bold;
    border-radius:12px;
    border:none;
    padding:12px;
    width:100%;
    transition:0.3s;
}

div.stButton > button:first-child:hover{
    background:linear-gradient(90deg,#1D4ED8,#0891B2);
    transform:scale(1.03);
    color:white;
}

/* Title */
h1{
    color:#1E3A8A;
    text-align:center;
}

/* Headers */
h2,h3{
    color:#2563EB;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model Files
# -------------------------------



BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "random_forest_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
feature_columns = joblib.load(BASE_DIR / "models" / "feature_columns.pkl")

# -------------------------------
# Title
# -------------------------------

# -------------------------------
# Professional Header
# -------------------------------

st.markdown("""
<div style="text-align:center; padding:15px;">

<h1 style="
color:#1E3A8A;
font-size:48px;
font-weight:bold;
margin-bottom:5px;">
🏦 Bank Customer Churn Prediction
</h1>

<h4 style="
color:#475569;
font-size:22px;
margin-top:0px;">
AI-Powered Customer Retention Dashboard
</h4>

<p style="
font-size:18px;
color:#64748B;
max-width:900px;
margin:auto;
line-height:1.8;">

Predict whether a bank customer is likely to leave the bank using a
<strong>Random Forest Machine Learning Model.</strong>

This dashboard analyzes customer information such as
Credit Score, Age, Balance, Geography, Products, and Account Activity
to estimate the probability of customer churn.

</p>

</div>
""", unsafe_allow_html=True)

st.markdown("---")



# ==========================================
# Customer Input Section
# ==========================================

st.header("Enter Customer Details")

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_products = st.selectbox(
    "Number of Products",
    [1, 2, 3, 4]
)

has_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=100000.0
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)
# ==========================================
# Prepare Input Data
# ==========================================

gender = 1 if gender == "Male" else 0

geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_card],
    "IsActiveMember": [active_member],
    "EstimatedSalary": [salary],
    "Geography_Germany": [geography_germany],
    "Geography_Spain": [geography_spain]
})

input_data = input_data[feature_columns]

scaled_input = scaler.transform(input_data)

if st.button(" 🚀 Predict Churn"):

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Details")

        st.write(f"**Credit Score:** {credit_score}")
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {'Male' if gender == 1 else 'Female'}")
        st.write(f"**Balance:** ₹ {balance:,.2f}")
        st.write(f"**Estimated Salary:** ₹ {salary:,.2f}")

    with col2:
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Customer is likely to churn.")
        else:
            st.success("✅ Customer is likely to stay.")

        st.metric(
            label="Churn Probability",
            value=f"{probability:.2%}"
        )

        st.progress(float(probability))

        st.sidebar.title("🏦 Bank Churn Dashboard")


# ==========================================
# Sidebar
# ==========================================

st.sidebar.markdown("""
# 🏦 Navigation

Welcome to the **Bank Customer Churn Prediction Dashboard**.

This application predicts whether a customer is likely to leave the bank using a trained Random Forest Machine Learning model.
""")

st.sidebar.markdown("---")

st.sidebar.success("✅ Model : Random Forest")
st.sidebar.info("📊 Dataset : Bank Customer Churn")
st.sidebar.info("🤖 Machine Learning Project")


st.markdown("---")

st.caption(
    "Developed by Anchal Patel | Bank Customer Churn Prediction Project"
)

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:gray;">

### 👨‍💻 Developed by Anchal Patel

Bank Customer Churn Prediction using Machine Learning

© 2026 All Rights Reserved

</div>
""", unsafe_allow_html=True)


    # -----------------------------------------------------
    #  streamlit run dashboard/app.py
    # ------------------------------------------------------