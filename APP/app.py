import streamlit as st
import pandas as pd
import joblib
import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "EDA and ML", "PKL", "Model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "EDA and ML", "PKL", "features.pkl")

# Load model and features
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳")

st.title("💳 Credit Risk Prediction")
st.write("Enter customer details to predict credit risk")

# --- ALL MAPPINGS (from dataset) ---
maps = {
    "status_account": {
        "< 0 DM": "A11", "0 to < 200 DM": "A12", ">= 200 DM": "A13", "no checking account": "A14"
    },
    "status_savings": {
        "< 100 DM": "A61", "100 to < 500 DM": "A62", "500 to < 1000 DM": "A63", 
        ">= 1000 DM": "A64", "unknown/ no savings account": "A65"
    },
    "years_employment": {
        ">= 7 years": "A75", "4 to < 7 years": "A74", "1 to < 4 years": "A73", 
        "< 1 year": "A72", "unemployed": "A71"
    },
    "job": {
        "unskilled - resident": "A171", "unskilled - non-resident": "A172", 
        "skilled employee/ official": "A173", "management/ self-employed/highly qualified employee": "A174",
        "unemployed/ unskilled - non-resident": "A172"  # extra from data
    },
    "housing": {"own": "A151", "rent": "A152", "for free": "A153"},
    "credit_history": {
        "no credits taken/ all credits paid back duly": "A30",
        "all credits at this bank paid back duly": "A31",
        "existing credits paid back duly till now": "A32",
        "delay in paying off in the past": "A33",
        "critical account/ other credits existing (not at this bank)": "A34"
    },
    "collateral": {
        "real estate": "A121", "savings agreement/life insurance": "A122", "car": "A123", "none": "A124"
    },
    "other_installment_plans": {"none": "A141", "bank": "A142", "stores": "A143"},
    "telephone": {"none": "A191", "yes, registered under the customers name": "A192"},
    "is_foreign_worker": {"yes": "A201", "no": "A202"},
    "purpose": {
        "car (new)": "A40", "car (used)": "A45", "furniture/equipment": "A42",
        "radio/television": "A43", "domestic appliances": "A44", "repairs": "A46",
        "education": "A47", "retraining": "A48", "business": "A41", "others": "A410"
    },
    "status_and_sex": {  # gender + martial Status combined
        "male : single": "A91", "female : divorced/separated/married": "A93",
        "male : married/widowed": "A95", "female : single": "A94", 
        "male : divorced/separated": "A92"
    },
    "secondary_obligor": {"none": "none", "co-applicant": "co-applicant", "guarantor": "guarantor"}
}

# --- INPUTS ---
c1, c2 = st.columns(2)

with c1:
    status_account = st.selectbox("Checking Account", list(maps["status_account"].keys()), index=0)
    status_savings = st.selectbox("Savings Account", list(maps["status_savings"].keys()), index=0)
    credit_amount = st.number_input("Credit Amount ($)", 250, 20000, 5000)
    month_duration = st.number_input("Duration (months)", 4, 72, 12)
    purpose = st.selectbox("Purpose", list(maps["purpose"].keys()), index=3)
    payment_ratio = st.slider("Installment Rate", 1, 4, 2)

with c2:
    years_employment = st.selectbox("Employment", list(maps["years_employment"].keys()), index=2)
    job = st.selectbox("Job", list(maps["job"].keys()), index=2)
    age = st.number_input("Age", 18, 75, 30)
    housing = st.selectbox("Housing", list(maps["housing"].keys()), index=0)
    residence_since = st.slider("Years at Residence", 1, 4, 2)
    n_credits = st.slider("Existing Credits", 1, 4, 1)

c3, c4 = st.columns(2)

with c3:
    credit_history = st.selectbox("Credit History", list(maps["credit_history"].keys()), index=2)
    collateral = st.selectbox("Collateral", list(maps["collateral"].keys()), index=3)
    n_guarantors = st.slider("Guarantors", 1, 3, 1)

with c4:
    other_installment = st.selectbox("Other Plans", list(maps["other_installment_plans"].keys()), index=0)
    telephone = st.selectbox("Telephone", list(maps["telephone"].keys()), index=1)
    is_foreign_worker = st.selectbox("Foreign Worker", list(maps["is_foreign_worker"].keys()), index=0)

c5, c6 = st.columns(2)

with c5:
    status_and_sex = st.selectbox("Gender / Status", list(maps["status_and_sex"].keys()), index=0)

with c6:
    secondary_obligor = st.selectbox("Secondary Obligor", list(maps["secondary_obligor"].keys()), index=0)
    
# --- BUILD INPUT DATA ---
data = {
    "status_account": maps["status_account"][status_account],
    "month_duration": month_duration,
    "credit_history": maps["credit_history"][credit_history],
    "purpose": maps["purpose"][purpose],
    "credit_amount": credit_amount,
    "status_savings": maps["status_savings"][status_savings],
    "years_employment": maps["years_employment"][years_employment],
    "payment_to_income_ratio": payment_ratio,
    "gender": maps["status_and_sex"][status_and_sex],  # mapped from status_and_sex
    "n_guarantors": n_guarantors,
    "residence_since": residence_since,
    "collateral": maps["collateral"][collateral],
    "age": age,
    "other_installment_plans": maps["other_installment_plans"][other_installment],
    "housing": maps["housing"][housing],
    "n_credits": n_credits,
    "job": maps["job"][job],
    "martial Status": maps["status_and_sex"][status_and_sex],  # same as gender for encoding
    "telephone": maps["telephone"][telephone],
    "is_foreign_worker": maps["is_foreign_worker"][is_foreign_worker],
    "secondary_obligor": maps["secondary_obligor"][secondary_obligor]
}

# Ensure correct column order
input_df = pd.DataFrame([data])[features]

# --- PREDICTION ---
st.markdown("---")

if st.button("🔮 Predict Credit Risk", type="primary"):
    pred = model.predict(input_df)[0]
    
    st.subheader("Result")
    
    if pred == 1:
        st.error(f"⚠️ HIGH RISK probability)")
    else:
        st.success(f"✅ LOW RISK probability)")

    
    st.info("💡 This is a ML prediction - use as guidance only!")

