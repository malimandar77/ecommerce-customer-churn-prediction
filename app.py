import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"
FEATURE_PATH = BASE_DIR / "feature_names.joblib"

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Load saved model and feature names
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    return model, feature_names

model, feature_names = load_model()

# Application title
st.title("🛒 E-Commerce Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to stop purchasing "
    "based on their historical purchasing behavior."
)

st.info(
    "This application uses a Random Forest classifier trained "
    "on the UCI Online Retail dataset."
)

st.divider()

# Input form
st.subheader("Enter Customer Information")

with st.form("customer_input_form"):

    recency = st.number_input(
        "Recency (days since last purchase)",
        min_value=0,
        max_value=1000,
        value=30
    )

    frequency = st.number_input(
        "Frequency (number of orders)",
        min_value=1,
        max_value=10000,
        value=5
    )

    monetary = st.number_input(
        "Monetary (total historical spending)",
        min_value=0.01,
        max_value=10000000.0,
        value=1000.0,
        step=100.0
    )

    total_quantity = st.number_input(
        "Total Quantity (items purchased)",
        min_value=1,
        max_value=1000000,
        value=100
    )

    unique_products = st.number_input(
        "Unique Products",
        min_value=1,
        max_value=100000,
        value=10
    )

    active_days = st.number_input(
        "Active Days",
        min_value=1,
        max_value=365,
        value=5
    )

    submitted = st.form_submit_button(
        "Predict Customer Churn"
    )

# Prediction
if submitted:

    input_data = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "TotalQuantity": total_quantity,
        "UniqueProducts": unique_products,
        "ActiveDays": active_days
    }])

    # Ensure feature order matches model training
    input_data = input_data[feature_names]

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    st.divider()
    st.subheader("Prediction Results")

    st.metric(
        "Estimated Churn Probability",
        f"{probability:.1%}"
    )

    st.progress(probability)

    if prediction == 1:
        st.error(
            "⚠️ Prediction: Customer is classified as churned "
            "or at risk of churn."
        )
    else:
        st.success(
            "✅ Prediction: Customer is classified as retained."
        )

    st.caption(
        "This is a machine-learning prediction, not a guarantee "
        "of future customer behavior. The model was trained using "
        "historical retail transactions."
    )

st.divider()

st.caption(
    "Academic Mini Project | Machine Learning | Random Forest"
)

