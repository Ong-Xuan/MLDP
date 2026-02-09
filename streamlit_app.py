# streamlit_app.py
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Predictor")
st.write("This app predicts whether an individual is **at risk of diabetes** based on health indicators.")

# -----------------------------
# Load model package
# -----------------------------
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as f:
        pkg = pickle.load(f)
    return pkg

pkg = load_model()
model = pkg["model"]
COLUMNS = pkg["columns"]   # expected input feature names in correct order
TARGET = pkg.get("target", "Diabetes_binary")

st.caption(f"Model loaded. Target: `{TARGET}`")

# -----------------------------
# Build input form (auto from columns)
# -----------------------------
st.subheader("Enter health indicators")

with st.form("input_form"):
    user_inputs = {}

    # Make the UI nicer: BMI first, then the rest
    cols_order = COLUMNS.copy()
    if "BMI" in cols_order:
        cols_order.remove("BMI")
        cols_order = ["BMI"] + cols_order

    for col in cols_order:
        # Most BRFSS fields are numeric (0/1 or small integers). We'll use number_input.
        # If you want sliders later, we can customize per feature.
        default_val = 0.0
        step = 1.0

        # BMI tends to be continuous-ish
        if col == "BMI":
            default_val = 25.0
            step = 0.1

        user_inputs[col] = st.number_input(
            label=col,
            value=float(default_val),
            step=float(step),
            format="%.2f" if col == "BMI" else "%.0f"
        )

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Predict
# -----------------------------
if submitted:
    # Create 1-row dataframe, enforce same columns/order
    row = pd.DataFrame([user_inputs])
    row = row.reindex(columns=COLUMNS, fill_value=0)

    pred = int(model.predict(row)[0])

    # Probability if available
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(row)[0][1])

    st.subheader("Result")
    if pred == 1:
        st.error("Prediction: **At Risk (1)**")
    else:
        st.success("Prediction: **No Diabetes Risk (0)**")

    if prob is not None:
        st.write(f"Estimated probability of being at risk: **{prob:.2%}**")

    st.caption("Note: This tool supports screening/education and is not a medical diagnosis.")
