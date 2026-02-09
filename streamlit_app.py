# streamlit_app.py
import pickle
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Risk Predictor")
st.write(
    "This tool estimates **diabetes risk** based on health and lifestyle factors. "
    "It is intended for **screening and educational purposes only** and is **not a medical diagnosis**."
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as f:
        return pickle.load(f)

pkg = load_model()
model = pkg["model"]
COLUMNS = pkg["columns"]
TARGET = pkg.get("target", "Diabetes_binary")

# --------------------------------------------------
# Binary fields (Yes / No)
# --------------------------------------------------
BINARY_FIELDS = {
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"
}

YESNO = {"No": 0, "Yes": 1}

# --------------------------------------------------
# User-friendly explanations
# --------------------------------------------------
HELP = {
    "BMI": "Body Mass Index. A measure of body fat based on height and weight.",
    "HighBP": "Have you ever been told by a doctor that you have high blood pressure?",
    "HighChol": "Have you ever been told by a doctor that you have high cholesterol?",
    "CholCheck": "Have you had your cholesterol checked by a doctor in the past 5 years?",
    "Smoker": "Have you smoked at least 100 cigarettes in your lifetime?",
    "Stroke": "Have you ever been told by a doctor that you had a stroke?",
    "HeartDiseaseorAttack": "Have you ever been diagnosed with heart disease or had a heart attack?",
    "PhysActivity": "Have you done any physical activity or exercise in the past 30 days?",
    "Fruits": "Do you usually eat fruit one or more times per day?",
    "Veggies": "Do you usually eat vegetables one or more times per day?",
    "HvyAlcoholConsump": "Do you consume alcohol heavily?",
    "AnyHealthcare": "Do you currently have any form of health insurance or healthcare coverage?",
    "NoDocbcCost": "In the past year, was there a time you needed to see a doctor but could not due to cost?",
    "GenHlth": (
        "How would you rate your general health?\n"
        "1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor"
    ),
    "MentHlth": "Number of days in the past 30 days when your mental health was not good.",
    "PhysHlth": "Number of days in the past 30 days when your physical health was not good.",
    "DiffWalk": "Do you have serious difficulty walking or climbing stairs?",
    "Sex": "Sex assigned at birth.",
    "Age": "Age group category (higher number indicates older age group).",
    "Education": "Highest level of education completed.",
    "Income": "Household income level category."
}

# --------------------------------------------------
# Default values
# --------------------------------------------------
DEFAULTS = {col: 0 for col in COLUMNS}
DEFAULTS.update({
    "BMI": 25.0,
    "GenHlth": 3,
    "MentHlth": 0,
    "PhysHlth": 0,
    "Age": 8,
    "Sex": 1,
    "Education": 4,
    "Income": 5
})

# --------------------------------------------------
# Session state handling
# --------------------------------------------------
def init_state():
    for col in COLUMNS:
        if col not in st.session_state:
            st.session_state[col] = DEFAULTS.get(col, 0)

def reset_state():
    for col in COLUMNS:
        st.session_state[col] = DEFAULTS.get(col, 0)

init_state()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Options")
mode = st.sidebar.radio("Input mode", ["Simple", "Advanced"])
if st.sidebar.button("Reset all inputs"):
    reset_state()
    st.rerun()

# --------------------------------------------------
# Input helpers
# --------------------------------------------------
def binary_select(col):
    st.selectbox(
        col,
        ["No", "Yes"],
        index=st.session_state[col],
        key=col,
        help=HELP.get(col)
    )

def num_input(col, step=1.0, min_v=None, max_v=None):
    st.number_input(
        col,
        value=float(st.session_state[col]),
        step=step,
        min_value=min_v,
        max_value=max_v,
        key=col,
        help=HELP.get(col)
    )

def slider_input(col, min_v, max_v):
    st.slider(
        col,
        min_v,
        max_v,
        int(st.session_state[col]),
        key=col,
        help=HELP.get(col)
    )

def build_row():
    data = {}
    for col in COLUMNS:
        if col in BINARY_FIELDS:
            data[col] = YESNO.get(st.session_state[col], 0)
        else:
            data[col] = st.session_state[col]

    return pd.DataFrame([data]).reindex(columns=COLUMNS, fill_value=0)

def predict_and_display(row):
    pred = int(model.predict(row)[0])
    prob = None

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(row)[0][1]

    st.subheader("Prediction Result")

    if pred == 1:
        st.error("⚠️ **At Risk of Diabetes**")
    else:
        st.success("✅ **No Diabetes Risk Detected**")

    if prob is not None:
        st.write(f"Estimated risk probability: **{prob:.1%}**")

    st.caption("This result is for educational screening only and not a medical diagnosis.")

# --------------------------------------------------
# UI
# --------------------------------------------------
st.subheader("Enter your health information")

if mode == "Simple":
    st.info("Simple mode uses key health indicators. Advanced mode allows full input.")

    with st.form("simple_form"):
        c1, c2 = st.columns(2)

        with c1:
            num_input("BMI", step=0.1, min_v=0.0)
            binary_select("HighBP")
            binary_select("HighChol")
            binary_select("Smoker")
            binary_select("PhysActivity")

        with c2:
            slider_input("GenHlth", 1, 5)
            st.caption("1 = Excellent • 2 = Very good • 3 = Good • 4 = Fair • 5 = Poor")

            num_input("MentHlth", min_v=0.0, max_v=30.0)
            num_input("PhysHlth", min_v=0.0, max_v=30.0)
            binary_select("DiffWalk")

        st.markdown("**Optional demographic information (improves accuracy):**")
        d1, d2 = st.columns(2)
        with d1:
            num_input("Age", min_v=1)
            num_input("Sex", min_v=0)
        with d2:
            num_input("Education", min_v=0)
            num_input("Income", min_v=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = build_row()
        predict_and_display(row)

else:
    st.warning("Advanced mode exposes all model features.")

    with st.form("advanced_form"):
        for col in COLUMNS:
            if col in BINARY_FIELDS:
                binary_select(col)
            elif col == "GenHlth":
                slider_input(col, 1, 5)
            elif col in ["MentHlth", "PhysHlth"]:
                num_input(col, min_v=0.0, max_v=30.0)
            elif col == "BMI":
                num_input(col, step=0.1, min_v=0.0)
            else:
                num_input(col)

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = build_row()
        predict_and_display(row)
