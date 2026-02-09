# streamlit_app.py
import pickle
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Predictor")
st.write(
    "This tool estimates **diabetes risk** using health and lifestyle indicators. "
    "It is for **screening/education only** and is **not a medical diagnosis**."
)

# --------------------------------------------------
# Load model package
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
# Field groups
# --------------------------------------------------
BINARY_FIELDS = {
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk"
}

# Treat these as integer-coded categories in BRFSS
INT_FIELDS = {"Age", "Sex", "Education", "Income", "GenHlth"}

# Treat these as continuous-ish numeric fields
FLOAT_FIELDS = {"BMI"}

# These are counts (0–30) but easiest as int
DAY_FIELDS = {"MentHlth", "PhysHlth"}

YESNO = {"No": 0, "Yes": 1}

# --------------------------------------------------
# User-friendly explanations
# --------------------------------------------------
HELP = {
    "BMI": "Body Mass Index. A measure of body fat based on height and weight.",
    "HighBP": "Have you ever been told you have high blood pressure?",
    "HighChol": "Have you ever been told you have high cholesterol?",
    "CholCheck": "Had your cholesterol checked in the past 5 years?",
    "Smoker": "Have you smoked at least 100 cigarettes in your lifetime?",
    "Stroke": "Have you ever been told you had a stroke?",
    "HeartDiseaseorAttack": "Have you ever had heart disease or a heart attack?",
    "PhysActivity": "Any physical activity/exercise in the past 30 days?",
    "Fruits": "Do you usually eat fruit 1+ times per day?",
    "Veggies": "Do you usually eat vegetables 1+ times per day?",
    "HvyAlcoholConsump": "Do you consume alcohol heavily?",
    "AnyHealthcare": "Do you have any healthcare coverage/insurance?",
    "NoDocbcCost": "In the past year, needed a doctor but couldn’t go due to cost?",
    "GenHlth": (
        "General health rating:\n"
        "1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor"
    ),
    "MentHlth": "Days (0–30) in the past month when mental health was not good.",
    "PhysHlth": "Days (0–30) in the past month when physical health was not good.",
    "DiffWalk": "Serious difficulty walking or climbing stairs?",
    "Sex": "Sex assigned at birth (dataset encoding).",
    "Age": "Age category code (higher number = older age group).",
    "Education": "Education level category code.",
    "Income": "Income level category code."
}

# --------------------------------------------------
# Defaults (IMPORTANT: keep consistent types)
# --------------------------------------------------
DEFAULTS = {col: 0 for col in COLUMNS}

# floats
if "BMI" in DEFAULTS:
    DEFAULTS["BMI"] = 25.0

# ints
for k, v in {
    "GenHlth": 3,
    "MentHlth": 0,
    "PhysHlth": 0,
    "Age": 8,
    "Sex": 1,
    "Education": 4,
    "Income": 5
}.items():
    if k in DEFAULTS:
        DEFAULTS[k] = int(v)

# binary defaults should be "No" label in session state for selectbox
for b in BINARY_FIELDS:
    if b in DEFAULTS:
        DEFAULTS[b] = "No"

# --------------------------------------------------
# Session state init + reset
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
# Input helpers (NO mixed numeric types!)
# --------------------------------------------------
def binary_select(col):
    st.selectbox(
        col,
        ["No", "Yes"],
        key=col,
        help=HELP.get(col, "")
    )

def float_input(col, step=0.1, min_v=0.0, max_v=None):
    # force FLOATS everywhere
    st.number_input(
        col,
        value=float(st.session_state.get(col, 0.0)),
        step=float(step),
        min_value=float(min_v),
        max_value=float(max_v) if max_v is not None else None,
        key=col,
        help=HELP.get(col, "")
    )

def int_input(col, step=1, min_v=0, max_v=None):
    # force INTS everywhere
    st.number_input(
        col,
        value=int(st.session_state.get(col, 0)),
        step=int(step),
        min_value=int(min_v),
        max_value=int(max_v) if max_v is not None else None,
        key=col,
        help=HELP.get(col, "")
    )

def genhlth_slider(col="GenHlth"):
    st.slider(
        col,
        min_value=1,
        max_value=5,
        value=int(st.session_state.get(col, 3)),
        key=col,
        help=HELP.get(col, "")
    )
    st.caption("1 = Excellent • 2 = Very good • 3 = Good • 4 = Fair • 5 = Poor")

def build_row():
    data = {}
    for col in COLUMNS:
        if col in BINARY_FIELDS:
            data[col] = YESNO.get(st.session_state.get(col, "No"), 0)
        elif col in FLOAT_FIELDS:
            data[col] = float(st.session_state.get(col, 0.0))
        elif col in DAY_FIELDS:
            data[col] = int(st.session_state.get(col, 0))
        elif col in INT_FIELDS:
            data[col] = int(st.session_state.get(col, 0))
        else:
            # fallback
            data[col] = float(st.session_state.get(col, 0))
    return pd.DataFrame([data]).reindex(columns=COLUMNS, fill_value=0)

def predict_and_display(row):
    pred = int(model.predict(row)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(row)[0][1])

    st.subheader("Prediction Result")
    if pred == 1:
        st.error("⚠️ **At Risk of Diabetes (1)**")
    else:
        st.success("✅ **No Diabetes Risk Detected (0)**")

    if prob is not None:
        st.write(f"Estimated probability of being at risk: **{prob:.1%}**")
        st.progress(min(max(prob, 0.0), 1.0))

    st.caption("Educational screening only — not a diagnosis.")

# --------------------------------------------------
# UI
# --------------------------------------------------
st.subheader("Enter your health information")

if mode == "Simple":
    st.info("Simple mode uses key indicators. Advanced mode allows full input.")

    with st.form("simple_form"):
        c1, c2 = st.columns(2)

        with c1:
            if "BMI" in COLUMNS:
                float_input("BMI", step=0.1, min_v=0.0)
            for col in ["HighBP", "HighChol", "Smoker", "PhysActivity"]:
                if col in COLUMNS:
                    binary_select(col)

        with c2:
            if "GenHlth" in COLUMNS:
                genhlth_slider("GenHlth")
            if "MentHlth" in COLUMNS:
                int_input("MentHlth", min_v=0, max_v=30)
                st.caption("0 = none, 30 = every day")
            if "PhysHlth" in COLUMNS:
                int_input("PhysHlth", min_v=0, max_v=30)
                st.caption("0 = none, 30 = every day")
            if "DiffWalk" in COLUMNS:
                binary_select("DiffWalk")

        st.markdown("**Optional demographics (can improve accuracy):**")
        d1, d2 = st.columns(2)
        with d1:
            if "Age" in COLUMNS:
                int_input("Age", min_v=1)
            if "Sex" in COLUMNS:
                int_input("Sex", min_v=0)
        with d2:
            if "Education" in COLUMNS:
                int_input("Education", min_v=0)
            if "Income" in COLUMNS:
                int_input("Income", min_v=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = build_row()
        predict_and_display(row)

else:
    st.warning("Advanced mode allows editing all model fields.")

    with st.form("advanced_form"):
        for col in COLUMNS:
            if col in BINARY_FIELDS:
                binary_select(col)
            elif col == "BMI":
                float_input(col, step=0.1, min_v=0.0)
            elif col == "GenHlth":
                genhlth_slider(col)
            elif col in DAY_FIELDS:
                int_input(col, min_v=0, max_v=30)
            elif col in INT_FIELDS:
                int_input(col, min_v=0)
            else:
                # fallback: float
                float_input(col, step=1.0, min_v=0.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = build_row()
        predict_and_display(row)
