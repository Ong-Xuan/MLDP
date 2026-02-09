# streamlit_app.py
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.write("Predicts diabetes risk (screening only, not a medical diagnosis).")

# -----------------------------
# Load model package
# -----------------------------
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", "rb") as f:
        return pickle.load(f)

pkg = load_model()
model = pkg["model"]
COLUMNS = pkg["columns"]
TARGET = pkg.get("target", "Diabetes_binary")

# -----------------------------
# Field metadata (BRFSS-style)
# -----------------------------
BINARY_FIELDS = {
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk"
}

HELP = {
    "BMI": "Body Mass Index. Higher BMI is often associated with higher diabetes risk.",
    "HighBP": "High blood pressure (1 = yes, 0 = no).",
    "HighChol": "High cholesterol (1 = yes, 0 = no).",
    "CholCheck": "Had cholesterol check in the past 5 years (1 = yes, 0 = no).",
    "Smoker": "Smoked at least 100 cigarettes in lifetime (1 = yes, 0 = no).",
    "Stroke": "Ever told you had a stroke (1 = yes, 0 = no).",
    "HeartDiseaseorAttack": "Coronary heart disease or myocardial infarction (1 = yes, 0 = no).",
    "PhysActivity": "Physical activity in past 30 days (1 = yes, 0 = no).",
    "Fruits": "Consume fruit 1+ times per day (1 = yes, 0 = no).",
    "Veggies": "Consume vegetables 1+ times per day (1 = yes, 0 = no).",
    "HvyAlcoholConsump": "Heavy alcohol consumption indicator (1 = yes, 0 = no).",
    "AnyHealthcare": "Have any healthcare coverage (1 = yes, 0 = no).",
    "NoDocbcCost": "Could not see doctor due to cost (1 = yes, 0 = no).",
    "GenHlth": "General health rating (1=Excellent ... 5=Poor).",
    "MentHlth": "Days of poor mental health in past 30 days (0–30).",
    "PhysHlth": "Days of poor physical health in past 30 days (0–30).",
    "DiffWalk": "Serious difficulty walking or climbing stairs (1 = yes, 0 = no).",
    "Sex": "Sex (dataset encoding). Commonly 1=Male, 2=Female (confirm with dataset codebook).",
    "Age": "Age category code (BRFSS uses grouped age categories).",
    "Education": "Education level code (ordinal category).",
    "Income": "Income level code (ordinal category).",
}

# -----------------------------
# Defaults (safe starter values)
# -----------------------------
DEFAULTS = {col: 0 for col in COLUMNS}
if "BMI" in DEFAULTS: DEFAULTS["BMI"] = 25.0
if "GenHlth" in DEFAULTS: DEFAULTS["GenHlth"] = 3
if "MentHlth" in DEFAULTS: DEFAULTS["MentHlth"] = 0
if "PhysHlth" in DEFAULTS: DEFAULTS["PhysHlth"] = 0
if "Age" in DEFAULTS: DEFAULTS["Age"] = 8
if "Sex" in DEFAULTS: DEFAULTS["Sex"] = 1
if "Education" in DEFAULTS: DEFAULTS["Education"] = 4
if "Income" in DEFAULTS: DEFAULTS["Income"] = 5

# -----------------------------
# Session state init + Reset
# -----------------------------
def init_state():
    for col in COLUMNS:
        if col not in st.session_state:
            st.session_state[col] = DEFAULTS.get(col, 0)

def reset_state():
    for col in COLUMNS:
        st.session_state[col] = DEFAULTS.get(col, 0)

init_state()

# Sidebar options + Reset button
st.sidebar.header("Options")
mode = st.sidebar.radio("Input mode", ["Simple (recommended)", "Advanced (all fields)"])
if st.sidebar.button("Reset inputs"):
    reset_state()
    st.rerun()

# -----------------------------
# UI helpers
# -----------------------------
def binary_select(col):
    """
    Stores numeric 0/1 in session_state (prevents crashes like int("No (0)")).
    Displays friendly labels using format_func.
    """
    current = st.session_state.get(col, 0)
    try:
        current = int(current)
    except:
        current = 0
    current = 1 if current == 1 else 0

    st.selectbox(
        col,
        options=[0, 1],
        index=current,
        format_func=lambda v: "No (0)" if v == 0 else "Yes (1)",
        key=col,
        help=HELP.get(col, "")
    )

def num_input(col, step=1.0, min_value=None, max_value=None, fmt=None):
    kwargs = dict(
        label=col,
        value=float(st.session_state.get(col, DEFAULTS.get(col, 0))),
        step=float(step),
        key=col,
        help=HELP.get(col, "")
    )
    if min_value is not None:
        kwargs["min_value"] = float(min_value)
    if max_value is not None:
        kwargs["max_value"] = float(max_value)
    if fmt is not None:
        kwargs["format"] = fmt
    st.number_input(**kwargs)

def slider_input(col, min_v, max_v):
    st.slider(
        col,
        min_v,
        max_v,
        int(st.session_state.get(col, DEFAULTS.get(col, min_v))),
        key=col,
        help=HELP.get(col, "")
    )

def make_row_from_state():
    """
    Build a 1-row dataframe with EXACT expected columns/order.
    Binary fields are already numeric because binary_select stores 0/1.
    """
    user_inputs = {col: st.session_state.get(col, DEFAULTS.get(col, 0)) for col in COLUMNS}
    row = pd.DataFrame([user_inputs]).reindex(columns=COLUMNS, fill_value=0)
    return row

def predict_and_show(row: pd.DataFrame):
    pred = int(model.predict(row)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(row)[0][1])

    st.subheader("Result")
    if pred == 1:
        st.error("Prediction: **At Risk (1)**")
    else:
        st.success("Prediction: **No Diabetes Risk (0)**")

    if prob is not None:
        st.write(f"Estimated probability of being at risk: **{prob:.1%}**")

    st.caption("Note: This tool supports screening/education and is not a medical diagnosis.")

# -----------------------------
# Input UI
# -----------------------------
st.subheader("Enter health indicators")

if mode == "Simple (recommended)":
    st.info("Simple mode uses key inputs. Advanced mode allows editing all 21 indicators.")

    with st.form("simple_form"):
        c1, c2 = st.columns(2)

        # Left column: common risk factors
        with c1:
            if "BMI" in COLUMNS:
                num_input("BMI", step=0.1, min_value=0.0, fmt="%.1f")
            for col in ["HighBP", "HighChol", "Smoker", "PhysActivity"]:
                if col in COLUMNS:
                    binary_select(col)

        # Right column: health status
        with c2:
            if "GenHlth" in COLUMNS:
                slider_input("GenHlth", 1, 5)
            if "MentHlth" in COLUMNS:
                num_input("MentHlth", step=1.0, min_value=0.0, max_value=30.0, fmt="%.0f")
            if "PhysHlth" in COLUMNS:
                num_input("PhysHlth", step=1.0, min_value=0.0, max_value=30.0, fmt="%.0f")
            if "DiffWalk" in COLUMNS:
                binary_select("DiffWalk")

        # Optional demographics
        st.markdown("**Optional demographics (improves accuracy):**")
        d1, d2 = st.columns(2)
        with d1:
            if "Age" in COLUMNS:
                num_input("Age", step=1.0, min_value=1.0, fmt="%.0f")
            if "Sex" in COLUMNS:
                num_input("Sex", step=1.0, min_value=0.0, fmt="%.0f")
        with d2:
            if "Education" in COLUMNS:
                num_input("Education", step=1.0, min_value=0.0, fmt="%.0f")
            if "Income" in COLUMNS:
                num_input("Income", step=1.0, min_value=0.0, fmt="%.0f")

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = make_row_from_state()
        predict_and_show(row)

else:
    st.warning("Advanced mode shows all fields. Binary fields use Yes/No dropdowns.")

    with st.form("advanced_form"):
        with st.expander("Vitals & Conditions", expanded=True):
            if "BMI" in COLUMNS:
                num_input("BMI", step=0.1, min_value=0.0, fmt="%.1f")
            for col in ["HighBP", "HighChol", "CholCheck", "Stroke", "HeartDiseaseorAttack"]:
                if col in COLUMNS:
                    if col in BINARY_FIELDS:
                        binary_select(col)
                    else:
                        num_input(col)

        with st.expander("Lifestyle", expanded=False):
            for col in ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"]:
                if col in COLUMNS:
                    binary_select(col)

        with st.expander("Healthcare Access", expanded=False):
            for col in ["AnyHealthcare", "NoDocbcCost"]:
                if col in COLUMNS:
                    binary_select(col)

        with st.expander("General Health & Demographics", expanded=False):
            for col in ["GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]:
                if col in COLUMNS:
                    if col in BINARY_FIELDS:
                        binary_select(col)
                    elif col == "GenHlth":
                        slider_input("GenHlth", 1, 5)
                    elif col in ["MentHlth", "PhysHlth"]:
                        num_input(col, step=1.0, min_value=0.0, max_value=30.0, fmt="%.0f")
                    else:
                        num_input(col, step=1.0, min_value=0.0, fmt="%.0f")

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = make_row_from_state()
        predict_and_show(row)
