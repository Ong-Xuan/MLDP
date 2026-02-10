# streamlit_app.py
import pickle
import pandas as pd
import streamlit as st

# -----------------------------
# Page setup + light styling
# -----------------------------
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

st.markdown("""
<style>
/* Make the main container a bit narrower for a cleaner look */
.block-container { max-width: 900px; padding-top: 1.2rem; }

/* Subtle card */
.card {
  border: 1px solid rgba(250,250,250,0.10);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
}

/* Section title */
.section-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 0.35rem;
}

/* Small note text */
.small-note {
  font-size: 0.9rem;
  opacity: 0.85;
}

/* Result badge */
.badge-ok {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(76,175,80,0.15);
  border: 1px solid rgba(76,175,80,0.35);
}
.badge-risk {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(244,67,54,0.15);
  border: 1px solid rgba(244,67,54,0.35);
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🩺 Diabetes Risk Predictor")
st.markdown(
    "<div class='small-note'>A screening tool to estimate diabetes risk using health & lifestyle indicators. "
    "<b>Not a medical diagnosis.</b></div>",
    unsafe_allow_html=True
)

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
    "HeartDiseaseorAttack": "Coronary heart disease or heart attack history (1 = yes, 0 = no).",
    "PhysActivity": "Physical activity in past 30 days (1 = yes, 0 = no).",
    "Fruits": "Consume fruit 1+ times per day (1 = yes, 0 = no).",
    "Veggies": "Consume vegetables 1+ times per day (1 = yes, 0 = no).",
    "HvyAlcoholConsump": "Heavy alcohol consumption indicator (1 = yes, 0 = no).",
    "AnyHealthcare": "Has any healthcare coverage (1 = yes, 0 = no).",
    "NoDocbcCost": "Could not see doctor due to cost (1 = yes, 0 = no).",
    "GenHlth": "General health rating (1=Excellent ... 5=Poor).",
    "MentHlth": "Days of poor mental health in past 30 days (0–30).",
    "PhysHlth": "Days of poor physical health in past 30 days (0–30).",
    "DiffWalk": "Serious difficulty walking/climbing stairs (1 = yes, 0 = no).",
    "Sex": "Sex (dataset encoding). Often 1=Male, 2=Female (confirm with dataset codebook).",
    "Age": "Age category code (BRFSS grouped categories).",
    "Education": "Education level code (ordinal category).",
    "Income": "Income level code (ordinal category).",
}

DISPLAY_NAME = {
    "BMI": "Body Mass Index (BMI)",
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Checked (Last 5 Years)",
    "Smoker": "Smoker (100+ cigarettes lifetime)",
    "Stroke": "History of Stroke",
    "HeartDiseaseorAttack": "Heart Disease / Heart Attack History",
    "PhysActivity": "Physical Activity (Past 30 Days)",
    "Fruits": "Eats Fruits Daily",
    "Veggies": "Eats Vegetables Daily",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "AnyHealthcare": "Has Healthcare Coverage",
    "NoDocbcCost": "Could Not See Doctor Due to Cost",
    "GenHlth": "Overall Health Rating",
    "MentHlth": "Poor Mental Health Days (Past 30 Days)",
    "PhysHlth": "Poor Physical Health Days (Past 30 Days)",
    "DiffWalk": "Difficulty Walking / Climbing Stairs",
    "Sex": "Sex (1=Male, 2=Female)",
    "Age": "Age Group Code",
    "Education": "Education Level Code",
    "Income": "Income Level Code",
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

# -----------------------------
# Sidebar controls (professional)
# -----------------------------
st.sidebar.markdown("### Controls")
mode = st.sidebar.radio("Input mode", ["Simple (recommended)", "Advanced (all fields)"])
show_debug = st.sidebar.toggle("Show technical details", value=False)
st.sidebar.markdown("---")
if st.sidebar.button("↩️ Reset inputs", use_container_width=True):
    reset_state()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Tips")
st.sidebar.caption("• Simple mode is best for demos.\n• Advanced mode edits all 21 indicators.\n• Recall-focused model: aims to reduce false negatives.")

# -----------------------------
# UI helpers
# -----------------------------
def binary_select(col):
    current = st.session_state.get(col, 0)
    try:
        current = int(current)
    except:
        current = 0
    current = 1 if current == 1 else 0

    label = DISPLAY_NAME.get(col, col)

    st.selectbox(
        label,
        options=[0, 1],
        index=current,
        format_func=lambda v: "No (0)" if v == 0 else "Yes (1)",
        key=col,  # IMPORTANT: keep key as the original column name
        help=HELP.get(col, "")
    )

def num_input(col, step=1.0, min_value=None, max_value=None, fmt=None):
    label = DISPLAY_NAME.get(col, col)

    kwargs = dict(
        label=label,
        value=float(st.session_state.get(col, DEFAULTS.get(col, 0))),
        step=float(step),
        key=col,  # keep key
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
    label = DISPLAY_NAME.get(col, col)

    st.slider(
        label,
        min_v,
        max_v,
        int(st.session_state.get(col, DEFAULTS.get(col, min_v))),
        key=col,  # keep key
        help=HELP.get(col, "")
    )


def make_row_from_state():
    user_inputs = {col: st.session_state.get(col, DEFAULTS.get(col, 0)) for col in COLUMNS}
    return pd.DataFrame([user_inputs]).reindex(columns=COLUMNS, fill_value=0)

def predict(row: pd.DataFrame):
    pred = int(model.predict(row)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(row)[0][1])
    return pred, prob

# -----------------------------
# Inputs section (card)
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Enter health indicators</div>", unsafe_allow_html=True)
st.caption("Fill in the inputs below, then click **Predict**. Use **Reset inputs** in the sidebar to start over.")

if mode == "Simple (recommended)":
    with st.form("simple_form"):
        colL, colR = st.columns(2)

        with colL:
            if "BMI" in COLUMNS:
                num_input("BMI", step=0.1, min_value=0.0, fmt="%.1f")
            for c in ["HighBP", "HighChol", "Smoker", "PhysActivity"]:
                if c in COLUMNS:
                    binary_select(c)

        with colR:
            if "GenHlth" in COLUMNS:
                slider_input("GenHlth", 1, 5)
            if "MentHlth" in COLUMNS:
                num_input("MentHlth", step=1, min_value=0, max_value=30, fmt="%.0f")
            if "PhysHlth" in COLUMNS:
                num_input("PhysHlth", step=1, min_value=0, max_value=30, fmt="%.0f")
            if "DiffWalk" in COLUMNS:
                binary_select("DiffWalk")

        st.markdown("**Optional demographics (improves accuracy):**")
        d1, d2 = st.columns(2)
        with d1:
            for c in ["Age", "Sex"]:
                if c in COLUMNS:
                    num_input(c, step=1, min_value=0, fmt="%.0f")
        with d2:
            for c in ["Education", "Income"]:
                if c in COLUMNS:
                    num_input(c, step=1, min_value=0, fmt="%.0f")

        submitted = st.form_submit_button("Predict", use_container_width=True)

else:
    with st.form("advanced_form"):
        with st.expander("Vitals & Conditions", expanded=True):
            if "BMI" in COLUMNS:
                num_input("BMI", step=0.1, min_value=0.0, fmt="%.1f")
            for c in ["HighBP", "HighChol", "CholCheck", "Stroke", "HeartDiseaseorAttack"]:
                if c in COLUMNS:
                    binary_select(c) if c in BINARY_FIELDS else num_input(c)

        with st.expander("Lifestyle", expanded=False):
            for c in ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"]:
                if c in COLUMNS:
                    binary_select(c)

        with st.expander("Healthcare Access", expanded=False):
            for c in ["AnyHealthcare", "NoDocbcCost"]:
                if c in COLUMNS:
                    binary_select(c)

        with st.expander("General Health & Demographics", expanded=False):
            for c in ["GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]:
                if c in COLUMNS:
                    if c in BINARY_FIELDS:
                        binary_select(c)
                    elif c == "GenHlth":
                        slider_input("GenHlth", 1, 5)
                    elif c in ["MentHlth", "PhysHlth"]:
                        num_input(c, step=1, min_value=0, max_value=30, fmt="%.0f")
                    else:
                        num_input(c, step=1, min_value=0, fmt="%.0f")

        submitted = st.form_submit_button("Predict", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Results section
# -----------------------------
if submitted:
    row = make_row_from_state()
    pred, prob = predict(row)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)

    if pred == 1:
        st.markdown("<span class='badge-risk'>At Risk (1)</span>", unsafe_allow_html=True)
        st.write("The model predicts the user may be at risk of diabetes. Consider further screening and professional advice.")
    else:
        st.markdown("<span class='badge-ok'>No Diabetes Risk (0)</span>", unsafe_allow_html=True)
        st.write("The model predicts lower risk based on the provided indicators.")

    if prob is not None:
        st.markdown("**Estimated probability of being at risk:**")
        st.progress(min(max(prob, 0.0), 1.0))
        st.write(f"**{prob:.1%}**")

    st.caption("Reminder: This is a screening prediction and does not replace medical diagnosis.")

    if show_debug:
        st.markdown("---")
        st.markdown("**Technical details (for marking):**")
        st.write("Model input row (first 5 columns shown):")
        st.dataframe(row.iloc[:, :5])
        st.write("All input columns used by the model:")
        st.code(", ".join(COLUMNS))

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# About section (marks-friendly)
# -----------------------------
with st.expander("About this app (for report/marking)", expanded=False):
    st.markdown("""
- **Goal:** Predict diabetes risk (0/1) from BRFSS health indicators for early screening support.
- **Why Recall:** Missing an at-risk individual (false negative) is more costly than flagging extra cases.
- **Model input:** Uses the same feature columns saved during training to avoid mismatched inputs.
- **Disclaimer:** Educational screening tool, not a diagnosis.
""")
