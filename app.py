import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

body{
    background-color:#F4F8FB;
}

.main{
    background-color:#F4F8FB;
}

h1{
    color:#0B5394;
    text-align:center;
}

h3{
    color:#1565C0;
}

.stButton>button{
    width:100%;
    height:55px;
    background:#1565C0;
    color:white;
    font-size:20px;
    border-radius:10px;
    border:none;
}

.stButton>button:hover{
    background:#0B5394;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:10px;
    padding:15px;
    box-shadow:2px 2px 10px lightgray;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
# ❤️ Heart Disease Prediction System

### AI-Powered Cardiovascular Risk Assessment
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("❤️ Heart Disease")

st.sidebar.markdown("---")

st.sidebar.info("""
This application predicts whether a patient has Heart Disease using a Machine Learning model.

Algorithm Used:
• Random Forest

Accuracy:
• 83%

Dataset:
• UCI Heart Disease Dataset
""")

st.sidebar.markdown("---")

st.sidebar.success("Developed using Streamlit")

# -----------------------------
# Dashboard Cards
# -----------------------------
col1,col2,col3=st.columns(3)

with col1:
    st.metric(
        label="Model Accuracy",
        value="83%"
    )

with col2:
    st.metric(
        label="Algorithm",
        value="Random Forest"
    )

with col3:
    st.metric(
        label="Dataset",
        value="Heart Disease"
    )

st.markdown("---")

st.subheader("👨‍⚕️ Patient Information")

left,right=st.columns(2)
with left:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex = st.selectbox(
        "Gender",
        ("Female", "Male")
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=600,
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0,1]
    )

with right:

    restecg = st.selectbox(
        "Resting ECG",
        [0,1,2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0,1]
    )

    oldpeak = st.number_input(
        "Old Peak",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )

    slope = st.selectbox(
        "Slope",
        [0,1,2]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0,1,2,3,4]
    )

    thal = st.selectbox(
        "Thal",
        [0,1,2,3]
    )

sex_value = 1 if sex == "Male" else 0

st.markdown("<br>", unsafe_allow_html=True)

predict = st.button("🔍 Analyze Patient")
if predict:

    # Convert gender to numeric
    sex_value = 1 if sex == "Male" else 0

    # Create input array
    data = np.array([[
        age,
        sex_value,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    # Scale the input
    data = scaler.transform(data)

    # Make prediction
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    confidence = np.max(probability) * 100

    st.markdown("---")

    st.subheader("📋 Prediction Result")

    if prediction[0] == 1:

        st.error("❤️ **Heart Disease Detected**")

        st.warning(
            "Please consult a cardiologist for further medical evaluation."
        )

    else:

        st.success("💚 **No Heart Disease Detected**")

        st.info(
            "The patient appears to have a low risk of heart disease."
        )

    # -----------------------------
    # Confidence Bar
    # -----------------------------

    st.write("### Prediction Confidence")

    st.progress(int(confidence))

    st.write(f"**{confidence:.2f}%**")

    # -----------------------------
    # Probability
    # -----------------------------

    st.write("### Prediction Probability")

    st.write(
        f"🟢 No Heart Disease : {probability[0][0]*100:.2f}%"
    )

    st.write(
        f"🔴 Heart Disease : {probability[0][1]*100:.2f}%"
    )

    st.markdown("---")

    # -----------------------------
    # Patient Summary
    # -----------------------------

    st.subheader("👨‍⚕️ Patient Summary")

    c1, c2 = st.columns(2)

    with c1:

        st.write("**Age:**", age)
        st.write("**Gender:**", sex)
        st.write("**Chest Pain Type:**", cp)
        st.write("**Blood Pressure:**", trestbps)
        st.write("**Cholesterol:**", chol)
        st.write("**Blood Sugar:**", fbs)

    with c2:

        st.write("**Rest ECG:**", restecg)
        st.write("**Heart Rate:**", thalach)
        st.write("**Exercise Angina:**", exang)
        st.write("**Old Peak:**", oldpeak)
        st.write("**Slope:**", slope)
        st.write("**Major Vessels:**", ca)
        st.write("**Thal:**", thal)

    st.markdown("---")

    # -----------------------------
    # Model Information
    # -----------------------------

    st.subheader("🤖 Model Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "83%")
    col2.metric("Algorithm", "Random Forest")
    col3.metric("Features", "13")

    st.markdown("---")

    # -----------------------------
    # Footer
    # -----------------------------

    st.markdown(
        """
        <center>

        <h4>❤️ Heart Disease Prediction System</h4>

        Developed using <b>Python, Scikit-Learn and Streamlit</b>

        <br>

        Machine Learning Project

        </center>
        """,
        unsafe_allow_html=True
    )