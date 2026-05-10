# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Aircraft Engine Risk Prediction",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================

model = pickle.load(open("model.pkl", "rb"))

# ============================================
# TITLE
# ============================================

st.title("✈️ Aircraft Engine Risk Prediction System")

st.markdown("""
This system predicts aircraft engine risk using:

- Sensor vibration analysis
- Thermal stress monitoring
- Oil pressure behavior
- Probabilistic degradation modeling
- Explainable machine learning
""")

# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("Enter Aircraft Sensor Values")

rpm = st.sidebar.slider(
    "RPM",
    2000,
    7000,
    4500
)

vibration = st.sidebar.slider(
    "Vibration RMS",
    1.0,
    8.0,
    3.5
)

temperature = st.sidebar.slider(
    "Sensor Temperature",
    60,
    120,
    85
)

oil_pressure = st.sidebar.slider(
    "Oil Pressure",
    15,
    40,
    35
)

load_factor = st.sidebar.slider(
    "Load Factor",
    0.4,
    1.2,
    0.8
)

acoustic = st.sidebar.slider(
    "Acoustic dB",
    50,
    85,
    65
)

altitude = st.sidebar.slider(
    "Altitude (ft)",
    0,
    40000,
    30000
)

# ============================================
# FEATURE ENGINEERING
# ============================================

thermal_stress = temperature * load_factor

lubrication_risk = vibration / (oil_pressure + 1e-6)

mechanical_stress = rpm * load_factor

degradation_score = (
    vibration +
    temperature / 100 -
    oil_pressure / 40
)

# Rolling features
# Since real-time app has no previous sequence,
# use current values as approximation

vibration_roll_mean = vibration

temp_roll_mean = temperature

# ============================================
# INPUT DATAFRAME
# ============================================

input_data = pd.DataFrame([{
    "rpm": rpm,
    "vibration_rms": vibration,
    "sensor_temperature": temperature,
    "oil_pressure": oil_pressure,
    "load_factor": load_factor,
    "acoustic_db": acoustic,
    "altitude_ft": altitude,

    "thermal_stress": thermal_stress,
    "lubrication_risk": lubrication_risk,
    "mechanical_stress": mechanical_stress,
    "degradation_score": degradation_score,

    "vibration_roll_mean": vibration_roll_mean,
    "temp_roll_mean": temp_roll_mean
}])

# ============================================
# DISPLAY INPUT DATA
# ============================================

st.subheader("Input Sensor Data")

st.dataframe(input_data)

# ============================================
# PREDICTION
# ============================================

if st.button("Predict Engine Risk"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Results")

    st.metric(
        label="Risk Probability",
        value=f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    # ========================================
    # RISK LEVELS
    # ========================================

    if probability < 0.30:

        st.success("✅ ENGINE STATUS: SAFE")

        risk_level = "SAFE"

    elif probability < 0.60:

        st.warning("⚠️ ENGINE STATUS: MODERATE RISK")

        risk_level = "MODERATE"

    else:

        st.error("🚨 ENGINE STATUS: HIGH RISK")

        risk_level = "HIGH"

    # ========================================
    # EXTRA ANALYSIS
    # ========================================

    st.subheader("Engine Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.info(f"""
        Thermal Stress: {thermal_stress:.2f}

        Lubrication Risk: {lubrication_risk:.2f}

        Mechanical Stress: {mechanical_stress:.2f}
        """)

    with col2:

        st.info(f"""
        Degradation Score: {degradation_score:.2f}

        Risk Level: {risk_level}

        Altitude: {altitude} ft
        """)

    # ========================================
    # SIMPLE RECOMMENDATION SYSTEM
    # ========================================

    st.subheader("Recommended Actions")

    if probability < 0.30:

        st.success("""
        - Engine operating normally
        - Continue standard monitoring
        - No immediate maintenance required
        """)

    elif probability < 0.60:

        st.warning("""
        - Monitor vibration closely
        - Inspect lubrication system
        - Schedule preventive maintenance
        """)

    else:

        st.error("""
        - Immediate inspection required
        - Possible engine degradation detected
        - Reduce operational stress
        - Maintenance strongly recommended
        """)

# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.markdown("""
### Project Features

- Time-aware aircraft engine monitoring
- Probabilistic degradation modeling
- Feature engineered risk analysis
- Explainable machine learning
- Real-time engine risk prediction dashboard
""")