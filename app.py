import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="MCF-7 Nanoparticle Drug Release Predictor",
    layout="centered"
)

st.title("MCF-7 Nanoparticle Drug Release Predictor")
st.caption(
    "Predict drug release amount (%) from nanoparticle properties and release conditions (MCF-7 related dataset)."
)

# =========================
# Load model (relative path)
# =========================
MODEL_PATH = Path(__file__).parent / "BEST_Model_DatasetThird.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# Input section
# =========================
st.subheader("Input features")

size = st.number_input("Size (DLS) of nanoparticle (nm)-mean", value=100.0)
pdi = st.number_input("Polydispersity Index (PDI) of nanoparticle-mean", value=0.20)
zeta = st.number_input("Zeta potential of nanoparticle (mV)-mean", value=-10.0)
dlc = st.number_input("Drug loading capacity (%)-mean", value=5.0)
ee = st.number_input("Entrapment efficiency (%)-mean", value=70.0)
temp = st.number_input("Temperature (°C)", value=37.0)
ph = st.number_input("pH", value=7.4)
time_h = st.number_input("Time of Drug release (h)", value=24.0)

X = np.array([[size, pdi, zeta, dlc, ee, temp, ph, time_h]], dtype=float)

# =========================
# Prediction
# =========================
if st.button("Predict drug release (%)"):
    prediction = model.predict(X)[0]
    st.success(f"Predicted Drug release amount (%): {prediction:.2f}")
