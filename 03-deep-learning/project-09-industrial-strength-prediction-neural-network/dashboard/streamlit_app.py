"""Interactive Streamlit dashboard for concrete strength predictions."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference.predictor import ConcreteStrengthPredictor


MODEL_PATH = PROJECT_ROOT / "models" / "concrete_strength_mlp.pt"


@st.cache_resource
def load_predictor() -> ConcreteStrengthPredictor:
    return ConcreteStrengthPredictor.from_artifact(MODEL_PATH)


st.set_page_config(page_title="Concrete Strength Predictor", layout="wide")
st.title("Industrial Strength Prediction Neural Network")

left, right = st.columns([2, 1])

with left:
    st.subheader("Concrete mix inputs")
    cement = st.slider("Cement kg/m3", 0.0, 600.0, 300.0, 1.0)
    slag = st.slider("Blast furnace slag kg/m3", 0.0, 400.0, 50.0, 1.0)
    fly_ash = st.slider("Fly ash kg/m3", 0.0, 250.0, 50.0, 1.0)
    water = st.slider("Water kg/m3", 100.0, 250.0, 180.0, 1.0)
    superplasticizer = st.slider("Superplasticizer kg/m3", 0.0, 35.0, 6.0, 0.5)
    coarse = st.slider("Coarse aggregate kg/m3", 750.0, 1200.0, 970.0, 1.0)
    fine = st.slider("Fine aggregate kg/m3", 550.0, 1000.0, 775.0, 1.0)
    age = st.slider("Age days", 1.0, 365.0, 28.0, 1.0)

payload = {
    "cement": cement,
    "blast_furnace_slag": slag,
    "fly_ash": fly_ash,
    "water": water,
    "superplasticizer": superplasticizer,
    "coarse_aggregate": coarse,
    "fine_aggregate": fine,
    "age": age,
}

with right:
    st.subheader("Predicted strength")
    try:
        predictor = load_predictor()
        prediction = predictor.predict(payload)[0]
        st.metric("Compressive strength", f"{prediction:.2f} MPa")
        st.caption("Model: PyTorch MLP with standardized input features")
        st.json(predictor.metrics)
    except FileNotFoundError:
        st.error("Train the model first: python scripts/train.py")

