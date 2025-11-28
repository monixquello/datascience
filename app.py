
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


@st.cache_resource
def load_all_models():
    lstm = load_model("lstm_model.h5")

    with open("rf_model.pkl", "rb") as f:
        rf = pickle.load(f)

    with open("mlp_model.pkl", "rb") as f:
        mlp = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return lstm, rf, mlp, scaler


lstm_model, rf_model, mlp_model, scaler = load_all_models()

st.set_page_config(page_title="Demand Forecasting", layout="wide")

st.title("📊 Daily Demand Forecasting App")
st.write("This tool allows you to forecast store-wide daily demand using three models: **LSTM**, **Random Forest**, and **MLP**.")

# Sidebar
st.sidebar.header("⚙️ Model Options")
model_choice = st.sidebar.selectbox(
    "Choose a model to use:",
    ["LSTM (30-day sequence)", "Random Forest", "MLP"]
)

st.sidebar.info("Upload or enter your feature values below.")

if model_choice == "LSTM (30-day sequence)":

    st.subheader("LSTM Model Input (Enter last 30 days of sales)")

    values = []
    cols = st.columns(4)

    for i in range(30):
        col = cols[i % 4]
        val = col.number_input(f"Day {i+1}", min_value=0.0, step=1.0, key=f"d{i}")
        values.append(val)

    if st.button("Predict with LSTM"):
        seq = np.array(values).reshape(-1, 1)
        seq_scaled = scaler.transform(seq)
        seq_scaled = seq_scaled.reshape(1, 30, 1)

        pred_scaled = lstm_model.predict(seq_scaled)
        prediction = scaler.inverse_transform(pred_scaled)[0][0]

        st.success(f"📈 **LSTM Forecast:** {prediction:.2f} units")

else:
    st.subheader("📌 Feature Input for Machine Learning Models")

    lag_1 = st.number_input("Lag 1 (yesterday)", step=1.0, min_value=0.0)
    lag_7 = st.number_input("Lag 7 (last week)", step=1.0, min_value=0.0)
    lag_30 = st.number_input("Lag 30 (last month)", step=1.0, min_value=0.0)

    roll_7 = st.number_input("Rolling Mean (7 days)", step=1.0, min_value=0.0)
    roll_30 = st.number_input("Rolling Mean (30 days)", step=1.0, min_value=0.0)

    dayofweek = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))
    month = st.selectbox("Month (1-12)", list(range(1, 13)))
    is_weekend = st.selectbox("Weekend?", [0, 1])

    features = np.array([[lag_1, lag_7, lag_30,
                          roll_7, roll_30,
                          dayofweek, month, is_weekend]])

    if st.button(f"Predict with {model_choice}"):

        if model_choice == "Random Forest":
            pred = rf_model.predict(features)[0]
            st.success(f"🌲 **Random Forest Prediction:** {pred:.2f} units")

        elif model_choice == "MLP":
            pred = mlp_model.predict(features)[0]
            st.success(f"**MLP Prediction:** {pred:.2f} units")



