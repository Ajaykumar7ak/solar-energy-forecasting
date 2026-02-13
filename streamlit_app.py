import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -----------------------------
# LOAD MODEL & METRICS
# -----------------------------
model = joblib.load("solar_thermal_random_forest_model.pkl")

with open("model_performance_metrics.json", "r") as f:
    metrics = json.load(f)

# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Solar Thermal Power Forecast",
    layout="wide"
)

st.title("☀️ Solar Thermal Power Forecasting System")
st.markdown("### Next 48 Hours Electrical Power Prediction")

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("📍 Location Settings")

latitude = st.sidebar.number_input(
    "Latitude", value=13.08, format="%.4f"
)
longitude = st.sidebar.number_input(
    "Longitude", value=80.27, format="%.4f"
)

# -----------------------------
# WEATHER FORECAST FUNCTION
# -----------------------------
def get_weather_forecast(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "direct_normal_irradiance",
            "temperature_2m",
            "windspeed_10m"
        ],
        "forecast_days": 2,
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "DNI_W_m2": data["hourly"]["direct_normal_irradiance"],
        "Ambient_Temp_C": data["hourly"]["temperature_2m"],
        "Wind_Speed_mps": data["hourly"]["windspeed_10m"]
    })

    df["Hour"] = df["time"].dt.hour
    df["DayOfYear"] = df["time"].dt.dayofyear

    return df

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔮 Predict Next 48 Hours"):
    with st.spinner("Fetching weather & predicting..."):

        forecast_df = get_weather_forecast(latitude, longitude)

        # Feature selection
        features = [
            "DNI_W_m2",
            "Ambient_Temp_C",
            "Wind_Speed_mps",
            "Hour",
            "DayOfYear"
        ]

        X_future = forecast_df[features]

        # Prediction
        forecast_df["Predicted_Electrical_Power_kW"] = model.predict(X_future)
        forecast_df["Predicted_Electrical_Power_kW"] = forecast_df[
            "Predicted_Electrical_Power_kW"
        ].clip(lower=0)

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.success("Prediction completed successfully")

    st.subheader("📊 Forecast Table (Next 48 Hours)")
    st.dataframe(
        forecast_df[[
            "time",
            "DNI_W_m2",
            "Ambient_Temp_C",
            "Wind_Speed_mps",
            "Predicted_Electrical_Power_kW"
        ]]
    )

    # -----------------------------
    # PLOT
    # -----------------------------
    st.subheader("📈 Power Forecast Graph")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        forecast_df["time"],
        forecast_df["Predicted_Electrical_Power_kW"],
        linewidth=2
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Electrical Power (kW)")
    ax.set_title("Next 48 Hours Solar Thermal Power Forecast")
    ax.grid(True)

    st.pyplot(fig)

# -----------------------------
# MODEL PERFORMANCE SECTION
# -----------------------------
st.markdown("---")
st.subheader("📌 Model Performance Summary")

col1, col2, col3 = st.columns(3)

col1.metric("R² Score", metrics["r2_score"])
col2.metric("MAE (kW)", metrics["mae_kW"])
col3.metric("RMSE (kW)", metrics["rmse_kW"])

st.caption(
    f"Model: {metrics['model_name']} | "
    f"Total Samples: {metrics['total_samples']}"
)
