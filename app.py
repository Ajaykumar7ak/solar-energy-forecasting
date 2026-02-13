# ============================================================
# FLASK BACKEND
# Solar Thermal Power Forecasting (Next 48 Hours)
# ============================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import requests
import joblib
import json
from datetime import datetime

app = Flask(__name__)

# -----------------------------
# LOAD MODEL & METRICS
# -----------------------------
model = joblib.load("solar_thermal_random_forest_model.pkl")

with open("model_performance_metrics.json", "r") as f:
    metrics = json.load(f)

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
# HOME PAGE
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)

# -----------------------------
# PREDICTION API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    print("Received prediction request")
    try:
        data = request.json
        print(f"Request data: {data}")
        
        lat = float(data["latitude"])
        lon = float(data["longitude"])

        print("Fetching weather forecast...")
        forecast_df = get_weather_forecast(lat, lon)
        print("Weather forecast fetched successfully")

        features = [
            "DNI_W_m2",
            "Ambient_Temp_C",
            "Wind_Speed_mps",
            "Hour",
            "DayOfYear"
        ]

        X_future = forecast_df[features]
        print("Making prediction...")
        preds = model.predict(X_future)
        preds = np.clip(preds, 0, None)
        print("Prediction successful")

        forecast_df["Predicted_Electrical_Power_kW"] = preds
        
        # Calculate Thermal Power (before electrical conversion)
        # Solar thermal systems typically have 30-40% efficiency for thermal-to-electrical
        # Thermal Power = Electrical Power / Efficiency
        THERMAL_TO_ELECTRICAL_EFFICIENCY = 0.35
        forecast_df["Predicted_Thermal_Power_kW"] = preds / THERMAL_TO_ELECTRICAL_EFFICIENCY

        result = forecast_df[[
            "time",
            "Predicted_Electrical_Power_kW",
            "Predicted_Thermal_Power_kW",
            "DNI_W_m2",
            "Ambient_Temp_C",
            "Wind_Speed_mps"
        ]].copy()

        result["time"] = result["time"].astype(str)
        
        response_data = result.to_dict(orient="records")
        print(f"Returning {len(response_data)} records")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"ERROR in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
