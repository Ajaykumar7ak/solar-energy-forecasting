import requests
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. LOCATION & TIME RANGE (5 YEARS → >40k rows)
# --------------------------------------------------
LATITUDE = 13.08    # Example: Tamil Nadu
LONGITUDE = 80.27

START_DATE = "2019-01-01"
END_DATE   = "2023-12-31"

# --------------------------------------------------
# 2. OPEN-METEO API CALL
# --------------------------------------------------
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": [
        "direct_normal_irradiance",
        "temperature_2m",
        "windspeed_10m"
    ],
    "timezone": "auto"
}

response = requests.get(url, params=params)
data = response.json()

# --------------------------------------------------
# 3. CREATE DATAFRAME
# --------------------------------------------------
df = pd.DataFrame({
    "time": pd.to_datetime(data["hourly"]["time"]),
    "DNI_W_m2": data["hourly"]["direct_normal_irradiance"],
    "Ambient_Temp_C": data["hourly"]["temperature_2m"],
    "Wind_Speed_mps": data["hourly"]["windspeed_10m"]
})

df.set_index("time", inplace=True)

# Time features
df["Hour"] = df.index.hour
df["DayOfYear"] = df.index.dayofyear

# --------------------------------------------------
# 4. SOLAR THERMAL PHYSICS MODEL
# --------------------------------------------------

# System constants (assumed, realistic)
COLLECTOR_AREA = 600        # m²
THERMAL_EFF    = 0.55       # collector efficiency
TURBINE_EFF    = 0.35       # steam turbine efficiency

# Thermal power (kW)
df["Thermal_Power_kW"] = (
    df["DNI_W_m2"] * COLLECTOR_AREA * THERMAL_EFF / 1000
)

# Electrical output (kW)
df["Electrical_Power_kW"] = (
    df["Thermal_Power_kW"] * TURBINE_EFF
)

# --------------------------------------------------
# 5. WATER PUMPING MODEL
# --------------------------------------------------
RHO = 1000        # kg/m³
G = 9.81          # m/s²
HEAD = 15         # m
FLOW_RATE = 0.02  # m³/s
PUMP_EFF = 0.7

df["Pump_Power_kW"] = (
    (RHO * G * FLOW_RATE * HEAD) / (PUMP_EFF * 1000)
)

# --------------------------------------------------
# 6. CLEAN DATA (REMOVE NIGHT HOURS)
# --------------------------------------------------
df = df[df["DNI_W_m2"] > 0]

# --------------------------------------------------
# 7. SAVE DATASET
# --------------------------------------------------
df.to_csv("solar_thermal_time_series_openmeteo.csv")

print("Dataset generated successfully")
print("Total samples:", len(df))
print(df.head())
