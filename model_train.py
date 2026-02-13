# ============================================================
# SOLAR THERMAL SYSTEM PERFORMANCE PREDICTION
# Random Forest Regression
# Training + Evaluation + Model & Metrics Storage
# ============================================================

# -----------------------------
# 1. IMPORT REQUIRED LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------
# 2. LOAD DATASET
# -----------------------------
df = pd.read_csv(
    "solar_thermal_time_series_openmeteo.csv",
    index_col=0,
    parse_dates=True
)

print("Dataset Loaded Successfully")
print("Total Samples:", df.shape[0])
print(df.head())

# -----------------------------
# 3. FEATURE & TARGET SELECTION
# -----------------------------
features = [
    "DNI_W_m2",
    "Ambient_Temp_C",
    "Wind_Speed_mps",
    "Hour",
    "DayOfYear"
]

target = "Electrical_Power_kW"

X = df[features]
y = df[target]

# -----------------------------
# 4. TRAIN-TEST SPLIT
# (Time-series safe: no shuffle)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# -----------------------------
# 5. INITIALIZE RANDOM FOREST
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# 6. TRAIN THE MODEL
# -----------------------------
rf_model.fit(X_train, y_train)
print("\nModel Training Completed")

# -----------------------------
# 7. PREDICTION
# -----------------------------
y_pred = rf_model.predict(X_test)

# -----------------------------
# 8. MODEL PERFORMANCE METRICS
# -----------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nMODEL PERFORMANCE")
print("---------------------------")
print(f"R² Score  : {r2:.4f}")
print(f"MAE (kW) : {mae:.3f}")
print(f"RMSE (kW): {rmse:.3f}")

# -----------------------------
# 9. SAVE PERFORMANCE METRICS
# (For Web Page / Dashboard)
# -----------------------------
performance_metrics = {
    "model_name": "Random Forest Regressor",
    "application": "Solar Thermal System Performance Prediction",
    "r2_score": round(r2, 4),
    "mae_kW": round(mae, 3),
    "rmse_kW": round(rmse, 3),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "total_samples": int(len(df)),
    "features_used": features
}

with open("model_performance_metrics.json", "w") as f:
    json.dump(performance_metrics, f, indent=4)

print("\nPerformance metrics saved to 'model_performance_metrics.json'")

# -----------------------------
# 10. ACTUAL vs PREDICTED PLOT
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(y_test.values[:500], label="Actual", linewidth=2)
plt.plot(y_pred[:500], label="Predicted", linewidth=2)
plt.xlabel("Time (Samples)")
plt.ylabel("Electrical Power (kW)")
plt.title("Actual vs Predicted Electrical Power Output")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 11. FEATURE IMPORTANCE PLOT
# -----------------------------
importances = rf_model.feature_importances_

plt.figure(figsize=(6, 4))
plt.barh(features, importances)
plt.xlabel("Importance Score")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# -----------------------------
# 12. SAVE TRAINED MODEL
# -----------------------------
joblib.dump(rf_model, "solar_thermal_random_forest_model.pkl")
print("\nTrained model saved as 'solar_thermal_random_forest_model.pkl'")

# -----------------------------
# 13. SAMPLE PREDICTION CHECK
# -----------------------------
sample_input = X_test.iloc[0:1]
sample_output = rf_model.predict(sample_input)

print("\nSample Prediction Check")
print("---------------------------")
print("Input:")
print(sample_input)
print("Predicted Electrical Power (kW):", round(sample_output[0], 3))
