import joblib

# Load the existing model
model = joblib.load("solar_thermal_random_forest_model.pkl")

# Save the model with compression (level 9 is max)
joblib.dump(model, "solar_thermal_random_forest_model_compressed.pkl", compress=9)

print("Model compressed and saved as 'solar_thermal_random_forest_model_compressed.pkl'")
