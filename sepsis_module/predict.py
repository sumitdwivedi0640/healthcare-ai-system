import joblib
import numpy as np

# Load once when file is imported
sepsis_model = joblib.load("models/sepsis_model.pkl")
sepsis_scaler = joblib.load("models/sepsis_scaler.pkl")


def predict_sepsis(features: list):
    features_array = np.array(features).reshape(1, -1)

    scaled = sepsis_scaler.transform(features_array)

    prediction = sepsis_model.predict(scaled)[0]
    probability = sepsis_model.predict_proba(scaled)[0][1]

    return int(prediction), float(probability)


