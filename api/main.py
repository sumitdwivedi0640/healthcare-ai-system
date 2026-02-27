from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from sqlalchemy.orm import Session
from datetime import datetime

# Local Imports
from database import SessionLocal, engine
from models_db import Base, SepsisRecord, TumorRecord, TreatmentRecord
from logger_config import setup_logger

# ============================
# Setup
# ============================

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Healthcare System",
    description="Sepsis + Tumor Detection + Treatment Recommendation",
    version="1.0"
)

logger = setup_logger()
MODEL_VERSION = "v1.0"

# ============================
# API Key Security
# ============================

API_KEY = "healthcare-secret-key"
api_key_header = APIKeyHeader(name="x-api-key")


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        logger.warning("Unauthorized access attempt detected.")
        raise HTTPException(status_code=401, detail="Unauthorized")
    return api_key


# ============================
# Database Dependency
# ============================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================
# Load Models
# ============================

sepsis_model = joblib.load("models/sepsis_model.pkl")
tumor_model = tf.keras.models.load_model(
    "models/brain_tumor_model_clean.keras"
)

# ============================
# Request Schemas
# ============================


class SepsisRequest(BaseModel):
    patient_name: str
    age: int
    gender: str
    features: list[float]


class TumorRequest(BaseModel):
    patient_name: str
    age: int
    gender: str
    image_path: str


class TreatmentRequest(BaseModel):
    patient_name: str
    age: int
    gender: str
    disease: str
    severity: str


# ============================
# SEPSIS ENDPOINT
# ============================

@app.post("/predict-sepsis", dependencies=[Depends(verify_api_key)])
def predict_sepsis(data: SepsisRequest, db: Session = Depends(get_db)):

    try:
        if len(data.features) != sepsis_model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"Model expects {sepsis_model.n_features_in_} features"
            )

        input_data = np.array(data.features).reshape(1, -1)

        prediction = int(sepsis_model.predict(input_data)[0])
        probability = float(sepsis_model.predict_proba(input_data)[0][1])

        record = SepsisRecord(
            patient_name=data.patient_name,
            age=data.age,
            gender=data.gender,
            prediction=prediction,
            probability=probability
        )

        db.add(record)
        db.commit()

        return {
            "prediction": prediction,
            "probability": probability
        }

    except Exception as e:
        logger.error(f"Sepsis prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Sepsis prediction failed")


# ============================
# TUMOR ENDPOINT
# ============================

@app.post("/predict-tumor", dependencies=[Depends(verify_api_key)])
def predict_tumor(data: TumorRequest, db: Session = Depends(get_db)):

    logger.info("===== Tumor Endpoint Triggered =====")

    try:
        from tensorflow.keras.preprocessing import image

        # Load image
        img = image.load_img(data.image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # -------------------------
        # STEP 1: Modality Detection
        # -------------------------
        modality_prediction = modality_model.predict(img_array)
        modality_index = int(np.argmax(modality_prediction))
        detected_modality = MODALITY_CLASSES[modality_index]

        logger.info(f"Detected Modality: {detected_modality}")

        # If not MRI, stop here
        if detected_modality != "MRI":
            return {
                "detected_modality": detected_modality,
                "message": "Tumor detection works only on MRI scans."
            }

        # -------------------------
        # STEP 2: Tumor Detection
        # -------------------------
        tumor_prediction = float(tumor_model.predict(img_array)[0][0])

        label = "Tumor Detected" if tumor_prediction > 0.5 else "No Tumor"
        confidence = tumor_prediction if tumor_prediction > 0.5 else 1 - tumor_prediction

        logger.info(f"Tumor result: {label}, confidence={confidence}")

        # Save to DB
        record = TumorRecord(
            patient_name=data.patient_name,
            age=data.age,
            gender=data.gender,
            label=label,
            confidence=confidence
        )
        db.add(record)
        db.commit()

        return {
            "detected_modality": detected_modality,
            "label": label,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Tumor prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Tumor prediction failed")



modality_model = tf.keras.models.load_model("models/modality_model.keras")

MODALITY_CLASSES = ["MRI", "CT", "XRay"]
# ============================
# TREATMENT ENDPOINT
# ============================

@app.post("/recommend-treatment")
def recommend_treatment(data: TreatmentRequest, db: Session = Depends(get_db)):

    try:
        disease = data.disease.lower()
        severity = data.severity.lower()

        if disease == "sepsis":
            if severity == "high":
                recommended_action = "Immediate ICU admission"
                treatment_plan = "Broad-spectrum IV antibiotics, fluids"
                urgency = "Critical"
            elif severity == "medium":
                recommended_action = "Hospital admission"
                treatment_plan = "IV antibiotics and monitoring"
                urgency = "High"
            else:
                recommended_action = "Outpatient monitoring"
                treatment_plan = "Oral antibiotics"
                urgency = "Moderate"

        elif disease == "tumor":
            if severity == "high":
                recommended_action = "Urgent neurosurgery consultation"
                treatment_plan = "MRI and surgical planning"
                urgency = "Critical"
            elif severity == "medium":
                recommended_action = "Oncology referral"
                treatment_plan = "MRI monitoring"
                urgency = "High"
            else:
                recommended_action = "Routine monitoring"
                treatment_plan = "Periodic MRI follow-up"
                urgency = "Moderate"
        else:
            raise HTTPException(status_code=400, detail="Invalid disease")

        record = TreatmentRecord(
            patient_name=data.patient_name,
            age=data.age,
            gender=data.gender,
            disease=disease,
            severity=severity,
            recommendation=recommended_action
        )

        db.add(record)
        db.commit()

        return {
            "recommended_action": recommended_action,
            "treatment_plan": treatment_plan,
            "urgency": urgency
        }

    except Exception as e:
        logger.error(f"Treatment recommendation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Treatment recommendation failed")


# ============================
# HISTORY ENDPOINTS
# ============================

@app.get("/history/sepsis")
def get_sepsis_history(db: Session = Depends(get_db)):
    records = db.query(SepsisRecord).order_by(
        SepsisRecord.timestamp.desc()).all()
    return records


@app.get("/history/tumor")
def get_tumor_history(db: Session = Depends(get_db)):
    records = db.query(TumorRecord).order_by(
        TumorRecord.timestamp.desc()).all()
    return records


@app.get("/history/treatment")
def get_treatment_history(db: Session = Depends(get_db)):
    records = db.query(TreatmentRecord).order_by(
        TreatmentRecord.timestamp.desc()).all()
    return records
