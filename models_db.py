from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
from datetime import datetime


class SepsisRecord(Base):
    __tablename__ = "sepsis_predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String)
    age = Column(Integer)
    gender = Column(String)

    prediction = Column(Integer)
    probability = Column(Float)

    timestamp = Column(DateTime, default=datetime.utcnow)


class TumorRecord(Base):
    __tablename__ = "tumor_predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String)
    age = Column(Integer)
    gender = Column(String)

    label = Column(String)
    confidence = Column(Float)

    timestamp = Column(DateTime, default=datetime.utcnow)


class TreatmentRecord(Base):
    __tablename__ = "treatment_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String)
    age = Column(Integer)
    gender = Column(String)

    disease = Column(String)
    severity = Column(String)
    recommendation = Column(String)

    timestamp = Column(DateTime, default=datetime.utcnow)
