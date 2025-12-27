"""
API Router Configuration
"""
from fastapi import APIRouter
from app.api.endpoints import auth, users, patients, medical_records, inference, statistics, face_recognition, training_pipeline

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(patients.router, prefix="/patients", tags=["Patients"])
api_router.include_router(medical_records.router, prefix="/medical-records", tags=["Medical Records"])
api_router.include_router(inference.router, prefix="/inference", tags=["Inference"])
api_router.include_router(statistics.router, prefix="/statistics", tags=["Statistics"])
api_router.include_router(face_recognition.router, prefix="/face", tags=["Face Recognition"])
api_router.include_router(training_pipeline.router, prefix="/training-pipeline", tags=["Training Pipeline"])
