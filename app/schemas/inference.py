"""
Inference Schemas
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from uuid import UUID


class InferenceRequest(BaseModel):
    medical_record_id: UUID
    guidance_scale: float = 1.0


class InferenceResponse(BaseModel):
    success: bool
    inference_id: str
    medical_record_id: UUID
    status: str
    message: str


class InferenceStatusResponse(BaseModel):
    inference_id: str
    status: str  # pending, processing, completed, failed
    xray_path: str
    ct_path: Optional[str] = None
    progress: Optional[int] = None  # 0-100
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class InferenceNotification(BaseModel):
    type: str = "inference_complete"
    inference_id: str
    medical_record_id: str
    patient_id: str
    status: str
    ct_path: Optional[str] = None
    message: str
    timestamp: str


class StatisticsResponse(BaseModel):
    total_patients: int
    total_records: int
    total_inferences: int
    completed_inferences: int
    pending_inferences: int
    failed_inferences: int
    records_today: int
    records_this_week: int
    records_this_month: int
    
    # Charts data
    records_by_month: List[Dict[str, Any]]
    inferences_by_status: Dict[str, int]
    top_doctors: List[Dict[str, Any]]
