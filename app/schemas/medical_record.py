"""
Medical Record Schemas
"""
from datetime import datetime, date
from typing import Optional, List, Any, Dict
from pydantic import BaseModel
from uuid import UUID


class InferenceHistoryItem(BaseModel):
    id: str
    xray_path: str
    ct_path: Optional[str] = None
    status: str  # pending, processing, completed, failed
    created_at: str
    completed_at: Optional[str] = None
    guidance_scale: Optional[float] = None
    error: Optional[str] = None
    progress: Optional[int] = None
    
    class Config:
        extra = "allow"  # Allow extra fields


class MedicalRecordBase(BaseModel):
    diagnosis: Optional[str] = None
    symptoms: Optional[str] = None
    notes: Optional[str] = None


class MedicalRecordCreate(MedicalRecordBase):
    patient_id: UUID
    visit_date: Optional[date] = None


class MedicalRecordUpdate(MedicalRecordBase):
    pass


class MedicalRecordResponse(MedicalRecordBase):
    id: UUID
    patient_id: UUID
    doctor_id: UUID
    created_by: UUID
    visit_date: date
    infer_history: List[Dict[str, Any]]  # Allow flexible JSONB structure
    created_at: datetime
    updated_at: datetime
    
    # Nested patient info
    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    
    class Config:
        from_attributes = True


class MedicalRecordListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    records: List[MedicalRecordResponse]


class MedicalRecordWithPatient(MedicalRecordResponse):
    patient: Any  # PatientResponse
    
    class Config:
        from_attributes = True
