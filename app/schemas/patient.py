"""
Patient Schemas
"""
from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, EmailStr
from uuid import UUID


class PatientBase(BaseModel):
    full_name: str
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    email: Optional[EmailStr] = None
    id_number: Optional[str] = None


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    full_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    email: Optional[EmailStr] = None


class PatientResponse(PatientBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    age: Optional[int] = None
    
    class Config:
        from_attributes = True


class PatientListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    patients: List[PatientResponse]


class PatientSearchRequest(BaseModel):
    query: str  # Search by name, phone, or id_number
    page: int = 1
    page_size: int = 10
