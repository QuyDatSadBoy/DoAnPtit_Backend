"""
Patient Schemas
"""
from datetime import datetime, date
from typing import Optional, List, Any
from pydantic import BaseModel, EmailStr, field_validator
from uuid import UUID


def empty_to_none(v: Any) -> Any:
    """Convert empty string to None"""
    if v == '' or v == 'null':
        return None
    return v


class PatientBase(BaseModel):
    full_name: str
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    email: Optional[EmailStr] = None
    id_number: Optional[str] = None
    
    @field_validator('date_of_birth', 'gender', 'phone', 'address', 'email', 'id_number', mode='before')
    @classmethod
    def convert_empty_to_none(cls, v):
        return empty_to_none(v)


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    full_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    email: Optional[EmailStr] = None
    
    @field_validator('date_of_birth', 'gender', 'phone', 'address', 'email', mode='before')
    @classmethod
    def convert_empty_to_none(cls, v):
        return empty_to_none(v)


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
