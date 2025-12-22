"""
User Schemas
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr
from uuid import UUID


class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: str = "doctor"


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DoctorInfo(BaseModel):
    specialty: Optional[str] = None
    phone: Optional[str] = None
    hospital: Optional[str] = None


class DoctorCreate(UserCreate):
    doctor_info: Optional[DoctorInfo] = None


class DoctorResponse(UserResponse):
    doctor: Optional[DoctorInfo] = None
    
    class Config:
        from_attributes = True
