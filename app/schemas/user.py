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
    phone: Optional[str] = None
    role: str = "doctor"


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None


class ProfileUpdate(BaseModel):
    """Schema for updating user profile (self-update)"""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None


class PasswordChange(BaseModel):
    """Schema for changing password"""
    current_password: str
    new_password: str


class UserResponse(UserBase):
    id: UUID
    avatar: Optional[str] = None
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
