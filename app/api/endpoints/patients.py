"""
Patient Management Endpoints
"""
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.db.session import get_db
from app.models.patient import Patient
from app.models.user import User
from app.schemas.patient import (
    PatientCreate, PatientUpdate, PatientResponse, 
    PatientListResponse
)
from app.core.security import require_doctor

router = APIRouter()


@router.get("", response_model=PatientListResponse)
async def get_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Get all patients with pagination and search"""
    query = db.query(Patient)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                Patient.full_name.ilike(search_term),
                Patient.phone.ilike(search_term),
                Patient.id_number.ilike(search_term),
                Patient.email.ilike(search_term)
            )
        )
    
    total = query.count()
    patients = query.order_by(Patient.created_at.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()
    
    return PatientListResponse(
        total=total,
        page=page,
        page_size=page_size,
        patients=[PatientResponse.model_validate(p) for p in patients]
    )


@router.post("", response_model=PatientResponse)
async def create_patient(
    patient_data: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Create new patient"""
    # Check if id_number already exists
    if patient_data.id_number:
        existing = db.query(Patient).filter(
            Patient.id_number == patient_data.id_number
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Patient with this ID number already exists"
            )
    
    patient = Patient(
        id=uuid.uuid4(),
        full_name=patient_data.full_name,
        date_of_birth=patient_data.date_of_birth,
        gender=patient_data.gender,
        phone=patient_data.phone,
        address=patient_data.address,
        email=patient_data.email,
        id_number=patient_data.id_number
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    
    return patient


@router.get("/search", response_model=PatientListResponse)
async def search_patients(
    q: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Search patients by name, phone, or ID number"""
    search_term = f"%{q}%"
    query = db.query(Patient).filter(
        or_(
            Patient.full_name.ilike(search_term),
            Patient.phone.ilike(search_term),
            Patient.id_number.ilike(search_term)
        )
    )
    
    total = query.count()
    patients = query.order_by(Patient.full_name).offset(
        (page - 1) * page_size
    ).limit(page_size).all()
    
    return PatientListResponse(
        total=total,
        page=page,
        page_size=page_size,
        patients=[PatientResponse.model_validate(p) for p in patients]
    )


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Get patient by ID"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    return patient


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: uuid.UUID,
    patient_data: PatientUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Update patient information"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Update fields
    update_data = patient_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(patient, field, value)
    
    db.commit()
    db.refresh(patient)
    
    return patient


@router.delete("/{patient_id}")
async def delete_patient(
    patient_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Delete patient"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Check if patient has medical records
    if patient.medical_records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete patient with medical records"
        )
    
    db.delete(patient)
    db.commit()
    
    return {"message": "Patient deleted successfully"}
