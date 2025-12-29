"""
Patient Management Endpoints
"""
import uuid
import unicodedata
import re
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, func, cast, String

from app.db.session import get_db
from app.models.patient import Patient
from app.models.user import User
from app.models.doctor import Doctor
from app.models.medical_record import MedicalRecord
from app.schemas.patient import (
    PatientCreate, PatientUpdate, PatientResponse, 
    PatientListResponse
)
from app.core.security import require_doctor

router = APIRouter()


def get_doctor_id_from_user(current_user: User, db: Session) -> Optional[uuid.UUID]:
    """Get doctor_id from current user if they are a doctor"""
    if current_user.role == "admin":
        return None  # Admin sees all patients
    
    # Check if user has doctor relationship
    if hasattr(current_user, 'doctor') and current_user.doctor:
        return current_user.doctor.id
    
    # Fallback: find doctor by user email
    doctor = db.query(Doctor).filter(Doctor.email == current_user.email).first()
    if doctor:
        return doctor.id
    
    return None


def get_patient_ids_for_doctor(db: Session, doctor_id: uuid.UUID) -> List[uuid.UUID]:
    """Get list of patient IDs that have medical records with this doctor"""
    patient_ids = db.query(MedicalRecord.patient_id).filter(
        MedicalRecord.doctor_id == doctor_id
    ).distinct().all()
    return [pid[0] for pid in patient_ids]


def remove_accents(text: str) -> str:
    """Remove Vietnamese accents from text for search comparison"""
    if not text:
        return ""
    # Normalize to NFD form (decomposed)
    text = unicodedata.normalize('NFD', text)
    # Remove combining diacritical marks
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    # Convert to lowercase
    text = text.lower()
    # Replace đ/Đ with d
    text = text.replace('đ', 'd').replace('Đ', 'd')
    return text


def patient_matches_search(patient: Patient, search: str) -> bool:
    """Check if patient matches search term (Vietnamese accent-insensitive)"""
    search_lower = search.lower().strip()
    search_no_accent = remove_accents(search)
    
    # Fields to search
    fields = [
        patient.full_name or "",
        patient.phone or "",
        patient.id_number or "",
        patient.email or "",
        patient.address or "",
    ]
    
    for field in fields:
        field_lower = field.lower()
        field_no_accent = remove_accents(field)
        
        # Match if search term found in original or unaccented version
        if search_lower in field_lower:
            return True
        if search_no_accent in field_no_accent:
            return True
    
    return False


def search_patients_in_memory(
    db: Session, 
    search: str, 
    page: int, 
    page_size: int,
    order_by_created: bool = True,
    patient_ids: Optional[List[uuid.UUID]] = None
) -> tuple:
    """
    Search patients with Vietnamese accent-insensitive matching.
    Returns (filtered_patients, total_count)
    
    Args:
        patient_ids: If provided, only search among these patient IDs (for doctor filtering)
    """
    # Get patients - filter by patient_ids if provided
    if patient_ids is not None:
        if len(patient_ids) == 0:
            return [], 0  # Doctor has no patients
        all_patients = db.query(Patient).filter(Patient.id.in_(patient_ids)).all()
    else:
        all_patients = db.query(Patient).all()
    
    # Filter in Python for accent-insensitive search
    if search and search.strip():
        filtered = [p for p in all_patients if patient_matches_search(p, search)]
    else:
        filtered = all_patients
    
    # Sort
    if order_by_created:
        filtered.sort(key=lambda p: p.created_at or "", reverse=True)
    else:
        filtered.sort(key=lambda p: (p.full_name or "").lower())
    
    total = len(filtered)
    
    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    paginated = filtered[start:end]
    
    return paginated, total


@router.get("", response_model=PatientListResponse)
async def get_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """
    Get ALL patients with pagination and search (Vietnamese accent-insensitive).
    All doctors can see all patients in the system.
    """
    try:
        if search and search.strip():
            # Use in-memory search for accent-insensitive matching
            patients, total = search_patients_in_memory(
                db, search, page, page_size, 
                order_by_created=True, 
                patient_ids=None  # No filter - show all patients
            )
        else:
            # No search - use database query directly
            query = db.query(Patient)
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
    except Exception as e:
        print(f"[ERROR] get_patients: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching patients: {str(e)}"
        )


@router.post("", response_model=PatientResponse)
async def create_patient(
    patient_data: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Create new patient"""
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"[ERROR] create_patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating patient: {str(e)}"
        )
    
    return patient


@router.get("/search", response_model=PatientListResponse)
async def search_patients(
    q: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """
    Search ALL patients by name, phone, or ID number (Vietnamese accent-insensitive).
    All doctors can search all patients in the system.
    """
    try:
        patients, total = search_patients_in_memory(
            db, q, page, page_size, 
            order_by_created=False,
            patient_ids=None  # No filter - search all patients
        )
        
        return PatientListResponse(
            total=total,
            page=page,
            page_size=page_size,
            patients=[PatientResponse.model_validate(p) for p in patients]
        )
    except Exception as e:
        print(f"[ERROR] search_patients: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching patients: {str(e)}"
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
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"[ERROR] update_patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating patient: {str(e)}"
        )


@router.delete("/{patient_id}")
async def delete_patient(
    patient_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Delete patient"""
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"[ERROR] delete_patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting patient: {str(e)}"
        )
