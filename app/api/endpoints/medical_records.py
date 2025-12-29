"""
Medical Record Endpoints
"""
import os
import uuid
import logging
from typing import Optional, List
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session, joinedload

from app.db.session import get_db
from app.models.user import User
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.models.medical_record import MedicalRecord
from app.schemas.medical_record import (
    MedicalRecordCreate, MedicalRecordUpdate, 
    MedicalRecordResponse, MedicalRecordListResponse
)
from app.core.security import require_doctor, get_current_user
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def get_doctor_id(current_user: User, db: Session) -> uuid.UUID:
    """Get doctor ID from current user"""
    doctor = db.query(Doctor).filter(Doctor.user_id == current_user.id).first()
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not a doctor"
        )
    return doctor.id


@router.get("", response_model=MedicalRecordListResponse)
async def get_medical_records(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    patient_id: Optional[uuid.UUID] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Get all medical records with pagination"""
    try:
        query = db.query(MedicalRecord).options(
            joinedload(MedicalRecord.patient),
            joinedload(MedicalRecord.doctor).joinedload(Doctor.user)
        )
        
        if patient_id:
            query = query.filter(MedicalRecord.patient_id == patient_id)
        
        total = query.count()
        records = query.order_by(MedicalRecord.visit_date.desc()).offset(
            (page - 1) * page_size
        ).limit(page_size).all()
        
        # Convert to response format
        record_responses = []
        for record in records:
            response = MedicalRecordResponse(
                id=record.id,
                patient_id=record.patient_id,
                doctor_id=record.doctor_id,
                created_by=record.created_by,
                visit_date=record.visit_date,
                diagnosis=record.diagnosis,
                symptoms=record.symptoms,
                notes=record.notes,
                infer_history=record.infer_history or [],
                created_at=record.created_at,
                updated_at=record.updated_at,
                patient_name=record.patient.full_name if record.patient else None,
                doctor_name=record.doctor.user.full_name if record.doctor and record.doctor.user else None
            )
            record_responses.append(response)
        
        return MedicalRecordListResponse(
            total=total,
            page=page,
            page_size=page_size,
            records=record_responses
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting medical records: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi lấy danh sách hồ sơ: {str(e)}")


@router.post("", response_model=MedicalRecordResponse)
async def create_medical_record(
    record_data: MedicalRecordCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Create new medical record for a patient"""
    try:
        # Verify patient exists
        patient = db.query(Patient).filter(Patient.id == record_data.patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        # Get doctor ID
        doctor_id = get_doctor_id(current_user, db)
        
        record = MedicalRecord(
            id=uuid.uuid4(),
            patient_id=record_data.patient_id,
            doctor_id=doctor_id,
            created_by=current_user.id,
            visit_date=record_data.visit_date or date.today(),
            diagnosis=record_data.diagnosis,
            symptoms=record_data.symptoms,
            notes=record_data.notes,
            infer_history=[]
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        
        return MedicalRecordResponse(
            id=record.id,
            patient_id=record.patient_id,
            doctor_id=record.doctor_id,
            created_by=record.created_by,
            visit_date=record.visit_date,
            diagnosis=record.diagnosis,
            symptoms=record.symptoms,
            notes=record.notes,
            infer_history=record.infer_history or [],
            created_at=record.created_at,
            updated_at=record.updated_at,
            patient_name=patient.full_name
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating medical record: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi tạo hồ sơ: {str(e)}")


@router.get("/patient/{patient_id}", response_model=List[MedicalRecordResponse])
async def get_patient_records(
    patient_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Get all medical records for a specific patient"""
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        records = db.query(MedicalRecord).options(
            joinedload(MedicalRecord.doctor).joinedload(Doctor.user)
        ).filter(
            MedicalRecord.patient_id == patient_id
        ).order_by(MedicalRecord.visit_date.desc()).all()
        
        return [
            MedicalRecordResponse(
                id=r.id,
                patient_id=r.patient_id,
                doctor_id=r.doctor_id,
                created_by=r.created_by,
                visit_date=r.visit_date,
                diagnosis=r.diagnosis,
                symptoms=r.symptoms,
                notes=r.notes,
                infer_history=r.infer_history or [],
                created_at=r.created_at,
                updated_at=r.updated_at,
                patient_name=patient.full_name,
                doctor_name=r.doctor.user.full_name if r.doctor and r.doctor.user else None
            )
            for r in records
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patient records: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi lấy hồ sơ bệnh nhân: {str(e)}")


@router.get("/{record_id}", response_model=MedicalRecordResponse)
async def get_medical_record(
    record_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Get medical record by ID"""
    try:
        record = db.query(MedicalRecord).options(
            joinedload(MedicalRecord.patient),
            joinedload(MedicalRecord.doctor).joinedload(Doctor.user)
        ).filter(MedicalRecord.id == record_id).first()
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Medical record not found"
            )
        
        return MedicalRecordResponse(
            id=record.id,
            patient_id=record.patient_id,
            doctor_id=record.doctor_id,
            created_by=record.created_by,
            visit_date=record.visit_date,
            diagnosis=record.diagnosis,
            symptoms=record.symptoms,
            notes=record.notes,
            infer_history=record.infer_history or [],
            created_at=record.created_at,
            updated_at=record.updated_at,
            patient_name=record.patient.full_name if record.patient else None,
            doctor_name=record.doctor.user.full_name if record.doctor and record.doctor.user else None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting medical record: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi lấy hồ sơ: {str(e)}")


@router.put("/{record_id}", response_model=MedicalRecordResponse)
async def update_medical_record(
    record_id: uuid.UUID,
    record_data: MedicalRecordUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Update medical record - only creator can edit"""
    try:
        record = db.query(MedicalRecord).options(
            joinedload(MedicalRecord.patient),
            joinedload(MedicalRecord.doctor).joinedload(Doctor.user)
        ).filter(MedicalRecord.id == record_id).first()
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Medical record not found"
            )
        
        # Check ownership - only the doctor who created this record can edit
        # Admin can also edit
        if current_user.role != "admin" and record.created_by != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Bạn không có quyền sửa hồ sơ này. Chỉ bác sĩ tạo hồ sơ mới được sửa."
            )
        
        # Update fields
        if record_data.diagnosis is not None:
            record.diagnosis = record_data.diagnosis
        if record_data.symptoms is not None:
            record.symptoms = record_data.symptoms
        if record_data.notes is not None:
            record.notes = record_data.notes
        
        db.commit()
        db.refresh(record)
        
        return MedicalRecordResponse(
            id=record.id,
            patient_id=record.patient_id,
            doctor_id=record.doctor_id,
            created_by=record.created_by,
            visit_date=record.visit_date,
            diagnosis=record.diagnosis,
            symptoms=record.symptoms,
            notes=record.notes,
            infer_history=record.infer_history or [],
            created_at=record.created_at,
            updated_at=record.updated_at,
            patient_name=record.patient.full_name if record.patient else None,
            doctor_name=record.doctor.user.full_name if record.doctor and record.doctor.user else None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating medical record: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi cập nhật hồ sơ: {str(e)}")


@router.delete("/{record_id}")
async def delete_medical_record(
    record_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Delete medical record - only creator or admin can delete"""
    try:
        record = db.query(MedicalRecord).filter(MedicalRecord.id == record_id).first()
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Medical record not found"
            )
        
        # Check ownership - only the doctor who created this record or admin can delete
        if current_user.role != "admin" and record.created_by != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Bạn không có quyền xóa hồ sơ này. Chỉ bác sĩ tạo hồ sơ mới được xóa."
            )
        
        db.delete(record)
        db.commit()
        
        return {"message": "Medical record deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting medical record: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi xóa hồ sơ: {str(e)}")


@router.get("/download/{patient_id}/{record_id}/{filename}")
async def download_file(
    patient_id: uuid.UUID,
    record_id: uuid.UUID,
    filename: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Download a file (X-ray image, CT result, etc.) with proper headers for download"""
    try:
        # Construct file path
        file_path = os.path.join(settings.PATIENT_FILES_DIR, str(patient_id), str(record_id), filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Determine media type
        if filename.endswith('.png'):
            media_type = "image/png"
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            media_type = "image/jpeg"
        elif filename.endswith('.nii') or filename.endswith('.nii.gz'):
            media_type = "application/octet-stream"
        else:
            media_type = "application/octet-stream"
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi tải file: {str(e)}")
