"""
Inference Endpoints - X-ray to CT conversion
"""
import os
import uuid
import shutil
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.session import get_db
from app.models.user import User
from app.models.medical_record import MedicalRecord
from app.schemas.inference import InferenceResponse, InferenceStatusResponse
from app.core.security import require_doctor
from app.core.config import settings, get_base_dir, get_patient_files_dir
from app.core.timezone import now_vn

router = APIRouter()


def get_patient_folder(patient_id: str, record_id: str) -> str:
    """Get ABSOLUTE folder path for patient files (for file operations)"""
    # Use get_patient_files_dir() to ensure absolute path
    base_path = get_patient_files_dir() / str(patient_id) / str(record_id)
    base_path.mkdir(parents=True, exist_ok=True)
    return str(base_path)


def get_relative_path(patient_id: str, record_id: str, filename: str) -> str:
    """Get RELATIVE path for storing in database (for serving via static files)"""
    return f"{settings.PATIENT_FILES_DIR}/{patient_id}/{record_id}/{filename}"


@router.post("/upload", response_model=InferenceResponse)
async def upload_xray_for_inference(
    medical_record_id: str = Form(...),
    file: UploadFile = File(...),
    guidance_scale: float = Form(1.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """
    Upload X-ray image and start inference process
    The inference will be processed asynchronously by Celery worker
    """
    # Validate file type
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.npy']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )
    
    # Get medical record
    record = db.query(MedicalRecord).filter(
        MedicalRecord.id == uuid.UUID(medical_record_id)
    ).first()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Medical record not found"
        )
    
    # Check permission: Only record creator or admin can upload X-ray
    if current_user.role != 'admin' and record.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bạn không có quyền tải ảnh X-ray lên hồ sơ này. Chỉ bác sĩ tạo hồ sơ hoặc admin mới được phép."
        )
    
    # Generate inference ID
    inference_id = str(uuid.uuid4())
    
    # Create folder structure: patient_files/{patient_id}/{record_id}/
    folder_path = get_patient_folder(str(record.patient_id), str(record.id))
    
    # Save X-ray file
    xray_filename = f"xray_{inference_id}{file_ext}"
    xray_full_path = os.path.join(folder_path, xray_filename)
    xray_relative_path = get_relative_path(str(record.patient_id), str(record.id), xray_filename)
    
    with open(xray_full_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Update medical record with inference history (store RELATIVE paths)
    if record.infer_history is None:
        record.infer_history = []
    
    record.infer_history = record.infer_history + [{
        "id": inference_id,
        "xray_path": xray_relative_path,  # Store relative path for static serving
        "ct_path": None,
        "status": "pending",
        "guidance_scale": guidance_scale,
        "created_at": now_vn().isoformat(),
        "completed_at": None
    }]
    
    db.commit()
    
    # Queue the inference task with Celery (pass ABSOLUTE path for file operations)
    try:
        from app.worker.tasks import process_inference
        process_inference.delay(
            inference_id=inference_id,
            medical_record_id=medical_record_id,
            patient_id=str(record.patient_id),
            user_id=str(current_user.id),  # For targeted Socket.IO notification
            xray_path=xray_full_path,  # Pass absolute path to worker
            guidance_scale=guidance_scale
        )
    except Exception as e:
        print(f"Warning: Could not queue task: {e}")
        # Update status to show queuing failed
        for item in record.infer_history:
            if item["id"] == inference_id:
                item["status"] = "queue_failed"
                item["error"] = str(e)
        db.commit()
    
    return InferenceResponse(
        success=True,
        inference_id=inference_id,
        medical_record_id=uuid.UUID(medical_record_id),
        status="pending",
        message="Inference queued successfully. You will be notified when complete."
    )


@router.get("/status/{inference_id}", response_model=InferenceStatusResponse)
async def get_inference_status(
    inference_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Get status of an inference task"""
    # Find the inference in any medical record
    records = db.query(MedicalRecord).filter(
        MedicalRecord.infer_history.contains([{"id": inference_id}])
    ).all()
    
    # Search through records
    for record in db.query(MedicalRecord).all():
        if record.infer_history:
            for item in record.infer_history:
                if item.get("id") == inference_id:
                    return InferenceStatusResponse(
                        inference_id=inference_id,
                        status=item.get("status", "unknown"),
                        xray_path=item.get("xray_path", ""),
                        ct_path=item.get("ct_path"),
                        progress=item.get("progress"),
                        error_message=item.get("error"),
                        created_at=item.get("created_at", ""),
                        completed_at=item.get("completed_at")
                    )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Inference not found"
    )


@router.get("/record/{record_id}/history")
async def get_record_inference_history(
    record_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Get all inference history for a medical record"""
    record = db.query(MedicalRecord).filter(MedicalRecord.id == record_id).first()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Medical record not found"
        )
    
    return {
        "record_id": str(record_id),
        "inferences": record.infer_history or []
    }


@router.delete("/{inference_id}")
async def cancel_inference(
    inference_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Cancel a pending inference task"""
    # Find and update the inference status
    for record in db.query(MedicalRecord).all():
        if record.infer_history:
            updated = False
            new_history = []
            for item in record.infer_history:
                if item.get("id") == inference_id:
                    if item.get("status") == "pending":
                        item["status"] = "cancelled"
                        updated = True
                new_history.append(item)
            
            if updated:
                record.infer_history = new_history
                db.commit()
                
                # Try to revoke Celery task
                try:
                    from app.worker.celery_app import celery_app
                    celery_app.control.revoke(inference_id, terminate=True)
                except:
                    pass
                
                return {"message": "Inference cancelled"}
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Inference not found or already completed"
    )
