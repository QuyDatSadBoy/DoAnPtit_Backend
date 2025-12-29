"""
Statistics Endpoints
"""
import uuid
import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, extract

from app.db.session import get_db
from app.models.user import User
from app.models.patient import Patient
from app.models.medical_record import MedicalRecord
from app.models.doctor import Doctor
from app.schemas.inference import StatisticsResponse
from app.core.security import require_doctor, require_admin

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("", response_model=StatisticsResponse)
async def get_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Get system statistics"""
    try:
        today = date.today()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        # Basic counts
        total_patients = db.query(func.count(Patient.id)).scalar() or 0
        total_records = db.query(func.count(MedicalRecord.id)).scalar() or 0
        
        # Records by time period
        records_today = db.query(func.count(MedicalRecord.id)).filter(
            MedicalRecord.visit_date == today
        ).scalar() or 0
        
        records_this_week = db.query(func.count(MedicalRecord.id)).filter(
            MedicalRecord.visit_date >= week_ago
        ).scalar() or 0
        
        records_this_month = db.query(func.count(MedicalRecord.id)).filter(
            MedicalRecord.visit_date >= month_ago
        ).scalar() or 0
        
        # Count inferences from all records
        total_inferences = 0
        completed_inferences = 0
        pending_inferences = 0
        failed_inferences = 0
        
        for record in db.query(MedicalRecord).all():
            if record.infer_history:
                for item in record.infer_history:
                    total_inferences += 1
                    status = item.get("status", "")
                    if status == "completed":
                        completed_inferences += 1
                    elif status in ["pending", "processing"]:
                        pending_inferences += 1
                    elif status == "failed":
                        failed_inferences += 1
        
        # Records by month (last 6 months)
        records_by_month = []
        for i in range(5, -1, -1):
            month_start = (today.replace(day=1) - timedelta(days=30*i)).replace(day=1)
            if month_start.month == 12:
                month_end = month_start.replace(year=month_start.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = month_start.replace(month=month_start.month + 1, day=1) - timedelta(days=1)
            
            count = db.query(func.count(MedicalRecord.id)).filter(
                MedicalRecord.visit_date >= month_start,
                MedicalRecord.visit_date <= month_end
            ).scalar() or 0
            
            records_by_month.append({
                "month": month_start.strftime("%Y-%m"),
                "count": count
            })
        
        # Inferences by status
        inferences_by_status = {
            "completed": completed_inferences,
            "pending": pending_inferences,
            "failed": failed_inferences
        }
        
        # Top doctors by number of records
        top_doctors_query = db.query(
            Doctor.id,
            User.full_name,
            func.count(MedicalRecord.id).label("record_count")
        ).join(
            User, Doctor.user_id == User.id
        ).join(
            MedicalRecord, MedicalRecord.doctor_id == Doctor.id
        ).group_by(
            Doctor.id, User.full_name
        ).order_by(
            func.count(MedicalRecord.id).desc()
        ).limit(5).all()
        
        top_doctors = [
            {"name": d.full_name or "Unknown", "count": d.record_count}
            for d in top_doctors_query
        ]
        
        return StatisticsResponse(
            total_patients=total_patients or 0,
            total_records=total_records or 0,
            total_inferences=total_inferences,
            completed_inferences=completed_inferences,
            pending_inferences=pending_inferences,
            failed_inferences=failed_inferences,
            records_today=records_today or 0,
            records_this_week=records_this_week or 0,
            records_this_month=records_this_month or 0,
            records_by_month=records_by_month,
            inferences_by_status=inferences_by_status,
            top_doctors=top_doctors
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi lấy thống kê: {str(e)}")


@router.get("/dashboard")
async def get_dashboard_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Get dashboard summary data with role-based statistics"""
    try:
        today = date.today()
        is_admin = current_user.role == "admin"
        
        # DEBUG: Log current user info
        logger.info(f"[DEBUG] Dashboard - User ID: {current_user.id}")
        logger.info(f"[DEBUG] Dashboard - Username: {current_user.username}")
        logger.info(f"[DEBUG] Dashboard - Role: {current_user.role}")
        
        # Get doctor_id for current user (if doctor)
        current_doctor_id = None
        if current_user.role == "doctor":
            # Try to get existing doctor record
            if hasattr(current_user, 'doctor') and current_user.doctor:
                current_doctor_id = current_user.doctor.id
                logger.info(f"[DEBUG] Got doctor_id from user.doctor: {current_doctor_id}")
            else:
                doctor = db.query(Doctor).filter(Doctor.user_id == current_user.id).first()
                logger.info(f"[DEBUG] Query doctor by user_id: {doctor}")
                if doctor:
                    current_doctor_id = doctor.id
                    logger.info(f"[DEBUG] Found existing doctor: {current_doctor_id}")
                else:
                    # Auto-create doctor record if missing for doctor role users
                    logger.info(f"[DEBUG] Creating new doctor record for user {current_user.id}")
                    new_doctor = Doctor(
                        id=uuid.uuid4(),
                        user_id=current_user.id,
                        specialty=None,
                        phone=None,
                        hospital=None
                    )
                    db.add(new_doctor)
                    db.commit()
                    db.refresh(new_doctor)
                    current_doctor_id = new_doctor.id
                    logger.info(f"[DEBUG] Created new doctor: {current_doctor_id}")
        
        logger.info(f"[DEBUG] Final current_doctor_id: {current_doctor_id}")
        logger.info(f"[DEBUG] is_admin: {is_admin}")
        
        # ===== SYSTEM-WIDE STATISTICS (for admins) =====
        total_patients = db.query(func.count(Patient.id)).scalar() or 0
        total_records = db.query(func.count(MedicalRecord.id)).scalar() or 0
        
        # Count all inferences from all records
        total_inferences = 0
        completed_inferences = 0
        for record in db.query(MedicalRecord).all():
            if record.infer_history:
                for item in record.infer_history:
                    total_inferences += 1
                    if item.get("status") == "completed":
                        completed_inferences += 1
        
        # ===== DOCTOR-SPECIFIC STATISTICS =====
        logger.info(f"[DEBUG] Calculating doctor-specific stats...")
        
        if is_admin:
            # For admins, "my" stats = all stats
            my_patients = total_patients
            my_records = total_records
            my_inferences = total_inferences
            my_completed_inferences = completed_inferences
            logger.info(f"[DEBUG] Admin mode - using total stats")
        elif current_doctor_id:
            # For doctors, filter by their doctor_id
            my_patients = db.query(func.count(func.distinct(MedicalRecord.patient_id))).filter(
                MedicalRecord.doctor_id == current_doctor_id
            ).scalar() or 0
            
            my_records = db.query(func.count(MedicalRecord.id)).filter(
                MedicalRecord.doctor_id == current_doctor_id
            ).scalar() or 0
            
            # Count inferences for doctor's records only
            my_inferences = 0
            my_completed_inferences = 0
            for record in db.query(MedicalRecord).filter(MedicalRecord.doctor_id == current_doctor_id).all():
                if record.infer_history:
                    for item in record.infer_history:
                        my_inferences += 1
                        if item.get("status") == "completed":
                            my_completed_inferences += 1
            
            logger.info(f"[DEBUG] Doctor mode - my_patients={my_patients}, my_records={my_records}")
        else:
            my_patients = 0
            my_records = 0
            my_inferences = 0
            my_completed_inferences = 0
            logger.info(f"[DEBUG] No doctor_id found - returning zeros")
        
        # Recent records (filtered by role)
        if is_admin:
            recent_records = db.query(MedicalRecord).order_by(
                MedicalRecord.created_at.desc()
            ).limit(5).all()
        else:
            recent_records = db.query(MedicalRecord).filter(
                MedicalRecord.doctor_id == current_doctor_id
            ).order_by(
                MedicalRecord.created_at.desc()
            ).limit(5).all() if current_doctor_id else []
        
        # Recent patients (filtered by role)
        if is_admin:
            recent_patients = db.query(Patient).order_by(
                Patient.created_at.desc()
            ).limit(5).all()
        else:
            # Get patients that have medical records from this doctor
            patient_ids = db.query(MedicalRecord.patient_id).filter(
                MedicalRecord.doctor_id == current_doctor_id
            ).distinct().subquery()
            
            recent_patients = db.query(Patient).filter(
                Patient.id.in_(patient_ids)
            ).order_by(
                Patient.created_at.desc()
            ).limit(5).all() if current_doctor_id else []
        
        # Recent inferences (from recent records)
        recent_inferences = []
        source_records = recent_records if is_admin else db.query(MedicalRecord).filter(
            MedicalRecord.doctor_id == current_doctor_id
        ).order_by(MedicalRecord.created_at.desc()).limit(10).all() if current_doctor_id else []
        
        for record in source_records:
            if record.infer_history:
                # Get patient name
                patient = db.query(Patient).filter(Patient.id == record.patient_id).first()
                patient_name = patient.full_name if patient else None
                
                for item in record.infer_history:
                    recent_inferences.append({
                        "inference_id": item.get("id"),
                        "record_id": str(record.id),
                        "patient_id": str(record.patient_id),
                        "patient_name": patient_name,
                        "status": item.get("status"),
                        "created_at": item.get("created_at")
                    })
        
        # Sort by created_at and limit
        recent_inferences.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        recent_inferences = recent_inferences[:10]
        
        return {
            # System-wide stats
            "total_patients": total_patients,
            "total_records": total_records,
            "total_inferences": total_inferences,
            "completed_inferences": completed_inferences,
            # Doctor-specific stats
            "my_patients": my_patients,
            "my_records": my_records,
            "my_inferences": my_inferences,
            "my_completed_inferences": my_completed_inferences,
            # Recent data
            "recent_records": [
                {
                    "id": str(r.id),
                    "patient_id": str(r.patient_id),
                    "visit_date": str(r.visit_date),
                    "diagnosis": r.diagnosis
                }
                for r in recent_records
            ],
            "recent_patients": [
                {
                    "id": str(p.id),
                    "full_name": p.full_name,
                    "created_at": p.created_at.isoformat()
                }
                for p in recent_patients
            ],
            "recent_inferences": recent_inferences
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi lấy dữ liệu dashboard: {str(e)}")
