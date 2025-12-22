"""
Statistics Endpoints
"""
from datetime import datetime, timedelta, date
from typing import Dict, Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, extract

from app.db.session import get_db
from app.models.user import User
from app.models.patient import Patient
from app.models.medical_record import MedicalRecord
from app.models.doctor import Doctor
from app.schemas.inference import StatisticsResponse
from app.core.security import require_doctor, require_admin

router = APIRouter()


@router.get("/", response_model=StatisticsResponse)
async def get_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Get system statistics"""
    today = date.today()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    # Basic counts
    total_patients = db.query(func.count(Patient.id)).scalar()
    total_records = db.query(func.count(MedicalRecord.id)).scalar()
    
    # Records by time period
    records_today = db.query(func.count(MedicalRecord.id)).filter(
        MedicalRecord.visit_date == today
    ).scalar()
    
    records_this_week = db.query(func.count(MedicalRecord.id)).filter(
        MedicalRecord.visit_date >= week_ago
    ).scalar()
    
    records_this_month = db.query(func.count(MedicalRecord.id)).filter(
        MedicalRecord.visit_date >= month_ago
    ).scalar()
    
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
        ).scalar()
        
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


@router.get("/dashboard")
async def get_dashboard_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """Get dashboard summary data"""
    today = date.today()
    
    # Recent records
    recent_records = db.query(MedicalRecord).order_by(
        MedicalRecord.created_at.desc()
    ).limit(5).all()
    
    # Recent patients
    recent_patients = db.query(Patient).order_by(
        Patient.created_at.desc()
    ).limit(5).all()
    
    # Pending inferences
    pending = []
    for record in db.query(MedicalRecord).all():
        if record.infer_history:
            for item in record.infer_history:
                if item.get("status") in ["pending", "processing"]:
                    pending.append({
                        "inference_id": item.get("id"),
                        "record_id": str(record.id),
                        "patient_id": str(record.patient_id),
                        "status": item.get("status"),
                        "created_at": item.get("created_at")
                    })
    
    return {
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
        "pending_inferences": pending[:10]
    }
