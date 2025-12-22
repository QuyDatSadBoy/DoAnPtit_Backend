"""
Medical Record Model
"""
import uuid
from datetime import datetime, date
from sqlalchemy import Column, String, DateTime, Date, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.db.session import Base


class MedicalRecord(Base):
    __tablename__ = "medical_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False, index=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    visit_date = Column(Date, nullable=False, default=date.today)
    diagnosis = Column(Text)
    symptoms = Column(Text)
    notes = Column(Text)
    
    # JSON field to store inference history
    # Format: [{"id": "uuid", "xray_path": "...", "ct_path": "...", "status": "pending|processing|completed|failed", "created_at": "..."}]
    infer_history = Column(JSONB, default=list, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    patient = relationship("Patient", back_populates="medical_records")
    doctor = relationship("Doctor", back_populates="medical_records")
    created_by_user = relationship("User", back_populates="created_medical_records")
    
    def __repr__(self):
        return f"<MedicalRecord {self.id} - Patient: {self.patient_id}>"
    
    def add_inference(self, inference_id: str, xray_path: str):
        """Add new inference entry to history"""
        if self.infer_history is None:
            self.infer_history = []
        
        self.infer_history.append({
            "id": inference_id,
            "xray_path": xray_path,
            "ct_path": None,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None
        })
    
    def update_inference_status(self, inference_id: str, status: str, ct_path: str = None):
        """Update inference status"""
        if self.infer_history:
            for item in self.infer_history:
                if item["id"] == inference_id:
                    item["status"] = status
                    if ct_path:
                        item["ct_path"] = ct_path
                    if status == "completed":
                        item["completed_at"] = datetime.utcnow().isoformat()
                    break
