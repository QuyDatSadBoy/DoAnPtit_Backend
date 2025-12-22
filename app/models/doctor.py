"""
Doctor Model
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.session import Base


class Doctor(Base):
    __tablename__ = "doctors"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)
    specialty = Column(String(255))
    phone = Column(String(50))
    hospital = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="doctor")
    medical_records = relationship("MedicalRecord", back_populates="doctor")
    
    def __repr__(self):
        return f"<Doctor {self.user.full_name if self.user else self.id}>"
