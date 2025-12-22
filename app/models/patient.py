"""
Patient Model
"""
import uuid
from datetime import datetime, date
from sqlalchemy import Column, String, DateTime, Date, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.session import Base


class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(255), nullable=False)
    date_of_birth = Column(Date)
    gender = Column(String(20))
    phone = Column(String(50))
    address = Column(Text)
    email = Column(String(255))
    id_number = Column(String(50), unique=True, index=True)  # CCCD/CMND
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    medical_records = relationship("MedicalRecord", back_populates="patient", order_by="desc(MedicalRecord.visit_date)")
    
    def __repr__(self):
        return f"<Patient {self.full_name}>"
    
    @property
    def age(self):
        if self.date_of_birth:
            today = date.today()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None
