"""
User Model
"""
import uuid
from sqlalchemy import Column, String, Boolean, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from app.db.session import Base
from app.core.timezone import now_vn


class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(50), nullable=False)  # admin | doctor
    full_name = Column(String(255))
    avatar = Column(String(500), nullable=True)  # Path to avatar image
    phone = Column(String(20), nullable=True)  # Phone number for all users
    
    # Face Recognition fields
    face_images_folder = Column(String(500), nullable=True)  # Path to folder containing face images
    face_encoding = Column(Text, nullable=True)  # JSON string of face encoding(s)
    face_registered = Column(Boolean, default=False, nullable=False)  # Whether face is registered
    face_registered_at = Column(DateTime, nullable=True)  # When face was registered
    
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=now_vn, nullable=False)
    updated_at = Column(DateTime, default=now_vn, onupdate=now_vn, nullable=False)
    
    # Relationships
    doctor = relationship("Doctor", back_populates="user", uselist=False)
    created_medical_records = relationship(
        "MedicalRecord", 
        back_populates="created_by_user",
        foreign_keys="MedicalRecord.created_by"
    )
    
    def __repr__(self):
        return f"<User {self.username}>"
