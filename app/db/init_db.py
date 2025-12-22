"""
Database initialization script
Creates tables and initial data
"""
import uuid
from sqlalchemy.orm import Session
from app.db.session import SessionLocal, engine, Base
from app.models.user import User
from app.models.doctor import Doctor
from app.core.security import get_password_hash


def create_tables():
    """Create all database tables"""
    from app.models import user, doctor, patient, medical_record
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")


def create_initial_admin(db: Session):
    """Create initial admin user if not exists"""
    admin = db.query(User).filter(User.username == "admin").first()
    if not admin:
        admin = User(
            id=uuid.uuid4(),
            username="admin",
            email="admin@medical.com",
            password_hash=get_password_hash("admin123"),
            role="admin",
            full_name="System Administrator",
            is_active=True
        )
        db.add(admin)
        db.commit()
        print("✅ Admin user created: admin / admin123")
    else:
        print("ℹ️ Admin user already exists")


def create_sample_doctor(db: Session):
    """Create sample doctor for testing"""
    doctor_user = db.query(User).filter(User.username == "doctor1").first()
    if not doctor_user:
        user_id = uuid.uuid4()
        doctor_user = User(
            id=user_id,
            username="doctor1",
            email="doctor1@medical.com",
            password_hash=get_password_hash("doctor123"),
            role="doctor",
            full_name="Dr. Nguyen Van A",
            is_active=True
        )
        db.add(doctor_user)
        db.flush()
        
        doctor = Doctor(
            id=uuid.uuid4(),
            user_id=user_id,
            specialty="Radiology",
            phone="0123456789",
            hospital="Bach Mai Hospital"
        )
        db.add(doctor)
        db.commit()
        print("✅ Sample doctor created: doctor1 / doctor123")
    else:
        print("ℹ️ Sample doctor already exists")


def init_database():
    """Initialize database with tables and initial data"""
    print("🔄 Initializing database...")
    create_tables()
    
    db = SessionLocal()
    try:
        create_initial_admin(db)
        create_sample_doctor(db)
    finally:
        db.close()
    
    print("✅ Database initialization complete!")


if __name__ == "__main__":
    init_database()
