#!/usr/bin/env python3
"""
Database Setup Script
Drops existing database, creates new one, and seeds with sample data.

Usage:
    python setup_database.py

WARNING: This will DELETE all existing data!
"""

import os
import sys
import uuid
import shutil
from datetime import datetime, date, timedelta
import random

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "datsql09")
POSTGRES_DB = os.getenv("POSTGRES_DB", "medical_imaging")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Sample files paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_XRAY_PNG = os.path.join(BASE_DIR, "DoAnPtit_Xray2CT", "x-ray input", "LIDC-IDRI-0001.png")
SAMPLE_XRAY_NPY = os.path.join(BASE_DIR, "DoAnPtit_Xray2CT", "x-ray input", "LIDC-IDRI-0001.npy")
SAMPLE_CT_NIFTI = os.path.join(BASE_DIR, "DoAnPtit_Xray2CT", "results", "LIDC-IDRI-0001_none_style.nii.gz")
SAMPLE_CT_GIF = os.path.join(BASE_DIR, "DoAnPtit_Xray2CT", "results", "LIDC-IDRI-0001_none_style.gif")
PATIENT_FILES_DIR = os.path.join(BASE_DIR, "patient_files")


def setup_sample_files(patient_id: str, record_id: str, inference_id: str, status: str = "completed"):
    """
    Create sample files for fake inference history
    Returns dict with RELATIVE file paths (for serving via FastAPI static files)
    """
    # Create directory structure: patient_files/{patient_id}/{record_id}/
    folder_path = os.path.join(PATIENT_FILES_DIR, str(patient_id), str(record_id))
    os.makedirs(folder_path, exist_ok=True)
    
    # Relative path prefix for database storage
    relative_prefix = f"patient_files/{patient_id}/{record_id}"
    
    # Copy X-ray file
    xray_filename = f"xray_{inference_id}.png"
    xray_full_path = os.path.join(folder_path, xray_filename)
    xray_relative_path = f"{relative_prefix}/{xray_filename}"
    
    if os.path.exists(SAMPLE_XRAY_PNG):
        shutil.copy2(SAMPLE_XRAY_PNG, xray_full_path)
    
    # Also copy NPY version
    xray_npy_filename = f"xray_{inference_id}.npy"
    xray_npy_path = os.path.join(folder_path, xray_npy_filename)
    if os.path.exists(SAMPLE_XRAY_NPY):
        shutil.copy2(SAMPLE_XRAY_NPY, xray_npy_path)
    
    ct_relative_path = None
    ct_gif_path = None
    
    if status == "completed":
        # Copy CT NIfTI file
        ct_filename = f"ct_{inference_id}_none_style.nii.gz"
        ct_full_path = os.path.join(folder_path, ct_filename)
        ct_relative_path = f"{relative_prefix}/{ct_filename}"
        if os.path.exists(SAMPLE_CT_NIFTI):
            shutil.copy2(SAMPLE_CT_NIFTI, ct_full_path)
        
        # Copy CT GIF preview
        ct_gif_filename = f"ct_{inference_id}_none_style.gif"
        ct_gif_full_path = os.path.join(folder_path, ct_gif_filename)
        if os.path.exists(SAMPLE_CT_GIF):
            shutil.copy2(SAMPLE_CT_GIF, ct_gif_full_path)
    
    # Return RELATIVE paths for database storage
    return {
        "xray_path": xray_relative_path,
        "xray_npy_path": f"{relative_prefix}/{xray_npy_filename}",
        "ct_path": ct_relative_path,
        "ct_gif_path": f"{relative_prefix}/{ct_gif_filename}" if status == "completed" else None
    }


def clean_patient_files():
    """Remove existing patient_files directory"""
    if os.path.exists(PATIENT_FILES_DIR):
        print(f"🗑️  Cleaning patient_files directory...")
        shutil.rmtree(PATIENT_FILES_DIR)
    os.makedirs(PATIENT_FILES_DIR, exist_ok=True)
    print(f"📁 Created fresh patient_files directory")


def drop_and_create_database():
    """Drop existing database and create new one"""
    print("=" * 60)
    print("🗄️  Database Setup Script")
    print("=" * 60)
    
    # Connect to PostgreSQL server (not specific database)
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Terminate existing connections to the database
    print(f"\n🔌 Terminating connections to '{POSTGRES_DB}'...")
    cursor.execute(f"""
        SELECT pg_terminate_backend(pg_stat_activity.pid)
        FROM pg_stat_activity
        WHERE pg_stat_activity.datname = '{POSTGRES_DB}'
        AND pid <> pg_backend_pid();
    """)
    
    # Drop database if exists
    print(f"🗑️  Dropping database '{POSTGRES_DB}' if exists...")
    cursor.execute(f"DROP DATABASE IF EXISTS {POSTGRES_DB}")
    
    # Create database
    print(f"✨ Creating database '{POSTGRES_DB}'...")
    cursor.execute(f"CREATE DATABASE {POSTGRES_DB}")
    
    cursor.close()
    conn.close()
    
    print(f"✅ Database '{POSTGRES_DB}' created successfully!")


def create_tables():
    """Create all database tables"""
    print("\n📋 Creating tables...")
    
    from app.db.session import Base, engine
    from app.models import user, doctor, patient, medical_record
    
    Base.metadata.create_all(bind=engine)
    print("✅ All tables created!")


def seed_data():
    """Seed database with sample data"""
    print("\n🌱 Seeding sample data...")
    
    from app.db.session import SessionLocal
    from app.models.user import User
    from app.models.doctor import Doctor
    from app.models.patient import Patient
    from app.models.medical_record import MedicalRecord
    from app.core.security import get_password_hash
    
    db = SessionLocal()
    
    try:
        # ==================== USERS ====================
        print("   👤 Creating users...")
        
        # Admin user
        admin_id = uuid.uuid4()
        admin = User(
            id=admin_id,
            username="admin",
            email="admin@medical.com",
            password_hash=get_password_hash("admin123"),
            role="admin",
            full_name="Quản Trị Viên",
            phone="0900000000",
            avatar=None,  # Admin starts without avatar
            is_active=True
        )
        db.add(admin)
        
        # Doctor users
        doctors_data = [
            {"username": "doctor1", "email": "doctor1@medical.com", "full_name": "BS. Nguyễn Văn An", "specialty": "Chẩn đoán hình ảnh", "phone": "0901234567", "hospital": "Bệnh viện Bạch Mai"},
            {"username": "doctor2", "email": "doctor2@medical.com", "full_name": "BS. Trần Thị Bình", "specialty": "Nội khoa", "phone": "0902345678", "hospital": "Bệnh viện Việt Đức"},
            {"username": "doctor3", "email": "doctor3@medical.com", "full_name": "BS. Lê Hoàng Cường", "specialty": "Phổi", "phone": "0903456789", "hospital": "Bệnh viện 108"},
            {"username": "doctor4", "email": "doctor4@medical.com", "full_name": "BS. Phạm Minh Đức", "specialty": "Tim mạch", "phone": "0904567890", "hospital": "Bệnh viện Tim Hà Nội"},
            {"username": "doctor5", "email": "doctor5@medical.com", "full_name": "BS. Hoàng Thị Lan", "specialty": "Chẩn đoán hình ảnh", "phone": "0905678901", "hospital": "Bệnh viện E"},
        ]
        
        doctor_ids = []
        for doc_data in doctors_data:
            user_id = uuid.uuid4()
            user = User(
                id=user_id,
                username=doc_data["username"],
                email=doc_data["email"],
                password_hash=get_password_hash("doctor123"),
                role="doctor",
                full_name=doc_data["full_name"],
                phone=doc_data["phone"],
                avatar=None,  # Doctors start without avatar
                is_active=True
            )
            db.add(user)
            db.flush()
            
            doctor = Doctor(
                id=uuid.uuid4(),
                user_id=user_id,
                specialty=doc_data["specialty"],
                phone=doc_data["phone"],
                hospital=doc_data["hospital"]
            )
            db.add(doctor)
            doctor_ids.append(doctor.id)
        
        db.flush()
        print(f"   ✅ Created 1 admin + {len(doctors_data)} doctors")
        
        # ==================== PATIENTS ====================
        print("   🏥 Creating patients...")
        
        patients_data = [
            {"full_name": "Nguyễn Văn Hùng", "date_of_birth": date(1985, 3, 15), "gender": "male", "phone": "0911111111", "email": "hung.nguyen@email.com", "address": "123 Phố Huế, Hà Nội", "id_number": "001085012345"},
            {"full_name": "Trần Thị Mai", "date_of_birth": date(1990, 7, 22), "gender": "female", "phone": "0922222222", "email": "mai.tran@email.com", "address": "45 Láng Hạ, Hà Nội", "id_number": "001090023456"},
            {"full_name": "Lê Minh Tuấn", "date_of_birth": date(1978, 11, 8), "gender": "male", "phone": "0933333333", "email": "tuan.le@email.com", "address": "78 Nguyễn Trãi, Hà Nội", "id_number": "001078034567"},
            {"full_name": "Phạm Thị Hương", "date_of_birth": date(1995, 5, 30), "gender": "female", "phone": "0944444444", "email": "huong.pham@email.com", "address": "90 Cầu Giấy, Hà Nội", "id_number": "001095045678"},
            {"full_name": "Hoàng Văn Nam", "date_of_birth": date(1982, 9, 12), "gender": "male", "phone": "0955555555", "email": "nam.hoang@email.com", "address": "15 Đội Cấn, Hà Nội", "id_number": "001082056789"},
            {"full_name": "Vũ Thị Lan Anh", "date_of_birth": date(1988, 1, 25), "gender": "female", "phone": "0966666666", "email": "lananh.vu@email.com", "address": "200 Trường Chinh, Hà Nội", "id_number": "001088067890"},
            {"full_name": "Đỗ Quang Minh", "date_of_birth": date(1975, 4, 18), "gender": "male", "phone": "0977777777", "email": "minh.do@email.com", "address": "55 Kim Mã, Hà Nội", "id_number": "001075078901"},
            {"full_name": "Bùi Thị Ngọc", "date_of_birth": date(1992, 8, 5), "gender": "female", "phone": "0988888888", "email": "ngoc.bui@email.com", "address": "88 Thái Hà, Hà Nội", "id_number": "001092089012"},
            {"full_name": "Ngô Văn Thành", "date_of_birth": date(1980, 12, 20), "gender": "male", "phone": "0999999999", "email": "thanh.ngo@email.com", "address": "120 Giải Phóng, Hà Nội", "id_number": "001080090123"},
            {"full_name": "Đinh Thị Phương", "date_of_birth": date(1998, 2, 14), "gender": "female", "phone": "0910101010", "email": "phuong.dinh@email.com", "address": "35 Hoàng Hoa Thám, Hà Nội", "id_number": "001098101234"},
            {"full_name": "Lý Văn Khánh", "date_of_birth": date(1970, 6, 28), "gender": "male", "phone": "0912121212", "email": "khanh.ly@email.com", "address": "67 Láng, Hà Nội", "id_number": "001070112345"},
            {"full_name": "Đặng Thị Thảo", "date_of_birth": date(1987, 10, 3), "gender": "female", "phone": "0913131313", "email": "thao.dang@email.com", "address": "99 Xuân Thủy, Hà Nội", "id_number": "001087123456"},
        ]
        
        patient_ids = []
        for pat_data in patients_data:
            patient = Patient(
                id=uuid.uuid4(),
                full_name=pat_data["full_name"],
                date_of_birth=pat_data["date_of_birth"],
                gender=pat_data["gender"],
                phone=pat_data["phone"],
                email=pat_data["email"],
                address=pat_data["address"],
                id_number=pat_data["id_number"]
            )
            db.add(patient)
            patient_ids.append(patient.id)
        
        db.flush()
        print(f"   ✅ Created {len(patients_data)} patients")
        
        # ==================== MEDICAL RECORDS ====================
        print("   📋 Creating medical records...")
        
        diagnoses = [
            "Viêm phổi cấp tính",
            "U phổi nghi ngờ, cần theo dõi",
            "Xẹp phổi một phần",
            "Tràn dịch màng phổi nhẹ",
            "Khối u trung thất",
            "Viêm phế quản mạn tính",
            "Giãn phế quản",
            "Lao phổi nghi ngờ",
            "Phổi bình thường, không có bất thường",
            "Nốt đơn độc phổi, theo dõi định kỳ",
        ]
        
        symptoms_list = [
            "Ho kéo dài, khó thở, sốt nhẹ",
            "Đau ngực, ho ra máu",
            "Khó thở khi gắng sức, mệt mỏi",
            "Ho khan, sụt cân không rõ nguyên nhân",
            "Sốt về chiều, đổ mồ hôi đêm",
            "Đau ngực khi hít thở sâu",
            "Ho có đờm, khó thở",
            "Tức ngực, khó chịu vùng ngực",
        ]
        
        records_created = 0
        for i, patient_id in enumerate(patient_ids):
            # Each patient has 1-3 medical records
            num_records = random.randint(1, 3)
            
            for j in range(num_records):
                doctor_id = random.choice(doctor_ids)
                visit_date = date.today() - timedelta(days=random.randint(1, 180))
                
                # Generate record ID upfront
                record_id = uuid.uuid4()
                
                # Some records have inference history
                infer_history = []
                if random.random() > 0.4:  # 60% chance to have inference
                    num_inferences = random.randint(1, 2)
                    for k in range(num_inferences):
                        status = random.choice(["completed", "completed", "completed", "failed"])
                        inference_id = str(uuid.uuid4())
                        
                        # Create actual sample files
                        file_paths = setup_sample_files(
                            patient_id=str(patient_id),
                            record_id=str(record_id),
                            inference_id=inference_id,
                            status=status
                        )
                        
                        created_time = datetime.now() - timedelta(days=random.randint(1, 30), hours=random.randint(0, 23))
                        completed_time = created_time + timedelta(minutes=random.randint(2, 5)) if status == "completed" else None
                        
                        inference = {
                            "id": inference_id,
                            "xray_path": file_paths["xray_path"],
                            "ct_path": file_paths["ct_path"],
                            "status": status,
                            "guidance_scale": round(random.uniform(0.8, 1.2), 2),
                            "created_at": created_time.isoformat(),
                            "completed_at": completed_time.isoformat() if completed_time else None,
                            "error": "GPU out of memory error - please try again later" if status == "failed" else None,
                            "progress": 100 if status == "completed" else None
                        }
                        infer_history.append(inference)
                
                record = MedicalRecord(
                    id=record_id,
                    patient_id=patient_id,
                    doctor_id=doctor_id,
                    created_by=admin_id,  # All created by admin for simplicity
                    visit_date=visit_date,
                    diagnosis=random.choice(diagnoses),
                    symptoms=random.choice(symptoms_list),
                    notes=f"Ghi chú khám bệnh lần {j+1}. Bệnh nhân cần tái khám sau 2 tuần.",
                    infer_history=infer_history
                )
                db.add(record)
                records_created += 1
        
        db.flush()
        print(f"   ✅ Created {records_created} medical records")
        
        # Count inference files
        total_inferences = sum(1 for root, dirs, files in os.walk(PATIENT_FILES_DIR) for f in files if f.startswith('xray_'))
        print(f"   📁 Created {total_inferences} inference sample files")
        
        # Commit all changes
        db.commit()
        
        print("\n" + "=" * 60)
        print("🎉 Database setup completed successfully!")
        print("=" * 60)
        print("\n📊 Summary:")
        print(f"   • Users: 1 admin + {len(doctors_data)} doctors")
        print(f"   • Patients: {len(patients_data)}")
        print(f"   • Medical Records: {records_created}")
        print(f"   • Inference samples: {total_inferences}")
        print(f"\n📁 Sample files location: {PATIENT_FILES_DIR}")
        print("\n🔐 Default Credentials:")
        print("   • Admin: admin / admin123")
        print("   • Doctors: doctor1-5 / doctor123")
        print("\n✅ You can now start the application!")
        
    except Exception as e:
        db.rollback()
        print(f"\n❌ Error seeding data: {e}")
        raise
    finally:
        db.close()


def main():
    """Main function"""
    print("\n⚠️  WARNING: This will DELETE all existing data in the database!")
    print("=" * 60)
    
    # Auto-confirm when running in non-interactive mode
    if sys.stdin.isatty():
        confirm = input("\nAre you sure you want to continue? (yes/no): ").strip().lower()
        if confirm not in ["yes", "y"]:
            print("\n❌ Operation cancelled.")
            sys.exit(0)
    else:
        print("\n🔄 Running in non-interactive mode, auto-confirming...")
    
    try:
        # Step 1: Drop and create database
        drop_and_create_database()
        
        # Step 2: Create tables
        create_tables()
        
        # Step 3: Clean patient_files directory
        clean_patient_files()
        
        # Step 4: Seed data (this also creates sample files)
        seed_data()
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
