"""
Application Configuration
Tất cả config được lấy từ file .env
Khi deploy lên server, chỉ cần thay đổi các giá trị trong .env

QUAN TRỌNG:
- INTERNAL URL: FE gọi BE qua localhost (tốc độ cao)
- PUBLIC URL: URL public qua Kong (để hiển thị ảnh)
- DB chỉ lưu relative path (patient_files/xxx/yyy.png)
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # ===========================================
    # URL Configuration (QUAN TRỌNG)
    # ===========================================
    
    # INTERNAL: FE gọi BE qua localhost (tốc độ cao, không qua Kong)
    INTERNAL_BACKEND_HOST: str = "localhost"
    INTERNAL_BACKEND_PORT: int = 8999
    
    # PUBLIC: URL public qua Kong (để hiển thị ảnh, static files)
    # Ví dụ: https://api.yourdomain.com hoặc http://123.456.789.10:8999
    PUBLIC_BACKEND_URL: str = "http://localhost:8999"
    
    # Frontend (để CORS)
    FRONTEND_HOST: str = "localhost"
    FRONTEND_PORT: int = 3000
    
    # ===========================================
    # App
    # ===========================================
    APP_NAME: str = "Medical Imaging System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1/medical"
    
    # ===========================================
    # Security
    # ===========================================
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # ===========================================
    # Database - PostgreSQL
    # ===========================================
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "datsql09"
    POSTGRES_DB: str = "medical_imaging"
    DATABASE_URL: Optional[str] = None
    
    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # ===========================================
    # Redis
    # ===========================================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URL: Optional[str] = None
    
    @property
    def redis_url(self) -> str:
        if self.REDIS_URL:
            return self.REDIS_URL
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"
    
    # ===========================================
    # RabbitMQ
    # ===========================================
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5670
    RABBITMQ_USER: str = "user"
    RABBITMQ_PASSWORD: str = "adminace123"
    RABBITMQ_VHOST: str = "/"
    
    @property
    def CELERY_BROKER_URL(self) -> str:
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/{self.RABBITMQ_VHOST}"
    
    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return self.redis_url
    
    # ===========================================
    # CORS - Tự động generate từ FRONTEND_HOST và FRONTEND_PORT
    # ===========================================
    CORS_ORIGINS: Optional[List[str]] = None
    
    @property
    def cors_origins(self) -> List[str]:
        if self.CORS_ORIGINS:
            return self.CORS_ORIGINS
        # Auto generate từ frontend host/port
        origins = [
            f"http://{self.FRONTEND_HOST}:{self.FRONTEND_PORT}",
            f"http://127.0.0.1:{self.FRONTEND_PORT}",
            "http://localhost:3000",
            "http://localhost:8000",  # Kong gateway local
            "http://222.252.25.51:55555",  # Kong gateway public IP
            "http://tranquydat.com:55555",  # Kong gateway domain
            "https://tranquydat.com:55555",  # Kong gateway domain HTTPS
            "http://tranquydat.com",  # Domain without port
            "https://tranquydat.com",  # Domain HTTPS without port
             "http://localhost:5000"
        ]
        # Thêm https và domain nếu không phải localhost
        if self.FRONTEND_HOST not in ["localhost", "127.0.0.1"]:
            origins.append(f"https://{self.FRONTEND_HOST}:{self.FRONTEND_PORT}")
            origins.append(f"https://{self.FRONTEND_HOST}")
            origins.append(f"http://{self.FRONTEND_HOST}")
        return origins
    
    # ===========================================
    # File Storage (Relative paths - KHÔNG CÓ URL)
    # DB chỉ lưu: patient_files/xxx/yyy.png
    # ===========================================
    STATIC_DIR: str = "static"
    PATIENT_FILES_DIR: str = "patient_files"
    FACE_IMAGES_DIR: str = "face_images"
    UPLOADS_DIR: str = "uploads"
    AVATARS_DIR: str = "avatars"
    
    # ===========================================
    # Directories (relative to BASE_DIR)
    # ===========================================
    DATASET_DIR: str = "../dataset_LIDC_IDRI_filtered"
    DIFFDRR_DIR: str = "DoAnPtit_DiffDrr"
    CYCLEGAN_DIR: str = "DoAnPtit_CycleGan"
    XRAY2CT_DIR: str = "DoAnPtit_Xray2CT"
    RESULTS_DIR: str = "results"
    
    # ===========================================
    # Inference
    # ===========================================
    MODEL_CHECKPOINT: str = "DoAnPtit_Xray2CT/checkpoints/model-81.pt"
    
    # ===========================================
    # Computed URL properties
    # ===========================================
    @property
    def internal_backend_url(self) -> str:
        """Internal URL: FE gọi BE qua localhost (tốc độ cao)"""
        return f"http://{self.INTERNAL_BACKEND_HOST}:{self.INTERNAL_BACKEND_PORT}"
    
    @property
    def internal_api_url(self) -> str:
        """Internal API URL with prefix"""
        return f"{self.internal_backend_url}{self.API_V1_PREFIX}"
    
    @property
    def public_api_url(self) -> str:
        """Public API URL with prefix (qua Kong)"""
        return f"{self.PUBLIC_BACKEND_URL}{self.API_V1_PREFIX}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


# ===========================================
# Path helpers (để sử dụng trong các module khác)
# ===========================================
def get_base_dir() -> Path:
    """Get base directory of the backend project"""
    return Path(__file__).resolve().parent.parent.parent


def get_dataset_dir() -> Path:
    """Get dataset directory (LIDC-IDRI)"""
    base = get_base_dir()
    return (base / settings.DATASET_DIR).resolve()


def get_diffdrr_dir() -> Path:
    """Get DiffDRR directory"""
    return get_base_dir() / settings.DIFFDRR_DIR


def get_cyclegan_dir() -> Path:
    """Get CycleGAN directory"""
    return get_base_dir() / settings.CYCLEGAN_DIR


def get_xray2ct_dir() -> Path:
    """Get Xray2CT directory"""
    return get_base_dir() / settings.XRAY2CT_DIR


def get_results_dir() -> Path:
    """Get results directory"""
    results = get_base_dir() / settings.RESULTS_DIR
    results.mkdir(parents=True, exist_ok=True)
    return results


def get_patient_files_dir() -> Path:
    """Get patient files directory"""
    patient_dir = get_base_dir() / settings.PATIENT_FILES_DIR
    patient_dir.mkdir(parents=True, exist_ok=True)
    return patient_dir


def get_face_images_dir() -> Path:
    """Get face images directory"""
    face_dir = get_base_dir() / settings.FACE_IMAGES_DIR
    face_dir.mkdir(parents=True, exist_ok=True)
    return face_dir
