"""
Medical Imaging Backend - Main Application
FastAPI + Socket.IO + Celery
"""
import os
import socketio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings, get_base_dir, get_patient_files_dir, get_face_images_dir
from app.api.router import api_router
from app.db.session import init_db
from app.db.init_db import init_database
from app.socket.manager import sio, start_redis_listener, stop_redis_listener

# Get base directory for absolute paths
BASE_DIR = get_base_dir()

# Create required directories with ABSOLUTE paths
STATIC_DIR = BASE_DIR / settings.STATIC_DIR
PATIENT_FILES_DIR = get_patient_files_dir()
AVATARS_DIR = BASE_DIR / settings.AVATARS_DIR
FACE_IMAGES_DIR = get_face_images_dir()

STATIC_DIR.mkdir(parents=True, exist_ok=True)
PATIENT_FILES_DIR.mkdir(parents=True, exist_ok=True)
AVATARS_DIR.mkdir(parents=True, exist_ok=True)
FACE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 Starting Medical Imaging Backend...")
    
    # Initialize database
    try:
        init_database()
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")
    
    # Start Redis listener for Socket.IO
    try:
        start_redis_listener()
        print("📡 Socket.IO Redis listener started")
    except Exception as e:
        print(f"⚠️ Redis listener warning: {e}")
    
    print("✅ Application started successfully!")
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")
    try:
        await stop_redis_listener()
    except Exception as e:
        print(f"⚠️ Redis listener stop warning: {e}")


# Create FastAPI app
# Dùng root_path để Kong hoạt động chuẩn với strip_path=true
# Static files sẽ được FE gọi trực tiếp qua REACT_APP_PUBLIC_BACKEND_URL (port 8999)
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Medical Imaging System - X-ray to CT Conversion",
    lifespan=lifespan,
    root_path=settings.API_V1_PREFIX,  # Kong strip_path=true sẽ strip /api/v1/medical
    # redirect_slashes=True (mặc định) - FE phải gọi ĐÚNG URL không có trailing slash
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes - KHÔNG dùng prefix vì đã có root_path
app.include_router(api_router)


# Health Check & Root endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "medical-imaging-api"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "api": settings.API_V1_PREFIX,
    }


# Mount static files - TRƯỚC khi wrap với Socket.IO
# Static files sẽ ở root path, không bị ảnh hưởng bởi root_path của API
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/patient_files", StaticFiles(directory=str(PATIENT_FILES_DIR)), name="patient_files")
app.mount("/avatars", StaticFiles(directory=str(AVATARS_DIR)), name="avatars")
app.mount("/face_images", StaticFiles(directory=str(FACE_IMAGES_DIR)), name="face_images")


# Socket.IO Integration - giống notification service
socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path="socket.io",
)

# Mount SocketIO app - giống notification service
app = socket_app


# Export for uvicorn
application = app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:application",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
