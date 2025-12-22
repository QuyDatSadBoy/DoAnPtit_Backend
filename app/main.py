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

from app.core.config import settings
from app.api.router import api_router
from app.db.session import init_db
from app.db.init_db import init_database
from app.socket.manager import sio, start_redis_listener


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
    
    # Create directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.STATIC_DIR, exist_ok=True)
    os.makedirs(settings.PATIENT_FILES_DIR, exist_ok=True)
    
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


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Medical Imaging System - X-ray to CT Conversion",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Mount static files
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
app.mount("/patient_files", StaticFiles(directory=settings.PATIENT_FILES_DIR), name="patient_files")

# Create Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio, app)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "api": settings.API_V1_PREFIX
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# Export for uvicorn
application = socket_app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:application",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
