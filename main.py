#!/usr/bin/env python3
"""
FastAPI Backend for X-ray2CT
"""

import os
import json
import uuid
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import torch
import SimpleITK as sitk
from PIL import Image
import base64
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make inference import optional for demo mode
try:
    from inference import XrayToCTPAInference
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("⚠️ Inference module not available - running in demo mode")
    # Create dummy class for type annotation
    class XrayToCTPAInference:
        pass

# Pydantic models
class GenerateRequest(BaseModel):
    session_id: str
    guidance_scale: float = 1.0

class WindowingRequest(BaseModel):
    session_id: str
    window_center: float = 50
    window_width: float = 350

class NiftiUploadResponse(BaseModel):
    success: bool
    session_id: str
    filename: str
    shape: List[int]
    num_slices: int
    spacing: List[float]
    origin: List[float]
    min_value: float
    max_value: float
    message: str

class SliceResponse(BaseModel):
    success: bool
    slice_data: str  # Base64 encoded image
    slice_idx: int
    total_slices: int
    metadata: Dict[str, Any]

class GenerateResponse(BaseModel):
    success: bool
    session_id: str
    num_slices: int
    shape: List[int]
    message: str
    processing_time: float

# Add new Pydantic models for evaluation
class EvaluationRequest(BaseModel):
    gt_session_id: str
    pred_session_id: str
    create_viz: bool = True
    language: str = "en"  # "en" or "vi"

class EvaluationResponse(BaseModel):
    success: bool
    evaluation_id: str
    metrics: Dict[str, float]
    interpretation: Dict[str, str]
    visualizations: Optional[Dict[str, Any]]
    readme_content: str
    message: str

# FastAPI app
app = FastAPI(
    title="X-ray2CT API",
    description="API to convert 2D X-ray to 3D CT with visualization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo thư mục cần thiết
# Determine if we're running from backend directory or project root
current_dir = Path.cwd()
if current_dir.name == "backend":
    # Running from backend directory
    UPLOAD_FOLDER = Path("uploads").resolve()
    RESULTS_FOLDER = Path("results").resolve()
    STATIC_FOLDER = Path("static").resolve()
else:
    # Running from project root
    UPLOAD_FOLDER = Path("backend/uploads").resolve()
    RESULTS_FOLDER = Path("backend/results").resolve()
    STATIC_FOLDER = Path("backend/static").resolve()

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, STATIC_FOLDER]:
    folder.mkdir(exist_ok=True)

# Debug: Print the actual paths being used
print(f"📁 Working directory: {Path.cwd()}")
print(f"📁 Upload folder: {UPLOAD_FOLDER}")
print(f"📁 Results folder: {RESULTS_FOLDER}")
print(f"📁 Static folder: {STATIC_FOLDER}")

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_FOLDER)), name="static")

# Global inference model
inference_model: Optional[XrayToCTPAInference] = None

def init_model(checkpoint_path: str) -> bool:
    """Initialize inference model"""
    global inference_model
    
    # If checkpoint_path is "demo" or "none", skip model initialization
    if checkpoint_path.lower() in ["demo", "none", "dummy"]:
        print("🎭 Running in DEMO MODE - no model loaded")
        return True
    
    if not INFERENCE_AVAILABLE:
        print("⚠️ Inference module not available - cannot load model")
        return False
    
    model_config = {
        'denoising_fn': 'Unet3D',
        'diffusion_img_size': 32,
        'diffusion_depth_size': 128,  # Updated from ddpm2.yaml
        'diffusion_num_channels': 4,
        'dim_mults': [1, 2, 4, 8],
        'cond_dim': 512,
        'timesteps': 1000,
        'loss_type': 'l1_lpips',  # Updated from ddpm2.yaml
        'l1_weight': 1.0,
        'perceptual_weight': 0.01,  # Updated from ddpm2.yaml
        'discriminator_weight': 0.0,
        'classification_weight': 0.0,
        'classifier_free_guidance': False,
        'medclip': True,
        'name_dataset': 'LIDC',  # Updated from ddpm2.yaml
        'dataset_min_value': -12.911299,  # Updated from ddmp2.yaml
        'dataset_max_value': 9.596558,   # Updated from ddpm2.yaml
        'vae_ckpt': None,  # Updated from ddpm2.yaml
        'vqgan_ckpt': None
    }
    
    try:
        inference_model = XrayToCTPAInference(
            model_checkpoint=checkpoint_path,
            model_config=model_config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✅ Model initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False

def allowed_file(filename: str) -> bool:
    """Check if file is allowed for upload"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'npy', 'dcm', 'nii', 'gz'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_nifti_file(file_path: str, session_id: str) -> tuple[np.ndarray, dict]:
    """
    Process NIfTI file and extract metadata
    
    Args:
        file_path: Path to NIfTI file
        session_id: Session ID for tracking
    
    Returns:
        tuple: (volume_array, metadata_dict)
    """
    # Đọc file NIfTI
    nifti_image = sitk.ReadImage(file_path)
    volume_array = sitk.GetArrayFromImage(nifti_image)
    
    # Trích xuất metadata
    metadata = {
        'shape': list(volume_array.shape),
        'spacing': list(nifti_image.GetSpacing()),
        'origin': list(nifti_image.GetOrigin()),
        'direction': list(nifti_image.GetDirection()),
        'min_value': float(volume_array.min()),
        'max_value': float(volume_array.max()),
        'mean_value': float(volume_array.mean()),
        'std_value': float(volume_array.std()),
        'data_type': str(volume_array.dtype),
        'session_id': session_id
    }
    
    print(f"📊 NIfTI metadata: {metadata}")
    
    return volume_array, metadata

def process_nifti_slices(volume_array: np.ndarray, session_id: str) -> tuple[List[str], int]:
    """
    Process NIfTI volume into slices for web display
    
    Args:
        volume_array: Volume array from NIfTI
        session_id: Session ID for tracking
    
    Returns:
        tuple: (slice_paths, num_slices)
    """
    
    # Normalize volume for visualization
    volume_normalized = (volume_array - volume_array.min()) / (volume_array.max() - volume_array.min())
    
    # Create directory for session
    session_dir = RESULTS_FOLDER / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Process different axes
    # Default to axial view
    plane = 'axial'
    
    if plane == 'axial':
        # Axial: slice along axis 0 (depth)
        slices = volume_normalized
        num_slices = slices.shape[0]
    elif plane == 'sagittal':
        # Sagittal: slice along axis 2 (width)
        slices = volume_normalized.transpose(2, 1, 0)
        num_slices = slices.shape[0]
    elif plane == 'coronal':
        # Coronal: slice along axis 1 (height)
        slices = volume_normalized.transpose(1, 0, 2)
        num_slices = slices.shape[0]
    
    slice_paths = []
    for i in range(num_slices):
        slice_data = slices[i]
        
        # Convert to 8-bit image
        slice_8bit = (slice_data * 255).astype(np.uint8)
        
        # Save slice
        slice_filename = f"slice_{i:03d}.png"
        slice_path = session_dir / slice_filename
        cv2.imwrite(str(slice_path), slice_8bit)
        
        slice_paths.append(f"/static/{session_id}/{slice_filename}")
    
    return slice_paths, num_slices

def process_ctpa_slices(ctpa_volume: np.ndarray, session_id: str) -> tuple[List[str], int]:
    """Process CTPA volume into slices for display"""
    
    # Create directory for session
    session_dir = RESULTS_FOLDER / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all slices
    slice_paths = []
    num_slices = ctpa_volume.shape[0]
    
    for i in range(num_slices):
        slice_data = ctpa_volume[i]
        
        # Convert to 8-bit image
        slice_8bit = (slice_data * 255).astype(np.uint8)
        
        # Lưu slice
        slice_filename = f"slice_{i:03d}.png"
        slice_path = session_dir / slice_filename
        cv2.imwrite(str(slice_path), slice_8bit)
        
        slice_paths.append(f"/static/{session_id}/{slice_filename}")
    
    return slice_paths, num_slices

def apply_medical_windowing(image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Apply medical imaging windowing"""
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    windowed = np.clip(image, img_min, img_max)
    # Normalize to 0-255
    windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return windowed

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_str}"

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "X-ray2CT API is running",
        "model_loaded": inference_model is not None,
        "demo_mode": not INFERENCE_AVAILABLE,
        "inference_available": INFERENCE_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "X-ray2CT API is running",
        "model_loaded": inference_model is not None,
        "demo_mode": not INFERENCE_AVAILABLE,
        "inference_available": INFERENCE_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload X-ray file"""
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # Tạo session ID unique
    session_id = str(uuid.uuid4())
    
    # Lưu file
    file_extension = file.filename.split('.')[-1]
    filename = f"{session_id}.{file_extension}"
    file_path = UPLOAD_FOLDER / filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "success": True,
            "session_id": session_id,
            "filename": filename,
            "original_filename": file.filename,
            "file_size": file_path.stat().st_size,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

@app.post("/api/upload")
async def api_upload_file(file: UploadFile = File(...)):
    """API endpoint for uploading X-ray files"""
    return await upload_file(file)

@app.post("/api/upload-nifti", response_model=NiftiUploadResponse)
async def upload_nifti_file(file: UploadFile = File(...)):
    """Upload and process NIfTI (.nii.gz) file"""
    if not file.filename.endswith(('.nii', '.nii.gz')):
        raise HTTPException(status_code=400, detail="Only NIfTI files (.nii, .nii.gz) are allowed")
    
    # Tạo session ID unique
    session_id = str(uuid.uuid4())
    
    # Lưu file
    if file.filename.endswith('.nii.gz'):
        filename = f"{session_id}.nii.gz"
    else:
        filename = f"{session_id}.nii"
    
    file_path = UPLOAD_FOLDER / filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Xử lý file NIfTI
        volume_array, metadata = process_nifti_file(str(file_path), session_id)
        
        # Tạo slices cho visualization
        slice_data, num_slices = process_nifti_slices(volume_array, session_id)
        
        # Lưu metadata
        results_dir = RESULTS_FOLDER / session_id
        results_dir.mkdir(exist_ok=True)
        
        # Lưu volume array
        np.save(results_dir / 'nifti_volume.npy', volume_array)
        
        # Cập nhật metadata với slice information
        metadata.update({
            'slice_data': slice_data,
            'uploaded_at': datetime.now().isoformat(),
            'original_filename': file.filename,
            'file_size': file_path.stat().st_size
        })
        
        metadata_path = results_dir / 'nifti_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return NiftiUploadResponse(
            success=True,
            session_id=session_id,
            filename=filename,
            shape=metadata['shape'],
            num_slices=num_slices,
            spacing=metadata['spacing'],
            origin=metadata['origin'],
            min_value=metadata['min_value'],
            max_value=metadata['max_value'],
            message="NIfTI file uploaded and processed successfully!"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing NIfTI file: {str(e)}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_ctpa(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate CTPA from X-ray"""
    
    # Tìm file X-ray
    xray_files = list(UPLOAD_FOLDER.glob(f"{request.session_id}.*"))
    if not xray_files:
        raise HTTPException(status_code=404, detail="X-ray file not found")
    
    xray_path = xray_files[0]
    
    try:
        start_time = datetime.now()
        
        # Tạo thư mục kết quả
        results_dir = RESULTS_FOLDER / request.session_id
        results_dir.mkdir(exist_ok=True)
        
        # Check if we have a real model or should use demo mode
        if inference_model is not None:
            # Use real model
            ctpa_volume, _ = inference_model.inference_pipeline(
                xray_path=str(xray_path),
                output_dir=str(results_dir),
                filename='generated_ctpa',
                guidance_scale=request.guidance_scale,
                show_viewer=False
            )
        else:
            # Use demo mode - generate synthetic CTPA
            print("🎭 Demo mode: generating synthetic CTPA...")
            ctpa_volume = generate_demo_ctpa()
            
            # Save demo results
            np.save(results_dir / 'generated_ctpa.npy', ctpa_volume)
            
            # Create GIF for demo
            from PIL import Image
            ctpa_normalized = (ctpa_volume - ctpa_volume.min()) / (ctpa_volume.max() - ctpa_volume.min())
            frames = []
            for i in range(0, ctpa_volume.shape[0], 2):
                slice_img = (ctpa_normalized[i] * 255).astype(np.uint8)
                slice_rgb = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2RGB)
                frames.append(Image.fromarray(slice_rgb))
            
            if frames:
                frames[0].save(
                    results_dir / 'generated_ctpa.gif',
                    save_all=True,
                    append_images=frames[1:],
                    duration=120,
                    loop=0,
                    optimize=True
                )
        
        # Lưu thêm file NIfTI với orientation đúng cho medical imaging
        ctpa_medical = ctpa_volume.copy()
        if len(ctpa_medical.shape) == 3:
            ctpa_medical = ctpa_medical.transpose(2, 1, 0)  # Reorder to (width, height, depth)
        
        # Denormalize về giá trị gốc của dataset để giống với DICOM
        dataset_min = -12.911299
        dataset_max = 9.596558
        
        # Giả sử model output đã được normalize về [-1, 1] hoặc [0, 1]
        # Denormalize về range gốc của dataset
        ctpa_medical = ctpa_medical * (dataset_max - dataset_min) + dataset_min
        ctpa_medical = ctpa_medical.astype(np.float32)
        
        print(f"📊 Denormalized CTPA range: [{ctpa_medical.min():.3f}, {ctpa_medical.max():.3f}]")
        print(f"📊 Dataset range: [{dataset_min:.3f}, {dataset_max:.3f}]")
        
        # Tạo NIfTI image với metadata phù hợp cho medical viewers
        ctpa_image = sitk.GetImageFromArray(ctpa_medical)
        ctpa_image.SetSpacing([1.0, 1.0, 1.0])  # 1mm isotropic voxels
        ctpa_image.SetOrigin([0.0, 0.0, 0.0])
        ctpa_image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        
        # Lưu file NIfTI với orientation đúng
        nii_path = results_dir / 'generated_ctpa_medical.nii.gz'
        sitk.WriteImage(ctpa_image, str(nii_path))
        print(f"✅ Đã lưu NIfTI với orientation đúng: {nii_path}")
        
        # Xử lý slices cho web
        slice_paths, num_slices = process_ctpa_slices(ctpa_volume, request.session_id)
        
        # Tính thời gian xử lý
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Lưu metadata
        metadata = {
            'session_id': request.session_id,
            'num_slices': num_slices,
            'shape': ctpa_volume.shape,
            'min_value': float(ctpa_volume.min()),
            'max_value': float(ctpa_volume.max()),
            'generated_at': datetime.now().isoformat(),
            'processing_time': processing_time,
            'guidance_scale': request.guidance_scale,
            'slice_paths': slice_paths,
            'demo_mode': inference_model is None
        }
        
        metadata_path = results_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return GenerateResponse(
            success=True,
            session_id=request.session_id,
            num_slices=num_slices,
            shape=list(ctpa_volume.shape),
            message="CTPA generated successfully!" + (" (Demo mode)" if inference_model is None else ""),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating CTPA: {str(e)}")

@app.get("/slice/{session_id}/{slice_idx}", response_model=SliceResponse)
async def get_slice(session_id: str, slice_idx: int, windowed: bool = Query(False)):
    """Get slice by index"""
    try:
        # Load metadata
        metadata_path = RESULTS_FOLDER / session_id / 'metadata.json'
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if slice_idx < 0 or slice_idx >= metadata['num_slices']:
            raise HTTPException(status_code=400, detail="Invalid slice index")
        
        if windowed:
            # Load windowed slice if exists
            windowed_path = STATIC_FOLDER / session_id / "windowed" / f"windowed_slice_{slice_idx:03d}.png"
            if windowed_path.exists():
                slice_img = cv2.imread(str(windowed_path), cv2.IMREAD_GRAYSCALE)
            else:
                raise HTTPException(status_code=404, detail="Windowed slice not found")
        else:
            # Load original slice
            slice_path = STATIC_FOLDER / session_id / f"slice_{slice_idx:03d}.png"
            if not slice_path.exists():
                raise HTTPException(status_code=404, detail="Slice not found")
            slice_img = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
        
        # Convert to base64
        slice_base64 = image_to_base64(slice_img)
        
        return SliceResponse(
            success=True,
            slice_data=slice_base64,
            slice_idx=slice_idx,
            total_slices=metadata['num_slices'],
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting slice: {str(e)}")

@app.get("/api/nifti-metadata/{session_id}")
async def get_nifti_metadata(session_id: str):
    """Get metadata of NIfTI file"""
    try:
        # Kiểm tra session tồn tại
        session_folder = RESULTS_FOLDER / session_id
        if not session_folder.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Load metadata
        metadata_path = session_folder / 'nifti_metadata.json'
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Metadata not found")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load volume to get current shape
        volume_path = session_folder / 'nifti_volume.npy'
        if volume_path.exists():
            volume_array = np.load(volume_path)
            metadata['shape'] = list(volume_array.shape)
            metadata['data_type'] = str(volume_array.dtype)
            metadata['min_value'] = float(np.min(volume_array))
            metadata['max_value'] = float(np.max(volume_array))
            metadata['mean_value'] = float(np.mean(volume_array))
            metadata['std_value'] = float(np.std(volume_array))
        
        return {
            "success": True,
            "metadata": metadata,
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_nifti_metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting NIfTI metadata: {str(e)}")

@app.get("/api/nifti-volume/{session_id}")
async def get_nifti_volume(session_id: str):
    """
    Get full volume data of NIfTI file at once
    
    Args:
        session_id: Session ID
    """
    try:
        # Load metadata
        metadata_path = RESULTS_FOLDER / session_id / 'nifti_metadata.json'
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="NIfTI session not found")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get volume data
        volume_path = RESULTS_FOLDER / session_id / 'nifti_volume.npy'
        if not volume_path.exists():
            raise HTTPException(status_code=404, detail="Volume data not found")
        
        volume_array = np.load(volume_path)
        
        # Convert to list for JSON serialization
        volume_data = volume_array.flatten().tolist()
        
        return {
            "success": True,
            "volume_data": volume_data,
            "shape": list(volume_array.shape),
            "data_type": str(volume_array.dtype),
            "min_value": float(np.min(volume_array)),
            "max_value": float(np.max(volume_array)),
            "mean_value": float(np.mean(volume_array)),
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_nifti_volume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting NIfTI volume: {str(e)}")

@app.get("/api/nifti-slice/{session_id}/{plane}/{slice_idx}")
async def get_nifti_slice(
    session_id: str, 
    plane: str, 
    slice_idx: int,
    window_center: float = Query(None),
    window_width: float = Query(None)
):
    """
    Get slice from NIfTI based on plane (axial, sagittal, coronal)
    
    Args:
        session_id: Session ID
        plane: Viewing plane ('axial', 'sagittal', 'coronal')
        slice_idx: Index of slice
        window_center: Window center for medical windowing
        window_width: Window width for medical windowing
    """
    try:
        # Kiểm tra plane hợp lệ
        if plane not in ['axial', 'sagittal', 'coronal']:
            raise HTTPException(status_code=400, detail="Invalid plane. Use 'axial', 'sagittal', or 'coronal'")
        
        # Load metadata
        metadata_path = RESULTS_FOLDER / session_id / 'nifti_metadata.json'
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="NIfTI session not found")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get volume shape to determine slice limits
        volume_path = RESULTS_FOLDER / session_id / 'nifti_volume.npy'
        if not volume_path.exists():
            raise HTTPException(status_code=404, detail="Volume data not found")
        
        volume_array = np.load(volume_path)
        
        # Determine max slices for each plane
        plane_idx_map = {'axial': 0, 'sagittal': 1, 'coronal': 2}
        max_slices = volume_array.shape[plane_idx_map[plane]]
        
        # Kiểm tra slice index
        if slice_idx < 0 or slice_idx >= max_slices:
            raise HTTPException(status_code=400, detail=f"Invalid slice index {slice_idx} for {plane} plane (max: {max_slices-1})")
        
        # Get slice theo plane
        if plane == 'axial':
            slice_data = volume_array[slice_idx]
        elif plane == 'sagittal':
            slice_data = volume_array[:, slice_idx, :]
        else:  # coronal
            slice_data = volume_array[:, :, slice_idx]
        
        # Apply windowing if specified
        if window_center is not None and window_width is not None:
            windowed_slice = apply_medical_windowing(slice_data, window_center, window_width)
            slice_base64 = image_to_base64(windowed_slice)
        else:
            # Use default windowing
            windowed_slice = apply_medical_windowing(slice_data, 50, 350)
            slice_base64 = image_to_base64(windowed_slice)
        
        return {
            "success": True,
            "slice_data": slice_base64,
            "slice_idx": slice_idx,
            "plane": plane,
            "total_slices": max_slices,
            "metadata": {
                "shape": volume_array.shape,
                "plane_shape": slice_data.shape,
                "data_type": str(volume_array.dtype),
                "min_value": float(np.min(slice_data)),
                "max_value": float(np.max(slice_data)),
                "mean_value": float(np.mean(slice_data))
            },
            "windowing": {
                "center": window_center if window_center is not None else 50,
                "width": window_width if window_width is not None else 350
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_nifti_slice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting NIfTI slice: {str(e)}")

@app.post("/windowing")
async def apply_windowing(request: WindowingRequest):
    """Apply medical windowing to slices"""
    try:
        # Load CTPA volume
        ctpa_path = RESULTS_FOLDER / request.session_id / 'generated_ctpa.npy'
        if not ctpa_path.exists():
            raise HTTPException(status_code=404, detail="CTPA volume not found")
        
        ctpa_volume = np.load(ctpa_path)
        
        # Apply windowing and create new slices
        session_dir = STATIC_FOLDER / request.session_id
        windowed_dir = session_dir / 'windowed'
        windowed_dir.mkdir(exist_ok=True)
        
        slice_paths = []
        for i in range(ctpa_volume.shape[0]):
            slice_img = ctpa_volume[i]
            windowed_slice = apply_medical_windowing(slice_img, request.window_center, request.window_width)
            
            slice_filename = f"windowed_slice_{i:03d}.png"
            slice_path = windowed_dir / slice_filename
            cv2.imwrite(str(slice_path), windowed_slice)
            
            slice_paths.append(f"/static/{request.session_id}/windowed/{slice_filename}")
        
        return {
            "success": True,
            "windowed_slice_paths": slice_paths,
            "window_center": request.window_center,
            "window_width": request.window_width,
            "message": "Windowing applied successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying windowing: {str(e)}")

@app.get("/download/{session_id}/{format}")
async def download_result(session_id: str, format: str):
    """Download result by format"""
    try:
        results_dir = RESULTS_FOLDER / session_id
        
        if format == 'npy':
            file_path = results_dir / 'generated_ctpa.npy'
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="NPY file not found")
            return FileResponse(file_path, filename=f'ctpa_{session_id}.npy')
        
        elif format == 'nifti':
            file_path = results_dir / 'generated_ctpa.nii.gz'
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="NIfTI file not found")
            return FileResponse(file_path, filename=f'ctpa_{session_id}.nii.gz')
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@app.get("/api/download-nifti/{session_id}/{format}")
async def download_nifti_result(session_id: str, format: str):
    """Download NIfTI result by format"""
    return await download_result(session_id, format)

@app.get("/api/download-slice/{session_id}/{plane}/{slice_idx}")
async def download_slice_as_png(
    session_id: str, 
    plane: str, 
    slice_idx: int,
    window_center: float = Query(50),
    window_width: float = Query(350)
):
    """Download current slice as PNG file"""
    try:
        # Kiểm tra plane hợp lệ
        if plane not in ['axial', 'sagittal', 'coronal']:
            raise HTTPException(status_code=400, detail="Invalid plane")
        
        # Load metadata
        metadata_path = RESULTS_FOLDER / session_id / 'nifti_metadata.json'
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="NIfTI session not found")
        
        # Get volume data
        volume_path = RESULTS_FOLDER / session_id / 'nifti_volume.npy'
        if not volume_path.exists():
            raise HTTPException(status_code=404, detail="Volume data not found")
        
        volume_array = np.load(volume_path)
        
        # Get slice based on plane
        plane_idx_map = {'axial': 0, 'sagittal': 1, 'coronal': 2}
        max_slices = volume_array.shape[plane_idx_map[plane]]
        
        if slice_idx < 0 or slice_idx >= max_slices:
            raise HTTPException(status_code=400, detail="Invalid slice index")
        
        if plane == 'axial':
            slice_data = volume_array[slice_idx]
        elif plane == 'sagittal':
            slice_data = volume_array[:, slice_idx, :]
        else:  # coronal
            slice_data = volume_array[:, :, slice_idx]
        
        # Apply windowing
        windowed_slice = apply_medical_windowing(slice_data, window_center, window_width)
        
        # Save to temporary file
        temp_dir = RESULTS_FOLDER / session_id / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        filename = f"slice_{session_id[:8]}_{plane}_{slice_idx:03d}.png"
        temp_path = temp_dir / filename
        
        # Convert to PIL Image and save
        img = Image.fromarray(windowed_slice)
        img.save(temp_path, format='PNG')
        
        return FileResponse(
            path=str(temp_path),
            media_type="image/png",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading slice: {str(e)}")

@app.get("/sessions/{session_id}/metadata")
async def get_session_metadata(session_id: str):
    """Get session metadata"""
    try:
        metadata_path = RESULTS_FOLDER / session_id / 'metadata.json'
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metadata: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete session and all related files"""
    try:
        # Delete upload file
        upload_files = list(UPLOAD_FOLDER.glob(f"{session_id}.*"))
        for file_path in upload_files:
            file_path.unlink()
        
        # Delete results
        results_dir = RESULTS_FOLDER / session_id
        if results_dir.exists():
            shutil.rmtree(results_dir)
        
        # Delete static files
        static_dir = STATIC_FOLDER / session_id
        if static_dir.exists():
            shutil.rmtree(static_dir)
        
        return {"success": True, "message": "Session deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app.get("/windowing_presets")
async def get_windowing_presets():
    """Get medical imaging windowing presets"""
    presets = {
        "lung": {"center": -600, "width": 1500},
        "soft_tissue": {"center": 50, "width": 350},
        "bone": {"center": 400, "width": 1500},
        "brain": {"center": 40, "width": 80},
        "liver": {"center": 60, "width": 160},
        "mediastinum": {"center": 50, "width": 350}
    }
    return presets

# Add evaluation functionality
@app.post("/upload-evaluation-files")
async def upload_evaluation_files(
    gt_file: UploadFile = File(...),
    pred_file: UploadFile = File(...)
):
    """Upload GT and prediction files for evaluation"""
    try:
        # Create evaluation session
        eval_session_id = str(uuid.uuid4())
        eval_dir = UPLOAD_FOLDER / "evaluations" / eval_session_id
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Save GT file
        gt_filename = f"gt_{gt_file.filename}"
        gt_path = eval_dir / gt_filename
        with open(gt_path, "wb") as buffer:
            content = await gt_file.read()
            buffer.write(content)
        
        # Save prediction file
        pred_filename = f"pred_{pred_file.filename}"
        pred_path = eval_dir / pred_filename
        with open(pred_path, "wb") as buffer:
            content = await pred_file.read()
            buffer.write(content)
        
        # Load and get basic info
        gt_data = np.load(gt_path)
        pred_data = np.load(pred_path)
        
        # Save session metadata
        session_metadata = {
            "eval_session_id": eval_session_id,
            "gt_file": gt_filename,
            "pred_file": pred_filename,
            "gt_shape": gt_data.shape,
            "pred_shape": pred_data.shape,
            "gt_range": [float(gt_data.min()), float(gt_data.max())],
            "pred_range": [float(pred_data.min()), float(pred_data.max())],
            "created_at": datetime.now().isoformat()
        }
        
        with open(eval_dir / "metadata.json", "w") as f:
            json.dump(session_metadata, f)
        
        return JSONResponse({
            "success": True,
            "eval_session_id": eval_session_id,
            "gt_shape": gt_data.shape,
            "pred_shape": pred_data.shape,
            "message": "Files uploaded successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_ct_pair(request: EvaluationRequest):
    """Evaluate CT pair using the evaluation script"""
    try:
        import sys
        import subprocess
        from pathlib import Path
        
        # Get evaluation session directory
        eval_dir = UPLOAD_FOLDER / "evaluations" / request.gt_session_id
        if not eval_dir.exists():
            raise HTTPException(status_code=404, detail="Evaluation session not found")
        
        # Load metadata
        with open(eval_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Get file paths
        gt_path = eval_dir / metadata["gt_file"]
        pred_path = eval_dir / metadata["pred_file"]
        
        # Create output directory for this evaluation
        output_dir = eval_dir / "results"
        output_dir.mkdir(exist_ok=True)
        
        # Import the evaluation class
        sys.path.append(str(Path(__file__).parent.parent))
        from simple_evaluation_with_viz import SimpleEvaluatorWithViz, interpret_results, create_readme
        
        # Initialize evaluator
        evaluator = SimpleEvaluatorWithViz()
        
        # Run evaluation
        result = evaluator.evaluate_pair(
            str(gt_path),
            str(pred_path),
            create_viz=request.create_viz,
            output_dir=str(output_dir / "visualizations")
        )
        
        metrics = result['metrics']
        viz_paths = result.get('visualizations', None)
        interpretation = interpret_results(metrics)
        
        # Create README content
        if request.language == "vi":
            readme_content = create_readme(metrics, interpretation, viz_paths)
        else:
            # Create English README
            readme_content = create_readme_en(metrics, interpretation, viz_paths)
        
        # Save results
        evaluation_results = {
            "eval_session_id": request.gt_session_id,
            "metrics": metrics,
            "interpretation": interpretation,
            "visualizations": viz_paths,
            "readme_content": readme_content,
            "created_at": datetime.now().isoformat()
        }
        
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        return EvaluationResponse(
            success=True,
            evaluation_id=request.gt_session_id,
            metrics=metrics,
            interpretation=interpretation,
            visualizations=viz_paths,
            readme_content=readme_content,
            message="Evaluation completed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/evaluation/{eval_session_id}/visualization/{filename}")
async def get_evaluation_visualization(eval_session_id: str, filename: str):
    """Get evaluation visualization files (GIFs, images)"""
    try:
        viz_path = UPLOAD_FOLDER / "evaluations" / eval_session_id / "results" / "visualizations" / filename
        
        if not viz_path.exists():
            raise HTTPException(status_code=404, detail="Visualization file not found")
        
        return FileResponse(
            str(viz_path),
            media_type="image/gif" if filename.endswith(".gif") else "image/png",
            filename=filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving visualization: {str(e)}")

@app.get("/evaluation/{eval_session_id}/slice/{file_type}/{plane}/{slice_idx}")
async def get_evaluation_slice(
    eval_session_id: str,
    file_type: str,  # "gt" or "pred"
    plane: str,      # "axial", "sagittal", "coronal"
    slice_idx: int,
    window_center: float = Query(-600),
    window_width: float = Query(1500)
):
    """Get individual slice from GT or prediction for comparison"""
    try:
        eval_dir = UPLOAD_FOLDER / "evaluations" / eval_session_id
        
        # Load metadata
        with open(eval_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load appropriate file
        if file_type == "gt":
            file_path = eval_dir / metadata["gt_file"]
        elif file_type == "pred":
            file_path = eval_dir / metadata["pred_file"]
        else:
            raise HTTPException(status_code=400, detail="Invalid file_type. Use 'gt' or 'pred'")
        
        volume = np.load(file_path)
        
        # Apply transpose for prediction if needed
        if file_type == "pred":
            volume = np.transpose(volume, (1, 2, 0))
        
        # Normalize to [-1, 1] format
        if volume.min() >= -1.1 and volume.max() <= 1.1:
            normalized = volume.copy()
        else:
            normalized = (volume - volume.min()) / (volume.max() - volume.min())
            normalized = normalized * 2 - 1
        
        # Denormalize to HU values
        volume_hu = (normalized + 1.0) / 2.0
        volume_hu = volume_hu * (1000 - (-1000)) + (-1000)  # Approximate HU range
        
        # Get slice based on plane
        if plane == "axial":
            if slice_idx >= volume_hu.shape[0]:
                raise HTTPException(status_code=400, detail="Slice index out of range")
            slice_data = volume_hu[slice_idx]
        elif plane == "sagittal":
            if slice_idx >= volume_hu.shape[2]:
                raise HTTPException(status_code=400, detail="Slice index out of range")
            slice_data = volume_hu[:, :, slice_idx]
        elif plane == "coronal":
            if slice_idx >= volume_hu.shape[1]:
                raise HTTPException(status_code=400, detail="Slice index out of range")
            slice_data = volume_hu[:, slice_idx, :]
        else:
            raise HTTPException(status_code=400, detail="Invalid plane. Use 'axial', 'sagittal', or 'coronal'")
        
        # Apply windowing
        windowed_slice = apply_medical_windowing(slice_data, window_center, window_width)
        
        # Convert to base64
        slice_base64 = image_to_base64(windowed_slice)
        
        return JSONResponse({
            "success": True,
            "slice_data": slice_base64,
            "slice_idx": slice_idx,
            "plane": plane,
            "file_type": file_type,
            "total_slices": volume_hu.shape[0] if plane == "axial" else 
                           volume_hu.shape[2] if plane == "sagittal" else volume_hu.shape[1],
            "shape": list(volume_hu.shape)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting slice: {str(e)}")

def create_readme_en(metrics: Dict[str, float], interpretation: Dict[str, str], viz_paths=None) -> str:
    """Create English README content"""
    
    # Create visualization section with actual paths
    viz_section = ""
    if viz_paths:
        gt_gifs = viz_paths['gt_gifs']
        pred_gifs = viz_paths['pred_gifs']
        
        viz_section = f"""## 🎥 Visual Comparison

### CT Axial View (Cross-sectional)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Axial]({gt_gifs['axial']}) | ![Pred Axial]({pred_gifs['axial']}) |

### CT Sagittal View (Longitudinal) 

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Sagittal]({gt_gifs['sagittal']}) | ![Pred Sagittal]({pred_gifs['sagittal']}) |

### CT Coronal View (Front-to-Back)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Coronal]({gt_gifs['coronal']}) | ![Pred Coronal]({pred_gifs['coronal']}) |

### Comprehensive Comparison

![Comparison]({viz_paths['comparison_image']})

"""
    
    readme_content = f"""# X-ray2CT Model Evaluation

## Evaluation Results

### 📊 Image Quality Metrics

- **SSIM**: {metrics['SSIM']:.4f} (**{interpretation['SSIM_Quality']}**)
- **PSNR**: {metrics['PSNR']:.2f} dB (**{interpretation['PSNR_Quality']}**)

### 📈 Analysis and Assessment

#### SSIM (Structural Similarity Index)
- **Value**: {metrics['SSIM']:.4f}/1.0
- **Meaning**: Measures structural similarity of images
- **Assessment**: Achieves **{interpretation['SSIM_Quality']}** level. The model can {"accurately reconstruct" if metrics['SSIM'] >= 0.7 else "reasonably reconstruct" if metrics['SSIM'] >= 0.6 else "needs improvement in reconstructing"} anatomical structures.

#### PSNR (Peak Signal-to-Noise Ratio)  
- **Value**: {metrics['PSNR']:.2f} dB
- **Meaning**: Measures signal-to-noise ratio, higher is better
- **Assessment**: Achieves **{interpretation['PSNR_Quality']}** level. Generated image quality is {"quite close" if metrics['PSNR'] >= 25 else "relatively close" if metrics['PSNR'] >= 20 else "needs improvement compared"} to ground truth.

{viz_section}

## 🏥 Medical Metrics

- **Lung Dice**: {metrics['Lung_Dice']:.4f} ({interpretation['Medical_Quality'] if metrics['Lung_Dice'] >= 0.8 else "Good" if metrics['Lung_Dice'] >= 0.6 else "Fair"})
- **Soft Tissue Dice**: {metrics['Soft_Tissue_Dice']:.4f} ({interpretation['Medical_Quality'] if metrics['Soft_Tissue_Dice'] >= 0.8 else "Good" if metrics['Soft_Tissue_Dice'] >= 0.6 else "Fair"})
- **Medical Quality**: **{interpretation['Medical_Quality']}** (Avg Dice: {interpretation['Average_Dice']})

## 🔍 Conclusion

The X-ray2CT model achieves **{interpretation['SSIM_Quality'].lower()}** results with:

- {"✅" if metrics['SSIM'] >= 0.6 else "⚠️"} **Anatomical structure** {"accurately" if metrics['SSIM'] >= 0.7 else "reasonably" if metrics['SSIM'] >= 0.6 else "needs improvement"} reconstructed (SSIM {metrics['SSIM']:.3f})
- {"✅" if metrics['PSNR'] >= 20 else "⚠️"} **Image quality** at {interpretation['PSNR_Quality'].lower()} level (PSNR {metrics['PSNR']:.1f} dB)
- {"✅" if float(interpretation['Average_Dice']) >= 0.8 else "⚠️"} **Medical accuracy** {interpretation['Medical_Quality'].lower()} (Avg Dice: {interpretation['Average_Dice']})

## 📊 Technical Details

- **MSE**: {metrics['MSE']:.6f}
- **MAE**: {metrics['MAE']:.6f}
- **Orientation**: Fixed alignment issue with transpose(1,2,0)
- **Format**: Compatible with LIDC training pipeline

---
*Results automatically generated from evaluation system*
"""
    
    return readme_content

def generate_demo_ctpa(shape: tuple = (128, 32, 32)) -> np.ndarray:
    """Generate demo CTPA data for testing"""
    depth, height, width = shape
    
    # Base volume
    ctpa = np.random.normal(0.0, 0.1, shape).astype(np.float32)
    
    # Add anatomical structures
    center_y, center_x = height // 2, width // 2
    
    for z in range(depth):
        # Lung regions
        lung_size = max(5, int(15 * (1 - abs(z - depth//2) / (depth//2))))
        
        # Left lung
        left_y = center_y + int(np.sin(z * 0.1) * 3)
        left_x = center_x - 8 + int(np.cos(z * 0.1) * 2)
        if 0 <= left_y < height and 0 <= left_x < width:
            y1, y2 = max(0, left_y - lung_size//2), min(height, left_y + lung_size//2)
            x1, x2 = max(0, left_x - lung_size//2), min(width, left_x + lung_size//2)
            ctpa[z, y1:y2, x1:x2] = np.random.normal(-0.8, 0.1, (y2-y1, x2-x1))
        
        # Right lung
        right_y = center_y + int(np.sin(z * 0.1 + np.pi/4) * 3)
        right_x = center_x + 8 + int(np.cos(z * 0.1 + np.pi/4) * 2)
        if 0 <= right_y < height and 0 <= right_x < width:
            y1, y2 = max(0, right_y - lung_size//2), min(height, right_y + lung_size//2)
            x1, x2 = max(0, right_x - lung_size//2), min(width, right_x + lung_size//2)
            ctpa[z, y1:y2, x1:x2] = np.random.normal(-0.8, 0.1, (y2-y1, x2-x1))
        
        # Heart/mediastinum
        heart_size = max(3, int(8 * (1 - abs(z - depth//2) / (depth//2))))
        heart_y = center_y + int(np.sin(z * 0.05) * 2)
        heart_x = center_x + int(np.cos(z * 0.05) * 1)
        if 0 <= heart_y < height and 0 <= heart_x < width:
            y1, y2 = max(0, heart_y - heart_size//2), min(height, heart_y + heart_size//2)
            x1, x2 = max(0, heart_x - heart_size//2), min(width, heart_x + heart_size//2)
            ctpa[z, y1:y2, x1:x2] = np.random.normal(0.5, 0.1, (y2-y1, x2-x1))
    
    # Normalize to dataset range
    dataset_min = -12.911299
    dataset_max = 9.596558
    ctpa = ctpa * (dataset_max - dataset_min) + dataset_min
    
    return ctpa

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="X-ray2CT FastAPI Server")
    parser.add_argument("--checkpoint", type=str, default="demo", help="Path to model checkpoint or 'demo' for demo mode")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server")
    parser.add_argument("--reload", action="store_true", help="Auto reload on code changes")
    
    args = parser.parse_args()
    
    # Initialize model
    print("🔄 Initializing backend...")
    if init_model(args.checkpoint):
        print("🚀 Starting FastAPI server...")
        if args.checkpoint.lower() in ["demo", "none", "dummy"]:
            print("🎭 Server running in DEMO MODE")
            print("   - X-ray upload: ✅ Available")
            print("   - NIfTI upload: ✅ Available")  
            print("   - Evaluation: ✅ Available")
            print("   - Visualization: ✅ Available")
            print("   - CTPA generation: 🎭 Demo data only")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    else:
        print("❌ Could not initialize backend. Exiting.") 