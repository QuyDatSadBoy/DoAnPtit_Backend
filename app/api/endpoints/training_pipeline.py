"""
Training Pipeline API Endpoints
Cung cấp API để demo quy trình training từ LIDC → DiffDRR → CycleGAN → Xray2CT
SỬ DỤNG CÁC MODULE THẬT - KHÔNG FAKE
"""
import os
import sys
import json
import subprocess
import logging
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pydicom
import numpy as np
from PIL import Image
import io
import base64
import torch

# Import config helpers
from app.core.config import (
    settings,
    get_base_dir,
    get_dataset_dir,
    get_diffdrr_dir,
    get_cyclegan_dir,
    get_xray2ct_dir,
    get_results_dir
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Paths configuration - Sử dụng từ config
BASE_DIR = get_base_dir()
DATASET_DIR = get_dataset_dir()
DIFFDRR_DIR = get_diffdrr_dir()
CYCLEGAN_DIR = get_cyclegan_dir()
XRAY2CT_DIR = get_xray2ct_dir()
RESULTS_DIR = get_results_dir() / "training_pipeline"

# DiffDRR results directories
DIFFDRR_RESULTS = DIFFDRR_DIR / "results"
DIFFDRR_FINAL_RESULTS = DIFFDRR_DIR / "final_results"

# CycleGAN results
CYCLEGAN_RESULTS = CYCLEGAN_DIR / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== STEP 1: LIDC Dataset ====================

@router.get("/patients")
async def get_available_patients():
    """Lấy danh sách bệnh nhân có sẵn trong dataset LIDC-IDRI"""
    patients = []
    
    if not DATASET_DIR.exists():
        raise HTTPException(status_code=404, detail=f"Dataset directory not found: {DATASET_DIR}")
    
    for folder in sorted(DATASET_DIR.iterdir()):
        if folder.is_dir() and folder.name.startswith("LIDC-IDRI-"):
            # Count DICOM files
            dcm_files = list(folder.glob("*.dcm"))
            xml_files = list(folder.glob("*.xml"))
            
            patients.append({
                "patient_id": folder.name,
                "folder_path": str(folder),
                "num_slices": len(dcm_files),
                "has_xml": len(xml_files) > 0,
                "xml_file": xml_files[0].name if xml_files else None
            })
    
    return {
        "total_patients": len(patients),
        "dataset_path": str(DATASET_DIR),
        "patients": patients
    }


@router.get("/patients/{patient_id}/info")
async def get_patient_info(patient_id: str):
    """Lấy thông tin chi tiết của một bệnh nhân"""
    patient_dir = DATASET_DIR / patient_id
    
    if not patient_dir.exists():
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    dcm_files = sorted(patient_dir.glob("*.dcm"))
    xml_files = list(patient_dir.glob("*.xml"))
    
    # Read first DICOM file for metadata
    dicom_info = {}
    if dcm_files:
        try:
            ds = pydicom.dcmread(str(dcm_files[0]))
            dicom_info = {
                "patient_name": str(getattr(ds, 'PatientName', 'N/A')),
                "modality": str(getattr(ds, 'Modality', 'N/A')),
                "rows": int(getattr(ds, 'Rows', 0)),
                "columns": int(getattr(ds, 'Columns', 0)),
                "pixel_spacing": [float(x) for x in getattr(ds, 'PixelSpacing', [0, 0])] if hasattr(ds, 'PixelSpacing') else [0, 0],
                "slice_thickness": float(getattr(ds, 'SliceThickness', 0)) if hasattr(ds, 'SliceThickness') else 0,
                "bits_allocated": int(getattr(ds, 'BitsAllocated', 0)),
                "bits_stored": int(getattr(ds, 'BitsStored', 0)),
            }
        except Exception as e:
            dicom_info = {"error": str(e)}
    
    return {
        "patient_id": patient_id,
        "folder_path": str(patient_dir),
        "num_slices": len(dcm_files),
        "slice_files": [f.name for f in dcm_files],
        "has_xml": len(xml_files) > 0,
        "xml_file": xml_files[0].name if xml_files else None,
        "dicom_info": dicom_info
    }


@router.get("/patients/{patient_id}/all-slices")
async def get_all_patient_slices(patient_id: str):
    """Lấy TẤT CẢ slices của một bệnh nhân cho CT Viewer"""
    patient_dir = DATASET_DIR / patient_id
    
    if not patient_dir.exists():
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    dcm_files = sorted(patient_dir.glob("*.dcm"))
    
    slices = []
    for idx, dcm_file in enumerate(dcm_files):
        try:
            ds = pydicom.dcmread(str(dcm_file))
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Apply HU transformation
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            
            # Apply lung window for CT visualization
            window_center = -600
            window_width = 1500
            img_min = window_center - window_width / 2
            img_max = window_center + window_width / 2
            pixel_array = np.clip(pixel_array, img_min, img_max)
            pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            
            # Convert to base64 PNG
            img = Image.fromarray(pixel_array)
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Get slice location
            slice_location = float(ds.SliceLocation) if hasattr(ds, 'SliceLocation') else idx
            
            slices.append({
                "index": idx,
                "slice_number": int(dcm_file.stem.split('-')[-1]) if '-' in dcm_file.stem else idx,
                "filename": dcm_file.name,
                "slice_location": slice_location,
                "image_data": f"data:image/png;base64,{img_base64}",
                "shape": list(ds.pixel_array.shape)
            })
        except Exception as e:
            slices.append({
                "index": idx,
                "slice_number": idx,
                "filename": dcm_file.name,
                "error": str(e)
            })
    
    # Sort by slice location
    slices.sort(key=lambda x: x.get('slice_location', x.get('index', 0)))
    
    return {
        "patient_id": patient_id,
        "total_slices": len(dcm_files),
        "slices": slices
    }


@router.get("/patients/{patient_id}/slices")
async def get_patient_slices(patient_id: str, start: int = 0, limit: int = 10):
    """Lấy danh sách slices và preview của một bệnh nhân (phân trang)"""
    patient_dir = DATASET_DIR / patient_id
    
    if not patient_dir.exists():
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    dcm_files = sorted(patient_dir.glob("*.dcm"))
    
    # Get subset of slices
    selected_files = dcm_files[start:start + limit]
    
    slices = []
    for dcm_file in selected_files:
        try:
            ds = pydicom.dcmread(str(dcm_file))
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Apply HU transformation
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            
            # Apply lung window
            window_center = -600
            window_width = 1500
            img_min = window_center - window_width / 2
            img_max = window_center + window_width / 2
            pixel_array = np.clip(pixel_array, img_min, img_max)
            pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            
            # Convert to base64 PNG
            img = Image.fromarray(pixel_array)
            img = img.resize((256, 256))  # Resize for preview
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            slices.append({
                "slice_number": int(dcm_file.stem.split('-')[-1]) if '-' in dcm_file.stem else 0,
                "filename": dcm_file.name,
                "image_data": f"data:image/png;base64,{img_base64}",
                "shape": list(ds.pixel_array.shape)
            })
        except Exception as e:
            slices.append({
                "slice_number": 0,
                "filename": dcm_file.name,
                "error": str(e)
            })
    
    return {
        "patient_id": patient_id,
        "total_slices": len(dcm_files),
        "start": start,
        "limit": limit,
        "slices": slices
    }


@router.get("/patients/{patient_id}/slice/{slice_index}")
async def get_single_slice(patient_id: str, slice_index: int, window: str = "lung"):
    """Lấy một slice cụ thể với full resolution và window setting"""
    patient_dir = DATASET_DIR / patient_id
    
    if not patient_dir.exists():
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    dcm_files = sorted(patient_dir.glob("*.dcm"))
    
    if slice_index >= len(dcm_files) or slice_index < 0:
        raise HTTPException(status_code=404, detail=f"Slice index {slice_index} out of range")
    
    dcm_file = dcm_files[slice_index]
    
    try:
        ds = pydicom.dcmread(str(dcm_file))
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Apply HU transformation
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # Window presets
        windows = {
            "lung": {"center": -600, "width": 1500},
            "mediastinum": {"center": 40, "width": 400},
            "bone": {"center": 300, "width": 1500},
            "soft_tissue": {"center": 50, "width": 350},
            "brain": {"center": 40, "width": 80},
        }
        
        window_setting = windows.get(window, windows["lung"])
        window_center = window_setting["center"]
        window_width = window_setting["width"]
        
        img_min = window_center - window_width / 2
        img_max = window_center + window_width / 2
        pixel_array = np.clip(pixel_array, img_min, img_max)
        pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        # Convert to base64 PNG
        img = Image.fromarray(pixel_array)
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "patient_id": patient_id,
            "slice_index": slice_index,
            "total_slices": len(dcm_files),
            "filename": dcm_file.name,
            "window": window,
            "window_center": window_center,
            "window_width": window_width,
            "image_data": f"data:image/png;base64,{img_base64}",
            "shape": list(ds.pixel_array.shape),
            "dicom_info": {
                "rows": int(ds.Rows),
                "columns": int(ds.Columns),
                "pixel_spacing": [float(x) for x in ds.PixelSpacing] if hasattr(ds, 'PixelSpacing') else None,
                "slice_location": float(ds.SliceLocation) if hasattr(ds, 'SliceLocation') else None,
                "slice_thickness": float(ds.SliceThickness) if hasattr(ds, 'SliceThickness') else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STEP 2: DiffDRR - SỬ DỤNG CODE THẬT ====================

@router.post("/diffdrr/generate/{patient_id}")
async def generate_drr_real(patient_id: str):
    """
    Tạo ảnh X-ray giả (DRR) từ CT scan sử dụng DiffDRR THẬT
    Sử dụng code từ create_dataset.py
    """
    patient_dir = DATASET_DIR / patient_id
    
    if not patient_dir.exists():
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    # Kiểm tra xem đã có kết quả trong final_results chưa
    final_result_path = DIFFDRR_FINAL_RESULTS / f"{patient_id}.png"
    raw_result_path = DIFFDRR_RESULTS / f"{patient_id}.png"
    
    # Nếu đã có kết quả sẵn trong final_results, trả về luôn
    if final_result_path.exists():
        with open(final_result_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        img = Image.open(final_result_path)
        
        return {
            "patient_id": patient_id,
            "status": "completed",
            "message": "DRR đã có sẵn trong final_results (đã được xử lý trước đó)",
            "source": "cached",
            "output_path": str(final_result_path),
            "image_data": f"data:image/png;base64,{img_base64}",
            "image_shape": [img.width, img.height]
        }
    
    # Nếu chưa có, cần chạy DiffDRR thật
    # Import và sử dụng code DiffDRR
    try:
        # Thêm path để import DiffDRR modules
        diffdrr_process_dir = DIFFDRR_DIR / "process_dicom_file"
        if str(diffdrr_process_dir) not in sys.path:
            sys.path.insert(0, str(diffdrr_process_dir))
        if str(DIFFDRR_DIR) not in sys.path:
            sys.path.insert(0, str(DIFFDRR_DIR))
        
        import torch
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        from diffdrr.data import read
        from diffdrr.drr import DRR
        from diffdrr.visualization import plot_drr
        
        # Ensure output directories exist
        DIFFDRR_RESULTS.mkdir(parents=True, exist_ok=True)
        DIFFDRR_FINAL_RESULTS.mkdir(parents=True, exist_ok=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Read CT volume
        subject = read(
            str(patient_dir), 
            labelmap=None, 
            labels=None, 
            orientation="AP", 
            bone_attenuation_multiplier=1.0
        )
        
        # Initialize DRR module
        drr = DRR(
            subject,
            sdd=1020.0,    # Source-to-detector distance
            height=200,    # Image height
            delx=2.0,      # Pixel spacing (mm)
        ).to(device)
        
        # Generate DRR with fixed rotation and translation
        rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
        
        with torch.no_grad():
            img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
        
        # Save raw result
        plt.figure(figsize=(10, 10))
        plot_drr(img, ticks=False)
        plt.savefig(str(raw_result_path))
        plt.close()
        
        # Crop the image (remove matplotlib borders)
        with Image.open(raw_result_path) as pil_img:
            # Crop margins: left=321, top=61, right=294, bottom=54 (from create_dataset.py)
            left = 321
            top = 61
            right = pil_img.width - 294
            bottom = pil_img.height - 54
            cropped = pil_img.crop((left, top, right, bottom))
            cropped.save(str(final_result_path))
        
        # Return the cropped result
        with open(final_result_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        cropped_img = Image.open(final_result_path)
        
        return {
            "patient_id": patient_id,
            "status": "completed",
            "message": "DRR generated successfully using DiffDRR",
            "source": "generated",
            "output_path": str(final_result_path),
            "image_data": f"data:image/png;base64,{img_base64}",
            "image_shape": [cropped_img.width, cropped_img.height],
            "device_used": str(device)
        }
        
    except ImportError as e:
        # Fallback: Nếu không import được DiffDRR, thử chạy subprocess
        raise HTTPException(
            status_code=500, 
            detail=f"DiffDRR module not available. Error: {str(e)}. Please install DiffDRR first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating DRR: {str(e)}")


@router.get("/diffdrr/result/{patient_id}")
async def get_drr_result(patient_id: str):
    """Lấy kết quả DRR đã tạo"""
    # Ưu tiên lấy từ final_results (đã crop)
    final_result_path = DIFFDRR_FINAL_RESULTS / f"{patient_id}.png"
    raw_result_path = DIFFDRR_RESULTS / f"{patient_id}.png"
    
    if final_result_path.exists():
        with open(final_result_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        img = Image.open(final_result_path)
        return {
            "patient_id": patient_id,
            "source": "final_results",
            "image_data": f"data:image/png;base64,{img_base64}",
            "output_path": str(final_result_path),
            "image_shape": [img.width, img.height]
        }
    elif raw_result_path.exists():
        with open(raw_result_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        img = Image.open(raw_result_path)
        return {
            "patient_id": patient_id,
            "source": "results",
            "image_data": f"data:image/png;base64,{img_base64}",
            "output_path": str(raw_result_path),
            "image_shape": [img.width, img.height]
        }
    else:
        raise HTTPException(status_code=404, detail="DRR result not found. Please generate first.")


# ==================== STEP 3: CycleGAN - SỬ DỤNG CODE THẬT ====================

@router.post("/cyclegan/convert/{patient_id}")
async def convert_with_cyclegan_real(patient_id: str):
    """
    Chuyển đổi X-ray giả sang X-ray thật sử dụng CycleGAN THẬT
    Sử dụng code từ cyclegan_converter.py
    """
    logger.info(f"[CycleGAN] Starting conversion for patient: {patient_id}")
    
    # Tìm ảnh DRR input
    drr_path = DIFFDRR_FINAL_RESULTS / f"{patient_id}.png"
    logger.info(f"[CycleGAN] Checking final_results path: {drr_path}")
    
    if not drr_path.exists():
        drr_path = DIFFDRR_RESULTS / f"{patient_id}.png"
        logger.info(f"[CycleGAN] Checking results path: {drr_path}")
    
    if not drr_path.exists():
        logger.error(f"[CycleGAN] DRR image not found for {patient_id}")
        raise HTTPException(status_code=404, detail="DRR image not found. Please run DiffDRR step first.")
    
    logger.info(f"[CycleGAN] Found DRR input at: {drr_path}")
    
    output_dir = RESULTS_DIR / "cyclegan" / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{patient_id}_cyclegan.png"
    logger.info(f"[CycleGAN] Output path: {output_path}")
    
    try:
        # Thêm path để import CycleGAN modules
        logger.info(f"[CycleGAN] Adding CYCLEGAN_DIR to sys.path: {CYCLEGAN_DIR}")
        if str(CYCLEGAN_DIR) not in sys.path:
            sys.path.insert(0, str(CYCLEGAN_DIR))
        
        logger.info(f"[CycleGAN] Importing CycleGANConverter...")
        from cyclegan_converter import CycleGANConverter
        logger.info(f"[CycleGAN] CycleGANConverter imported successfully")
        
        # Khởi tạo converter
        checkpoints_dir = str(CYCLEGAN_DIR / 'checkpoints')
        gpu_id = 0 if torch.cuda.is_available() else -1
        logger.info(f"[CycleGAN] Initializing converter with checkpoints_dir={checkpoints_dir}, gpu_id={gpu_id}")
        
        converter = CycleGANConverter(
            model_name='xray_cyclegan',
            checkpoints_dir=checkpoints_dir,
            gpu_id=gpu_id,
            load_size=256,
            crop_size=256
        )
        logger.info(f"[CycleGAN] Converter initialized successfully")
        
        # Chuyển đổi
        logger.info(f"[CycleGAN] Converting image...")
        result_path = converter.convert_single_image(
            str(drr_path), 
            str(output_path)
        )
        logger.info(f"[CycleGAN] Conversion completed. Result saved to: {result_path}")
        
        # Đọc kết quả
        with open(result_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        img = Image.open(result_path)
        logger.info(f"[CycleGAN] Result image shape: {img.width}x{img.height}")
        
        return {
            "patient_id": patient_id,
            "status": "completed",
            "message": "CycleGAN transformation completed using trained model",
            "input_path": str(drr_path),
            "output_path": result_path,
            "image_data": f"data:image/png;base64,{img_base64}",
            "image_shape": [img.width, img.height]
        }
        
    except ImportError as e:
        # Fallback nếu không có model
        logger.error(f"[CycleGAN] ImportError: {str(e)}")
        import traceback
        logger.error(f"[CycleGAN] Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"CycleGAN model not available. Error: {str(e)}. Please ensure model is trained."
        )
    except Exception as e:
        logger.error(f"[CycleGAN] Exception: {str(e)}")
        import traceback
        logger.error(f"[CycleGAN] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error in CycleGAN conversion: {str(e)}")


@router.get("/cyclegan/result/{patient_id}")
async def get_cyclegan_result(patient_id: str):
    """Lấy kết quả CycleGAN đã tạo"""
    output_dir = RESULTS_DIR / "cyclegan" / patient_id
    output_path = output_dir / f"{patient_id}_cyclegan.png"
    
    # Cũng kiểm tra trong thư mục results của CycleGAN
    cyclegan_result = CYCLEGAN_RESULTS / f"{patient_id}_converted.png"
    
    if output_path.exists():
        with open(output_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        img = Image.open(output_path)
        return {
            "patient_id": patient_id,
            "image_data": f"data:image/png;base64,{img_base64}",
            "output_path": str(output_path),
            "image_shape": [img.width, img.height]
        }
    elif cyclegan_result.exists():
        with open(cyclegan_result, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        img = Image.open(cyclegan_result)
        return {
            "patient_id": patient_id,
            "image_data": f"data:image/png;base64,{img_base64}",
            "output_path": str(cyclegan_result),
            "image_shape": [img.width, img.height]
        }
    else:
        raise HTTPException(status_code=404, detail="CycleGAN result not found.")


# ==================== STEP 4: Xray2CT - TRAINING DATA PREPARATION ====================

@router.post("/xray2ct/prepare-training/{patient_id}")
async def prepare_xray2ct_training_data(patient_id: str):
    """
    Chuẩn bị data cho Xray2CT training:
    - Input: X-ray từ CycleGAN (realistic X-ray)
    - Ground Truth: CT scan gốc từ LIDC
    
    Đây là bước quan trọng trong training pipeline!
    """
    logger.info(f"[Xray2CT] Preparing training data for patient: {patient_id}")
    
    # 1. Tìm X-ray input (từ CycleGAN - realistic X-ray)
    cyclegan_path = RESULTS_DIR / "cyclegan" / patient_id / f"{patient_id}_cyclegan.png"
    drr_path = DIFFDRR_FINAL_RESULTS / f"{patient_id}.png"
    
    xray_input_path = cyclegan_path if cyclegan_path.exists() else drr_path
    
    if not xray_input_path.exists():
        logger.error(f"[Xray2CT] X-ray input not found for {patient_id}")
        raise HTTPException(status_code=404, detail="X-ray input not found. Please run CycleGAN step first.")
    
    logger.info(f"[Xray2CT] Found X-ray input at: {xray_input_path}")
    
    # 2. Lấy CT Ground Truth từ LIDC
    patient_dir = DATASET_DIR / patient_id
    if not patient_dir.exists():
        logger.error(f"[Xray2CT] CT ground truth not found for {patient_id}")
        raise HTTPException(status_code=404, detail=f"CT ground truth not found for {patient_id}")
    
    dcm_files = sorted(patient_dir.glob("*.dcm"))
    logger.info(f"[Xray2CT] Found {len(dcm_files)} CT slices as ground truth")
    
    # 3. Load X-ray image
    xray_img = Image.open(xray_input_path).convert('L')
    xray_array = np.array(xray_img)
    
    buffer = io.BytesIO()
    xray_img.save(buffer, format='PNG')
    xray_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    # 4. Load sample CT slices as ground truth preview
    ct_slices_preview = []
    sample_indices = [0, len(dcm_files)//4, len(dcm_files)//2, 3*len(dcm_files)//4, len(dcm_files)-1]
    
    for idx in sample_indices:
        if idx < len(dcm_files):
            try:
                ds = pydicom.dcmread(str(dcm_files[idx]))
                pixel_array = ds.pixel_array.astype(np.float32)
                
                # Apply HU transformation
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                
                # Apply lung window
                window_center, window_width = -600, 1500
                img_min = window_center - window_width / 2
                img_max = window_center + window_width / 2
                pixel_array = np.clip(pixel_array, img_min, img_max)
                pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                
                ct_img = Image.fromarray(pixel_array)
                ct_img = ct_img.resize((256, 256))
                
                buffer = io.BytesIO()
                ct_img.save(buffer, format='PNG')
                ct_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                ct_slices_preview.append({
                    "slice_index": idx,
                    "slice_number": int(dcm_files[idx].stem.split('-')[-1]) if '-' in dcm_files[idx].stem else idx,
                    "image_data": f"data:image/png;base64,{ct_base64}"
                })
            except Exception as e:
                logger.warning(f"[Xray2CT] Error loading slice {idx}: {str(e)}")
    
    # 5. Tạo thông tin training pair
    training_pair = {
        "patient_id": patient_id,
        "input": {
            "type": "X-ray (CycleGAN realistic)",
            "path": str(xray_input_path),
            "image_data": f"data:image/png;base64,{xray_base64}",
            "shape": [xray_img.width, xray_img.height]
        },
        "ground_truth": {
            "type": "CT Volume (LIDC-IDRI)",
            "path": str(patient_dir),
            "total_slices": len(dcm_files),
            "preview_slices": ct_slices_preview
        },
        "training_info": {
            "description": "Xray2CT model học cách reconstruct CT 3D từ X-ray 2D",
            "input_shape": f"({xray_img.width}, {xray_img.height})",
            "output_shape": f"(512, 512, {len(dcm_files)})",
            "loss_function": "L1 + Perceptual Loss",
            "model_type": "3D U-Net with Diffusion"
        }
    }
    
    logger.info(f"[Xray2CT] Training data prepared successfully")
    
    return {
        "status": "success",
        "message": "Training data pair prepared",
        "training_pair": training_pair
    }


@router.post("/xray2ct/reconstruct/{patient_id}")
async def reconstruct_ct_real(patient_id: str):
    """
    Tái tạo CT 3D từ X-ray sử dụng Xray2CT model THẬT
    Sử dụng GPU inference với model đã train
    """
    logger.info(f"[Xray2CT] Starting CT reconstruction for patient: {patient_id}")
    
    # Tìm ảnh input (từ CycleGAN hoặc DRR)
    cyclegan_path = RESULTS_DIR / "cyclegan" / patient_id / f"{patient_id}_cyclegan.png"
    drr_path = DIFFDRR_FINAL_RESULTS / f"{patient_id}.png"
    
    input_path = cyclegan_path if cyclegan_path.exists() else drr_path
    logger.info(f"[Xray2CT] Input path: {input_path}")
    
    if not input_path.exists():
        logger.error(f"[Xray2CT] Input image not found")
        raise HTTPException(status_code=404, detail="Input image not found. Please run previous steps first.")
    
    output_dir = RESULTS_DIR / "xray2ct" / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lấy CT ground truth để so sánh
    patient_dir = DATASET_DIR / patient_id
    gt_dcm_files = sorted(patient_dir.glob("*.dcm")) if patient_dir.exists() else []
    
    try:
        # Thêm path để import Xray2CT modules
        if str(XRAY2CT_DIR) not in sys.path:
            sys.path.insert(0, str(XRAY2CT_DIR))
        
        logger.info(f"[Xray2CT] XRAY2CT_DIR: {XRAY2CT_DIR}")
        
        # Kiểm tra có model checkpoint không
        checkpoint_dir = XRAY2CT_DIR / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
        
        logger.info(f"[Xray2CT] Looking for checkpoints in: {checkpoint_dir}")
        logger.info(f"[Xray2CT] Found checkpoints: {[f.name for f in checkpoint_files]}")
        
        if checkpoint_files:
            # Có model thật - chạy inference
            logger.info(f"[Xray2CT] Running real inference with model...")
            
            try:
                from inference import XrayToCTPAInference
                
                # Load model config
                config = {
                    'denoising_fn': 'Unet3D',
                    'diffusion_img_size': 64,
                    'diffusion_depth_size': 64,
                    'diffusion_num_channels': 4,
                    'dim_mults': (1, 2, 4, 8),
                    'timesteps': 1000,
                    'vae_ckpt': str(XRAY2CT_DIR / 'checkpoints' / 'vae'),
                    'dataset_min_value': -12.911299,
                    'dataset_max_value': 9.596558,
                    'medclip': True,
                    'classifier_free_guidance': False,
                    'name_dataset': 'LIDC'
                }
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"[Xray2CT] Using device: {device}")
                
                # Initialize model
                model = XrayToCTPAInference(
                    model_checkpoint=str(checkpoint_files[0]),
                    model_config=config,
                    device=device
                )
                
                # Run inference
                result = model.generate_from_xray(
                    xray_path=str(input_path),
                    output_path=str(output_dir / f"{patient_id}_reconstructed.nii.gz"),
                    save_slices=True,
                    save_gif=True
                )
                
                # Prepare output
                slices_preview = []
                slice_files = sorted(output_dir.glob("slice_*.png"))
                for i, slice_file in enumerate(slice_files[:8]):  # First 8 slices
                    with open(slice_file, 'rb') as f:
                        slices_preview.append({
                            "slice_number": i,
                            "image_data": f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
                        })
                
                # Preview image
                preview_path = output_dir / f"{patient_id}_ct_preview.png"
                if preview_path.exists():
                    with open(preview_path, 'rb') as f:
                        preview_base64 = base64.b64encode(f.read()).decode()
                else:
                    preview_base64 = slices_preview[len(slices_preview)//2]["image_data"].split(",")[1] if slices_preview else ""
                
                return {
                    "patient_id": patient_id,
                    "status": "completed",
                    "message": f"CT reconstruction completed with real Xray2CT model on {device.upper()}",
                    "input_path": str(input_path),
                    "output_dir": str(output_dir),
                    "preview_image": f"data:image/png;base64,{preview_base64}",
                    "slices_preview": slices_preview,
                    "total_slices": len(slice_files),
                    "device_used": device,
                    "model_used": "Xray2CT 3D U-Net Diffusion",
                    "ground_truth_available": len(gt_dcm_files) > 0,
                    "ground_truth_slices": len(gt_dcm_files)
                }
                
            except Exception as model_error:
                logger.warning(f"[Xray2CT] Model inference failed: {str(model_error)}")
                logger.info(f"[Xray2CT] Falling back to demo mode...")
        
        # Fallback: Demo mode với visualization
        logger.info(f"[Xray2CT] Running in demo mode (no trained model checkpoint)")
        
        xray_img = Image.open(input_path).convert('L')
        xray_array = np.array(xray_img)
        
        # Load ground truth CT slices
        gt_slices_preview = []
        if gt_dcm_files:
            sample_indices = np.linspace(0, len(gt_dcm_files)-1, 8, dtype=int)
            for idx in sample_indices:
                try:
                    ds = pydicom.dcmread(str(gt_dcm_files[idx]))
                    pixel_array = ds.pixel_array.astype(np.float32)
                    
                    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                        pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                    
                    window_center, window_width = -600, 1500
                    img_min = window_center - window_width / 2
                    img_max = window_center + window_width / 2
                    pixel_array = np.clip(pixel_array, img_min, img_max)
                    pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    
                    ct_img = Image.fromarray(pixel_array)
                    ct_img = ct_img.resize((256, 256))
                    
                    buffer = io.BytesIO()
                    ct_img.save(buffer, format='PNG')
                    gt_slices_preview.append({
                        "slice_number": int(idx),
                        "image_data": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
                    })
                except Exception as e:
                    logger.warning(f"[Xray2CT] Error loading GT slice {idx}: {str(e)}")
        
        # Create X-ray preview
        buffer = io.BytesIO()
        xray_img_resized = xray_img.resize((256, 256))
        xray_img_resized.save(buffer, format='PNG')
        xray_preview_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "patient_id": patient_id,
            "status": "demo_mode",
            "message": "Demo mode: Showing training data pair (X-ray input + CT ground truth)",
            "training_data": {
                "input": {
                    "type": "X-ray (from CycleGAN)",
                    "path": str(input_path),
                    "image_data": f"data:image/png;base64,{xray_preview_base64}",
                    "description": "Realistic X-ray generated by CycleGAN from DiffDRR synthetic X-ray"
                },
                "ground_truth": {
                    "type": "CT Volume (LIDC-IDRI)",
                    "total_slices": len(gt_dcm_files),
                    "slices_preview": gt_slices_preview,
                    "description": "Original CT scan from LIDC-IDRI dataset used as ground truth for training"
                }
            },
            "training_objective": {
                "description": "Model learns to reconstruct CT volume from X-ray",
                "input_shape": f"({xray_img.width}, {xray_img.height})",
                "output_shape": f"(512, 512, {len(gt_dcm_files)})",
                "loss": "L1 Loss + Perceptual Loss between predicted CT and ground truth CT"
            },
            "note": "To run real inference, train the Xray2CT model and place checkpoint in DoAnPtit_Xray2CT/checkpoints/"
        }
        
    except Exception as e:
        logger.error(f"[Xray2CT] Exception: {str(e)}")
        import traceback
        logger.error(f"[Xray2CT] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error in CT reconstruction: {str(e)}")


@router.get("/xray2ct/result/{patient_id}")
async def get_xray2ct_result(patient_id: str):
    """Lấy kết quả Xray2CT đã tạo"""
    output_dir = RESULTS_DIR / "xray2ct" / patient_id
    
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Xray2CT result not found.")
    
    slices = []
    for slice_file in sorted(output_dir.glob("slice_*.png")):
        with open(slice_file, 'rb') as f:
            slices.append({
                "filename": slice_file.name,
                "slice_number": int(slice_file.stem.split('_')[1]),
                "image_data": f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
            })
    
    preview_path = output_dir / f"{patient_id}_ct_preview.png"
    preview_base64 = None
    if preview_path.exists():
        with open(preview_path, 'rb') as f:
            preview_base64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    
    return {
        "patient_id": patient_id,
        "preview_image": preview_base64,
        "slices": slices,
        "total_slices": len(slices),
        "output_dir": str(output_dir)
    }


# ==================== Pipeline Status ====================

@router.get("/pipeline-status/{patient_id}")
async def get_pipeline_status(patient_id: str):
    """Kiểm tra trạng thái của pipeline cho một bệnh nhân"""
    status = {
        "patient_id": patient_id,
        "steps": {
            "lidc": {"status": "pending", "result": None},
            "diffdrr": {"status": "pending", "result": None},
            "cyclegan": {"status": "pending", "result": None},
            "xray2ct": {"status": "pending", "result": None}
        }
    }
    
    # Check LIDC
    patient_dir = DATASET_DIR / patient_id
    if patient_dir.exists():
        dcm_count = len(list(patient_dir.glob("*.dcm")))
        status["steps"]["lidc"]["status"] = "completed"
        status["steps"]["lidc"]["result"] = {"num_slices": dcm_count}
    
    # Check DiffDRR
    drr_path = DIFFDRR_FINAL_RESULTS / f"{patient_id}.png"
    if not drr_path.exists():
        drr_path = DIFFDRR_RESULTS / f"{patient_id}.png"
    if drr_path.exists():
        status["steps"]["diffdrr"]["status"] = "completed"
        status["steps"]["diffdrr"]["result"] = {"path": str(drr_path)}
    
    # Check CycleGAN
    cyclegan_path = RESULTS_DIR / "cyclegan" / patient_id / f"{patient_id}_cyclegan.png"
    if cyclegan_path.exists():
        status["steps"]["cyclegan"]["status"] = "completed"
        status["steps"]["cyclegan"]["result"] = {"path": str(cyclegan_path)}
    
    # Check Xray2CT
    xray2ct_dir = RESULTS_DIR / "xray2ct" / patient_id
    if xray2ct_dir.exists() and list(xray2ct_dir.glob("slice_*.png")):
        status["steps"]["xray2ct"]["status"] = "completed"
        status["steps"]["xray2ct"]["result"] = {"path": str(xray2ct_dir)}
    
    return status
