#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline: DiffDRR → CycleGAN cho 1 bệnh nhân
Tạo ảnh X-ray thật từ CT scan của 1 bệnh nhân LIDC

Pipeline:
1. Đọc CT DICOM từ folder bệnh nhân
2. Tạo ảnh DRR (X-ray giả) bằng DiffDRR → resize về 256x256
3. Chuyển đổi DRR sang X-ray thật bằng CycleGAN

Output: Chỉ 2 file
- {patient_id}_drr.png: Ảnh DRR đã resize 256x256
- {patient_id}_xray_real.png: Ảnh X-ray sau CycleGAN

Usage:
    python test_single_patient.py
    python test_single_patient.py --patient LIDC-IDRI-0129 --gpu 0
"""

import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np

# Thêm các thư mục vào path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
DIFFDRR_DIR = os.path.join(BACKEND_DIR, 'DoAnPtit_DiffDrr')

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(DIFFDRR_DIR, 'notebooks', 'tutorials'))

# Import CycleGAN converter
from cyclegan_converter import CycleGANConverter


def check_environment():
    """Kiểm tra môi trường"""
    print("=" * 60)
    print("KIỂM TRA MÔI TRƯỜNG")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    

def create_drr_from_ct(ct_folder, output_path, device, 
                       sdd=1020.0, height=200, delx=2.0,
                       target_size=256):
    """
    Tạo ảnh DRR từ CT scan và resize về target_size
    
    Args:
        ct_folder: Đường dẫn folder chứa DICOM files
        output_path: Đường dẫn lưu ảnh DRR (đã resize)
        device: torch device
        sdd: Source-to-detector distance (mm)
        height: Kích thước ảnh DRR gốc
        delx: Pixel spacing
        target_size: Kích thước ảnh output (sẽ resize về target_size x target_size)
        
    Returns:
        str: Đường dẫn ảnh DRR đã resize
    """
    from diffdrr.data import read
    from diffdrr.drr import DRR
    
    print(f"\n[DiffDRR] Đang đọc CT từ: {ct_folder}")
    
    # Đọc CT volume
    subject = read(
        ct_folder,
        labelmap=None,
        labels=None,
        orientation="AP",
        bone_attenuation_multiplier=1.0
    )
    
    # Khởi tạo DRR
    print("[DiffDRR] Đang tạo ảnh DRR...")
    drr = DRR(
        subject,
        sdd=sdd,
        height=height,
        delx=delx,
    ).to(device)
    
    # Góc chiếu
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    # Tạo DRR
    with torch.no_grad():
        img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
    
    # Chuyển tensor sang numpy
    img_np = img.cpu().numpy()
    if len(img_np.shape) == 4:
        img_np = img_np[0, 0]  # (B, C, H, W) -> (H, W)
    elif len(img_np.shape) == 3:
        img_np = img_np[0]  # (B, H, W) -> (H, W)
    
    # Normalize về [0, 255]
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255
    img_np = img_np.astype(np.uint8)
    
    # Tạo ảnh PIL grayscale
    img_pil = Image.fromarray(img_np, mode='L')
    
    # Resize về target_size x target_size
    img_resized = img_pil.resize((target_size, target_size), Image.Resampling.BICUBIC)
    
    # Convert sang RGB (CycleGAN cần 3 channels)
    img_rgb = img_resized.convert('RGB')
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_rgb.save(output_path)
    
    print(f"[DiffDRR] Đã lưu DRR: {output_path}")
    print(f"[DiffDRR] Size: {img_rgb.size}, Mode: {img_rgb.mode}")
    
    return output_path


def process_single_patient(patient_folder, output_dir, gpu_id=0):
    """
    Xử lý pipeline đầy đủ cho 1 bệnh nhân
    
    Args:
        patient_folder: Đường dẫn folder CT của bệnh nhân
        output_dir: Thư mục output
        gpu_id: GPU ID (-1 cho CPU)
        
    Returns:
        dict: Các đường dẫn output
    """
    # Kiểm tra GPU
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        
    patient_id = os.path.basename(patient_folder)
    print(f"\n{'='*60}")
    print(f"XỬ LÝ BỆNH NHÂN: {patient_id}")
    print(f"{'='*60}")
    
    # Đếm file DICOM
    dcm_files = [f for f in os.listdir(patient_folder) if f.endswith('.dcm')]
    print(f"Số file DICOM: {len(dcm_files)}")
    
    # Tạo thư mục output
    patient_output_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # ==================== BƯỚC 1: TẠO DRR ====================
    print("\n" + "-"*40)
    print("BƯỚC 1: TẠO ẢNH DRR TỪ CT (resize 256x256)")
    print("-"*40)
    
    drr_output_path = os.path.join(patient_output_dir, f"{patient_id}_drr.png")
    
    drr_path = create_drr_from_ct(
        ct_folder=patient_folder,
        output_path=drr_output_path,
        device=device,
        sdd=1020.0,
        height=200,
        delx=2.0,
        target_size=256
    )
    
    # ==================== BƯỚC 2: CYCLEGAN ====================
    print("\n" + "-"*40)
    print("BƯỚC 2: CHUYỂN ĐỔI DRR SANG X-RAY THẬT")
    print("-"*40)
    
    cyclegan_output_path = os.path.join(patient_output_dir, f"{patient_id}_xray_real.png")
    
    # Khởi tạo CycleGAN converter
    converter = CycleGANConverter(
        model_name='xray_cyclegan',
        checkpoints_dir=os.path.join(SCRIPT_DIR, 'checkpoints'),
        gpu_id=gpu_id,
        load_size=256,
        crop_size=256
    )
    
    # Chuyển đổi
    real_xray_path = converter.convert_single_image(drr_path, cyclegan_output_path)
    
    # ==================== KẾT QUẢ ====================
    print("\n" + "="*60)
    print("HOÀN THÀNH!")
    print("="*60)
    
    results = {
        'patient_id': patient_id,
        'ct_folder': patient_folder,
        'drr_path': drr_path,
        'real_xray_path': real_xray_path,
        'output_dir': patient_output_dir
    }
    
    print(f"Bệnh nhân: {patient_id}")
    print(f"Ảnh DRR (256x256): {drr_path}")
    print(f"Ảnh X-ray thật: {real_xray_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Pipeline DiffDRR + CycleGAN cho 1 bệnh nhân')
    parser.add_argument('--patient', type=str, default='LIDC-IDRI-0001',
                        help='ID bệnh nhân (ví dụ: LIDC-IDRI-0001)')
    parser.add_argument('--data-dir', type=str, 
                        default='/home/quydat09-mic-ace/Documents/PTIT/dataset_LIDC_IDRI_filtered',
                        help='Thư mục chứa dataset LIDC')
    parser.add_argument('--output-dir', type=str,
                        default=None,
                        help='Thư mục output')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (-1 cho CPU)')
    
    args = parser.parse_args()
    
    # Kiểm tra môi trường
    check_environment()
    
    # Đường dẫn folder bệnh nhân
    patient_folder = os.path.join(args.data_dir, args.patient)
    
    if not os.path.exists(patient_folder):
        print(f"[ERROR] Không tìm thấy folder: {patient_folder}")
        print(f"\nCác bệnh nhân có sẵn:")
        for p in sorted(os.listdir(args.data_dir)):
            print(f"  - {p}")
        return
    
    # Thư mục output
    if args.output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, 'results', 'pipeline_output')
    else:
        output_dir = args.output_dir
        
    # Chạy pipeline
    results = process_single_patient(patient_folder, output_dir, args.gpu)
    
    return results


if __name__ == '__main__':
    main()
