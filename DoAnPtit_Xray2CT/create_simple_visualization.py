#!/usr/bin/env python3
"""
Script đơn giản tạo visualization với 3 hướng view và comparison
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import từ preprocess_lidc.py
from preprocess.preprocess_lidc import CONTRAST_HU_MIN, CONTRAST_HU_MAX

def load_and_normalize_ct(file_path, apply_transpose=None):
    """Load và normalize CT về training format [-1,1]"""
    volume = np.load(file_path)
    print(f"Original shape: {volume.shape}, range: [{volume.min():.3f}, {volume.max():.3f}]")
    
    # Apply transpose nếu cần
    if apply_transpose:
        volume = np.transpose(volume, apply_transpose)
        print(f"After transpose{apply_transpose}: {volume.shape}")
    
    # Normalize về [-1,1] format
    if volume.min() >= -1.1 and volume.max() <= 1.1:
        normalized = volume.copy()
        print("Data already in training format [-1, 1]")
    elif volume.min() >= 0 and volume.max() <= 255:
        normalized = (volume / 255.0) * 2 - 1
        print("Converted uint8 to training format")
    else:
        normalized = (volume - volume.min()) / (volume.max() - volume.min())
        normalized = normalized * 2 - 1
        print("Applied min-max normalization")
    
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    return normalized

def denormalize_to_hu(normalized_volume):
    """Denormalize từ [-1,1] về HU values"""
    volume = (normalized_volume + 1.0) / 2.0
    volume = np.clip(volume, 0.0, 1.0)
    volume = volume * (CONTRAST_HU_MAX - CONTRAST_HU_MIN) + CONTRAST_HU_MIN
    return volume.astype(np.float32)

def apply_windowing(image: np.ndarray, wl: int, ww: int) -> np.ndarray:
    """Apply windowing to CT image"""
    image = image.astype(np.float32)
    upper = wl + ww // 2
    lower = wl - ww // 2
    windowed = np.clip(image.copy(), lower, upper)
    windowed = (windowed - lower) / (upper - lower) * 255.0
    return windowed.astype(np.uint8)

def video_tensor_to_gif(tensor, path, duration=120):
    """Convert tensor thành GIF animation"""
    print(f"   Creating GIF: {path}")
    
    # Normalize to [0,1]
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    
    # Convert to PIL frames
    frames = []
    for i in range(tensor.shape[1]):  # Iterate over frames
        frame = tensor[0, i, :, :]  # Get (height, width)
        frame = frame.unsqueeze(0)  # Add channel -> (1, height, width)
        frames.append(T.ToPILImage()(frame))
    
    # Save GIF
    if frames:
        first_img = frames[0]
        rest_imgs = frames[1:]
        first_img.save(path, save_all=True, append_images=rest_imgs,
                       duration=duration, loop=0, optimize=True)

def create_3_view_gifs(ct_volume, output_dir, prefix, wl=-600, ww=1500):
    """Tạo GIF cho 3 hướng view: axial, sagittal, coronal với rotation cho axial và coronal"""
    
    # Denormalize to HU
    ct_hu = denormalize_to_hu(ct_volume)
    
    views = {
        "axial": ct_hu,  # (depth, height, width) - nhìn từ trên xuống
        "sagittal": np.transpose(ct_hu, (2, 0, 1)),  # (width, depth, height) - nhìn từ bên
        "coronal": np.transpose(ct_hu, (1, 0, 2))   # (height, depth, width) - nhìn từ trước/sau
    }
    
    for view_name, volume in views.items():
        # Apply windowing với handling rotation
        processed_slices = []
        
        for i in range(volume.shape[0]):
            slice_img = apply_windowing(volume[i], wl, ww)
            
            # Xoay ảnh axial và coronal sang trái 90 độ để có chiều dài nằm dưới
            if view_name in ["axial", "coronal"]:
                # Rotate 90 degrees counterclockwise (sang trái)
                slice_img = np.rot90(slice_img, k=1)  # k=1 means 90 degrees counterclockwise
            
            processed_slices.append(slice_img)
        
        # Convert list to numpy array với shape phù hợp
        if processed_slices:
            windowed_volume = np.stack(processed_slices, axis=0)
        else:
            windowed_volume = np.zeros_like(volume, dtype=np.float32)
        
        # Convert to tensor (1, frames, height, width)
        tensor = torch.from_numpy(windowed_volume).float().unsqueeze(0)
        
        # Create GIF
        gif_path = output_dir / f"{prefix}_{view_name}.gif"
        video_tensor_to_gif(tensor, str(gif_path), duration=150)

def create_comparison_image(gt_volume, pred_volume, output_path):
    """Tạo ảnh so sánh GT vs Prediction cho 3 view với rotation cho axial và coronal"""
    
    # Denormalize both volumes
    gt_hu = denormalize_to_hu(gt_volume)
    pred_hu = denormalize_to_hu(pred_volume)
    
    # Window setting (lung window)
    wl, ww = -600, 1500
    
    # Get middle slices for each view
    axial_idx = gt_volume.shape[0] // 2
    sagittal_idx = gt_volume.shape[2] // 2  
    coronal_idx = gt_volume.shape[1] // 2
    
    # Create 2x3 grid: GT (top row), Pred (bottom row), 3 views (columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Axial view - với rotation
    gt_axial = apply_windowing(gt_hu[axial_idx], wl, ww)
    pred_axial = apply_windowing(pred_hu[axial_idx], wl, ww)
    
    # Xoay axial sang trái 90 độ
    gt_axial = np.rot90(gt_axial, k=1)
    pred_axial = np.rot90(pred_axial, k=1)
    
    axes[0, 0].imshow(gt_axial, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f'GT - Axial View\nSlice {axial_idx}')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(pred_axial, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f'Pred - Axial View\nSlice {axial_idx}')
    axes[1, 0].axis('off')
    
    # Sagittal view - không xoay (giữ nguyên)
    gt_sagittal = apply_windowing(gt_hu[:, :, sagittal_idx], wl, ww)
    pred_sagittal = apply_windowing(pred_hu[:, :, sagittal_idx], wl, ww)
    
    axes[0, 1].imshow(gt_sagittal, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f'GT - Sagittal View\nSlice {sagittal_idx}')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(pred_sagittal, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title(f'Pred - Sagittal View\nSlice {sagittal_idx}')
    axes[1, 1].axis('off')
    
    # Coronal view - với rotation
    gt_coronal = apply_windowing(gt_hu[:, coronal_idx, :], wl, ww)
    pred_coronal = apply_windowing(pred_hu[:, coronal_idx, :], wl, ww)
    
    # Xoay coronal sang trái 90 độ
    gt_coronal = np.rot90(gt_coronal, k=1)
    pred_coronal = np.rot90(pred_coronal, k=1)
    
    axes[0, 2].imshow(gt_coronal, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(f'GT - Coronal View\nSlice {coronal_idx}')
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(pred_coronal, cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title(f'Pred - Coronal View\nSlice {coronal_idx}')
    axes[1, 2].axis('off')
    
    plt.suptitle('GT vs Prediction Comparison - 3 Views (Lung Window)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def calculate_metrics(gt_volume, pred_volume):
    """Tính toán metrics đánh giá"""
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    ssim_score = ssim(gt_volume, pred_volume, data_range=2.0)
    psnr_score = psnr(gt_volume, pred_volume, data_range=2.0)
    mse_score = np.mean((gt_volume - pred_volume) ** 2)
    mae_score = np.mean(np.abs(gt_volume - pred_volume))
    
    return {
        'SSIM': ssim_score,
        'PSNR': psnr_score,
        'MSE': mse_score,
        'MAE': mae_score
    }

def main():
    print("🎯 SIMPLE CT VISUALIZATION - 3 VIEWS + COMPARISON")
    print("=" * 55)
    
    # File paths - CẬP NHẬT ĐƯỜNG DẪN MỚI
    gt_path = "/home/quydat09/Music/Disk_D_SSD_2.5_Sata/X-ray2CTPA/CT_GT/LIDC-IDRI-0001.npy"
    pred_path = "/home/quydat09/Music/Disk_D_SSD_2.5_Sata/X-ray2CTPA/CT_PRE/generated_ctpa_raw.npy"
    output_dir = "./ct_simple_visualization"
    
    # Load data
    print("📂 Loading data...")
    gt_volume = load_and_normalize_ct(gt_path)
    pred_volume = load_and_normalize_ct(pred_path, apply_transpose=(1, 2, 0))  # CORRECTED!
    
    print(f"\nGT shape: {gt_volume.shape}")
    print(f"Pred shape (corrected): {pred_volume.shape}")
    
    # Verify shapes match
    if gt_volume.shape != pred_volume.shape:
        print("❌ Shapes don't match!")
        return
    
    print("✅ Shapes match!")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📹 Creating 3-view GIFs...")
    print("🔄 Axial và Coronal sẽ được xoay sang trái 90° để có chiều dài nằm dưới")
    print("--- GT GIFs ---")
    create_3_view_gifs(gt_volume, output_dir, "GT")
    
    print("--- Prediction GIFs ---")
    create_3_view_gifs(pred_volume, output_dir, "PRED")
    
    print("\n🖼️  Creating comparison image...")
    comparison_path = output_dir / "GT_vs_PRED_comparison.png"
    create_comparison_image(gt_volume, pred_volume, str(comparison_path))
    print(f"✅ Created: {comparison_path}")
    
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(gt_volume, pred_volume)
    
    print("\n" + "="*40)
    print("📊 EVALUATION METRICS:")
    print("="*40)
    for metric_name, value in metrics.items():
        if metric_name in ['SSIM']:
            print(f"   {metric_name}: {value:.4f}")
        elif metric_name in ['PSNR']:
            print(f"   {metric_name}: {value:.2f} dB")
        else:
            print(f"   {metric_name}: {value:.6f}")
    print("="*40)
    
    print(f"\n🎉 RESULTS SAVED TO: {output_dir}")
    print("\nFiles created:")
    print("- GT_axial.gif, GT_sagittal.gif, GT_coronal.gif (axial & coronal đã xoay 90°)")
    print("- PRED_axial.gif, PRED_sagittal.gif, PRED_coronal.gif (axial & coronal đã xoay 90°)") 
    print("- GT_vs_PRED_comparison.png (axial & coronal đã xoay 90°)")
    print("- Metrics calculated and displayed")
    print("\n📝 NOTE: Axial và Coronal views đã được xoay sang trái 90° để có tỷ lệ đẹp hơn")

if __name__ == "__main__":
    main() 