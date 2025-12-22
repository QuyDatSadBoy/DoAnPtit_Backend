#!/usr/bin/env python3
"""
Script đánh giá đơn giản cho X-ray2CTPA model
Tối giản hóa từ evaluation_metrics.py và evaluate_single_pair.py
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import argparse
import json
from typing import Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import từ preprocess_lidc.py
from preprocess.preprocess_lidc import CONTRAST_HU_MIN, CONTRAST_HU_MAX

# Import metrics libraries
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import zoom

class SimpleEvaluator:
    """Evaluator đơn giản cho CT generation"""
    
    def __init__(self):
        self.contrast_hu_min = CONTRAST_HU_MIN
        self.contrast_hu_max = CONTRAST_HU_MAX
    
    def load_and_normalize(self, file_path: str, apply_transpose=None) -> np.ndarray:
        """Load và normalize CT về training format [-1,1]"""
        volume = np.load(file_path)
        
        # Apply transpose nếu cần (cho prediction data)
        if apply_transpose:
            volume = np.transpose(volume, apply_transpose)
        
        # Detect format và normalize về [-1,1]
        if volume.min() >= -1.1 and volume.max() <= 1.1:
            # Already in training format
            normalized = volume.copy()
        elif volume.min() >= 0 and volume.max() <= 255:
            # Uint8 format
            normalized = (volume / 255.0) * 2 - 1
        else:
            # Min-max normalization
            normalized = (volume - volume.min()) / (volume.max() - volume.min())
            normalized = normalized * 2 - 1
        
        return normalized.astype(np.float32)
    
    def handle_shape_mismatch(self, gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle shape mismatch giữa GT và prediction"""
        if gt.shape != pred.shape:
            # Resize prediction để match GT shape
            zoom_factors = np.array(gt.shape) / np.array(pred.shape)
            pred = zoom(pred, zoom_factors, order=1)
        
        return gt, pred
    
    def denormalize_to_hu(self, normalized_volume: np.ndarray) -> np.ndarray:
        """Denormalize từ [-1,1] về HU values"""
        volume = (normalized_volume + 1.0) / 2.0
        volume = np.clip(volume, 0.0, 1.0)
        volume = volume * (self.contrast_hu_max - self.contrast_hu_min) + self.contrast_hu_min
        return volume.astype(np.float32)
    
    def compute_basic_metrics(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """Compute basic metrics: SSIM, PSNR, MSE, MAE"""
        # Normalize về [0,1] cho SSIM và PSNR
        gt_norm = (gt + 1.0) / 2.0
        pred_norm = (pred + 1.0) / 2.0
        
        # SSIM và PSNR slice by slice
        ssim_scores = []
        psnr_scores = []
        
        for i in range(gt_norm.shape[0]):
            ssim_score = ssim(gt_norm[i], pred_norm[i], data_range=1.0)
            psnr_score = psnr(gt_norm[i], pred_norm[i], data_range=1.0)
            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)
        
        # MSE và MAE trong training format space
        mse_val = np.mean((gt - pred) ** 2)
        mae_val = np.mean(np.abs(gt - pred))
        
        return {
            'SSIM': float(np.mean(ssim_scores)),
            'PSNR': float(np.mean(psnr_scores)),
            'MSE': float(mse_val),
            'MAE': float(mae_val)
        }
    
    def compute_medical_metrics(self, gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        """Compute medical-specific metrics"""
        # Convert về HU values
        gt_hu = self.denormalize_to_hu(gt)
        pred_hu = self.denormalize_to_hu(pred)
        
        # Lung window [-1000, 0] HU
        lung_mask_gt = (gt_hu >= -1000) & (gt_hu <= 0)
        lung_mask_pred = (pred_hu >= -1000) & (pred_hu <= 0)
        
        # Soft tissue window [-100, 300] HU  
        soft_mask_gt = (gt_hu >= -100) & (gt_hu <= 300)
        soft_mask_pred = (pred_hu >= -100) & (pred_hu <= 300)
        
        # Dice coefficient
        def dice_coefficient(mask1, mask2):
            intersection = np.logical_and(mask1, mask2).sum()
            return (2. * intersection) / (mask1.sum() + mask2.sum() + 1e-8)
        
        lung_dice = dice_coefficient(lung_mask_gt, lung_mask_pred)
        soft_dice = dice_coefficient(soft_mask_gt, soft_mask_pred)
        
        return {
            'Lung_Dice': float(lung_dice),
            'Soft_Tissue_Dice': float(soft_dice)
        }
    
    def evaluate_pair(self, gt_path: str, pred_path: str) -> Dict[str, float]:
        """Đánh giá một cặp GT-Prediction"""
        print(f"🔍 Evaluating:")
        print(f"   GT: {Path(gt_path).name}")
        print(f"   Pred: {Path(pred_path).name}")
        
        # Load data
        gt = self.load_and_normalize(gt_path)
        pred = self.load_and_normalize(pred_path, apply_transpose=(1, 2, 0))  # CORRECTED!
        
        print(f"   GT shape: {gt.shape}, range: [{gt.min():.3f}, {gt.max():.3f}]")
        print(f"   Pred shape: {pred.shape}, range: [{pred.min():.3f}, {pred.max():.3f}]")
        
        # Handle shape mismatch
        gt, pred = self.handle_shape_mismatch(gt, pred)
        
        # Compute metrics
        basic_metrics = self.compute_basic_metrics(gt, pred)
        medical_metrics = self.compute_medical_metrics(gt, pred)
        
        # Combine results
        metrics = {**basic_metrics, **medical_metrics}
        
        return metrics

def interpret_results(metrics: Dict[str, float]) -> Dict[str, str]:
    """Interpret metrics results"""
    interpretation = {}
    
    # SSIM interpretation
    if metrics['SSIM'] >= 0.9:
        interpretation['SSIM_Quality'] = "Excellent"
    elif metrics['SSIM'] >= 0.8:
        interpretation['SSIM_Quality'] = "Good"
    elif metrics['SSIM'] >= 0.6:
        interpretation['SSIM_Quality'] = "Fair"
    else:
        interpretation['SSIM_Quality'] = "Poor"
    
    # PSNR interpretation
    if metrics['PSNR'] >= 30:
        interpretation['PSNR_Quality'] = "Excellent"
    elif metrics['PSNR'] >= 25:
        interpretation['PSNR_Quality'] = "Good"
    elif metrics['PSNR'] >= 20:
        interpretation['PSNR_Quality'] = "Fair"
    else:
        interpretation['PSNR_Quality'] = "Poor"
    
    # Medical accuracy
    avg_dice = (metrics['Lung_Dice'] + metrics['Soft_Tissue_Dice']) / 2
    if avg_dice >= 0.8:
        interpretation['Medical_Quality'] = "Excellent"
    elif avg_dice >= 0.6:
        interpretation['Medical_Quality'] = "Good"
    elif avg_dice >= 0.4:
        interpretation['Medical_Quality'] = "Fair"
    else:
        interpretation['Medical_Quality'] = "Poor"
    
    interpretation['Average_Dice'] = f"{avg_dice:.4f}"
    
    return interpretation

def create_readme(metrics: Dict[str, float], interpretation: Dict[str, str]) -> str:
    """Tạo nội dung README với kết quả thực tế"""
    
    # Translate quality levels to Vietnamese
    quality_vn = {
        "Excellent": "Xuất sắc",
        "Good": "Tốt", 
        "Fair": "Khá",
        "Poor": "Kém"
    }
    
    ssim_quality_vn = quality_vn.get(interpretation['SSIM_Quality'], interpretation['SSIM_Quality'])
    psnr_quality_vn = quality_vn.get(interpretation['PSNR_Quality'], interpretation['PSNR_Quality'])
    medical_quality_vn = quality_vn.get(interpretation['Medical_Quality'], interpretation['Medical_Quality'])
    
    readme_content = f"""# Đánh Giá Model X-ray2CTPA

## Kết Quả Đánh Giá

### 📊 Chỉ Số Chất Lượng Hình Ảnh

- **SSIM**: {metrics['SSIM']:.4f} (**{ssim_quality_vn}**)
- **PSNR**: {metrics['PSNR']:.2f} dB (**{psnr_quality_vn}**)

### 📈 Nhận Xét và Đánh Giá

#### SSIM (Structural Similarity Index)
- **Giá trị**: {metrics['SSIM']:.4f}/1.0
- **Ý nghĩa**: Đo độ tương đồng về cấu trúc hình ảnh
- **Đánh giá**: Đạt mức **{ssim_quality_vn}**. Model có khả năng tái tạo cấu trúc giải phẫu {"một cách chính xác" if metrics['SSIM'] >= 0.7 else "ở mức chấp nhận được" if metrics['SSIM'] >= 0.6 else "cần cải thiện"}.

#### PSNR (Peak Signal-to-Noise Ratio)  
- **Giá trị**: {metrics['PSNR']:.2f} dB
- **Ý nghĩa**: Đo tỷ lệ tín hiệu trên nhiễu, càng cao càng tốt
- **Đánh giá**: Đạt mức **{psnr_quality_vn}**. Chất lượng hình ảnh sinh ra {"khá sát" if metrics['PSNR'] >= 25 else "tương đối sát" if metrics['PSNR'] >= 20 else "cần cải thiện so"} với ground truth.

## 🎥 So Sánh Trực Quan

### CT Axial View (Cắt Ngang)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Axial](visualization_output/gt_axial.gif) | ![Pred Axial](visualization_output/pred_axial.gif) |

### CT Sagittal View (Cắt Dọc) 

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Sagittal](visualization_output/gt_sagittal.gif) | ![Pred Sagittal](visualization_output/pred_sagittal.gif) |

### CT Coronal View (Cắt Trước-Sau)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Coronal](visualization_output/gt_coronal.gif) | ![Pred Coronal](visualization_output/pred_coronal.gif) |

## 🏥 Chỉ Số Y Tế

- **Lung Dice**: {metrics['Lung_Dice']:.4f} ({quality_vn.get("Excellent" if metrics['Lung_Dice'] >= 0.8 else "Good" if metrics['Lung_Dice'] >= 0.6 else "Fair" if metrics['Lung_Dice'] >= 0.4 else "Poor", "Tốt")})
- **Soft Tissue Dice**: {metrics['Soft_Tissue_Dice']:.4f} ({quality_vn.get("Excellent" if metrics['Soft_Tissue_Dice'] >= 0.8 else "Good" if metrics['Soft_Tissue_Dice'] >= 0.6 else "Fair" if metrics['Soft_Tissue_Dice'] >= 0.4 else "Poor", "Tốt")})
- **Chất lượng Y tế**: **{medical_quality_vn}** (Dice TB: {interpretation['Average_Dice']})

## 🔍 Kết Luận

Model X-ray2CTPA cho kết quả **{ssim_quality_vn.lower()}** với:

- {"✅" if metrics['SSIM'] >= 0.6 else "⚠️"} **Cấu trúc giải phẫu** được tái tạo {"chính xác" if metrics['SSIM'] >= 0.7 else "khá tốt" if metrics['SSIM'] >= 0.6 else "cần cải thiện"} (SSIM {metrics['SSIM']:.3f})
- {"✅" if metrics['PSNR'] >= 20 else "⚠️"} **Chất lượng hình ảnh** ở mức {psnr_quality_vn.lower()} (PSNR {metrics['PSNR']:.1f} dB)
- {"✅" if float(interpretation['Average_Dice']) >= 0.8 else "⚠️"} **Độ chính xác y tế** {medical_quality_vn.lower()} (Dice TB: {interpretation['Average_Dice']})

### Thang Đánh Giá

| Chỉ số | Xuất sắc | Tốt | Khá | Kém |
|--------|----------|-----|-----|-----|
| **SSIM** | ≥ 0.9 | ≥ 0.8 | ≥ 0.6 | < 0.6 |
| **PSNR** | ≥ 30 dB | ≥ 25 dB | ≥ 20 dB | < 20 dB |
| **Dice** | ≥ 0.8 | ≥ 0.6 | ≥ 0.4 | < 0.4 |

## 🚀 Cách Sử Dụng

Chạy đánh giá với:

```bash
python simple_evaluation.py \\
  --gt-file "/path/to/ground_truth.npy" \\
  --pred-file "/path/to/prediction.npy" \\
  --output results.json
```

## 📊 Chi Tiết Kỹ Thuật

- **MSE**: {metrics['MSE']:.6f}
- **MAE**: {metrics['MAE']:.6f}
- **Orientation**: Đã sửa lỗi alignment với transpose(1,2,0)
- **Format**: Tương thích với LIDC training pipeline

---
*Kết quả được tạo tự động từ simple_evaluation.py - {Path().absolute().name}*
"""
    
    return readme_content

def print_results(metrics: Dict[str, float], interpretation: Dict[str, str]):
    """Print evaluation results"""
    print(f"\n📊 EVALUATION RESULTS:")
    print("=" * 40)
    print(f"📈 Basic Image Quality Metrics:")
    print(f"   SSIM: {metrics['SSIM']:.4f} ({interpretation['SSIM_Quality']})")
    print(f"   PSNR: {metrics['PSNR']:.2f} dB ({interpretation['PSNR_Quality']})")
    print(f"   MSE:  {metrics['MSE']:.6f}")
    print(f"   MAE:  {metrics['MAE']:.6f}")
    
    print(f"\n🏥 Medical-Specific Metrics:")
    print(f"   Lung Dice: {metrics['Lung_Dice']:.4f}")
    print(f"   Soft Tissue Dice: {metrics['Soft_Tissue_Dice']:.4f}")
    print(f"   Medical Quality: {interpretation['Medical_Quality']} (Avg Dice: {interpretation['Average_Dice']})")

def main():
    parser = argparse.ArgumentParser(description="Simple CT evaluation for X-ray2CTPA")
    parser.add_argument("--gt-file", type=str, required=True, help="Ground truth CT file")
    parser.add_argument("--pred-file", type=str, required=True, help="Prediction CT file")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--create-readme", action="store_true", help="Tạo file README_EVALUATION.md")
    
    args = parser.parse_args()
    
    print("🎯 SIMPLE X-RAY2CTPA EVALUATION")
    print("=" * 40)
    
    # Initialize evaluator
    evaluator = SimpleEvaluator()
    
    try:
        # Evaluate
        metrics = evaluator.evaluate_pair(args.gt_file, args.pred_file)
        interpretation = interpret_results(metrics)
        
        # Print results
        print_results(metrics, interpretation)
        
        # Save results
        if args.output:
            results = {
                'gt_file': args.gt_file,
                'pred_file': args.pred_file,
                'metrics': metrics,
                'interpretation': interpretation,
                'model_info': {
                    'pipeline': 'X-ray2CTPA',
                    'orientation_corrected': True,
                    'format_handling': 'LIDC training pipeline compatible'
                }
            }
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        
        # Create README
        if args.create_readme:
            readme_content = create_readme(metrics, interpretation)
            with open("README_EVALUATION.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
            print(f"\n📄 README created: README_EVALUATION.md")
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 