#!/usr/bin/env python3
"""
Script đánh giá đơn giản cho X-ray2CTPA model với visualization tích hợp
Kết hợp simple_evaluation.py và create_simple_visualization.py
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import argparse
import json
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from torchvision import transforms as T

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import từ preprocess_lidc.py
from preprocess.preprocess_lidc import CONTRAST_HU_MIN, CONTRAST_HU_MAX

# Import metrics libraries
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import zoom

class SimpleEvaluatorWithViz:
    """Evaluator đơn giản cho CT generation với visualization"""
    
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
    
    def apply_windowing(self, image: np.ndarray, wl: int, ww: int) -> np.ndarray:
        """Apply windowing to CT image"""
        image = image.astype(np.float32)
        upper = wl + ww // 2
        lower = wl - ww // 2
        windowed = np.clip(image.copy(), lower, upper)
        windowed = (windowed - lower) / (upper - lower) * 255.0
        return windowed.astype(np.uint8)
    
    def video_tensor_to_gif(self, tensor, path, duration=120):
        """Convert tensor thành GIF animation"""
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
    
    def create_3_view_gifs(self, ct_volume, output_dir, prefix, wl=-600, ww=1500):
        """Tạo GIF cho 3 hướng view: axial, sagittal, coronal"""
        
        # Denormalize to HU
        ct_hu = self.denormalize_to_hu(ct_volume)
        
        views = {
            "axial": ct_hu,  # (depth, height, width) - nhìn từ trên xuống
            "sagittal": np.transpose(ct_hu, (2, 0, 1)),  # (width, depth, height) - nhìn từ bên
            "coronal": np.transpose(ct_hu, (1, 0, 2))   # (height, depth, width) - nhìn từ trước/sau
        }
        
        gif_paths = {}
        
        for view_name, volume in views.items():
            # Apply windowing
            windowed_volume = np.zeros_like(volume, dtype=np.float32)
            for i in range(volume.shape[0]):
                windowed_volume[i] = self.apply_windowing(volume[i], wl, ww)
            
            # Convert to tensor (1, frames, height, width)
            tensor = torch.from_numpy(windowed_volume).float().unsqueeze(0)
            
            # Create GIF
            gif_path = output_dir / f"{prefix}_{view_name}.gif"
            self.video_tensor_to_gif(tensor, str(gif_path), duration=150)
            
            # Store relative path for README
            gif_paths[view_name] = f"{output_dir.name}/{prefix}_{view_name}.gif"
        
        return gif_paths
    
    def create_comparison_image(self, gt_volume, pred_volume, output_path):
        """Tạo ảnh so sánh GT vs Prediction cho 3 view"""
        
        # Denormalize both volumes
        gt_hu = self.denormalize_to_hu(gt_volume)
        pred_hu = self.denormalize_to_hu(pred_volume)
        
        # Window setting (lung window)
        wl, ww = -600, 1500
        
        # Get middle slices for each view
        axial_idx = gt_volume.shape[0] // 2
        sagittal_idx = gt_volume.shape[2] // 2  
        coronal_idx = gt_volume.shape[1] // 2
        
        # Create 2x3 grid: GT (top row), Pred (bottom row), 3 views (columns)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Axial view
        gt_axial = self.apply_windowing(gt_hu[axial_idx], wl, ww)
        pred_axial = self.apply_windowing(pred_hu[axial_idx], wl, ww)
        
        axes[0, 0].imshow(gt_axial, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title(f'GT - Axial View\nSlice {axial_idx}')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(pred_axial, cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title(f'Pred - Axial View\nSlice {axial_idx}')
        axes[1, 0].axis('off')
        
        # Sagittal view
        gt_sagittal = self.apply_windowing(gt_hu[:, :, sagittal_idx], wl, ww)
        pred_sagittal = self.apply_windowing(pred_hu[:, :, sagittal_idx], wl, ww)
        
        axes[0, 1].imshow(gt_sagittal, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title(f'GT - Sagittal View\nSlice {sagittal_idx}')
        axes[0, 1].axis('off')
        
        axes[1, 1].imshow(pred_sagittal, cmap='gray', vmin=0, vmax=255)
        axes[1, 1].set_title(f'Pred - Sagittal View\nSlice {sagittal_idx}')
        axes[1, 1].axis('off')
        
        # Coronal view
        gt_coronal = self.apply_windowing(gt_hu[:, coronal_idx, :], wl, ww)
        pred_coronal = self.apply_windowing(pred_hu[:, coronal_idx, :], wl, ww)
        
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
        
        # Return relative path for README
        return f"{output_path.parent.name}/{output_path.name}"
    
    def create_visualizations(self, gt_volume, pred_volume, output_dir):
        """Tạo tất cả visualizations và trả về đường dẫn"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n🎨 Creating visualizations...")
        
        # Create GIFs for GT
        print("   Creating GT GIFs...")
        gt_gifs = self.create_3_view_gifs(gt_volume, output_dir, "gt")
        
        # Create GIFs for Prediction
        print("   Creating Prediction GIFs...")
        pred_gifs = self.create_3_view_gifs(pred_volume, output_dir, "pred")
        
        # Create comparison image
        print("   Creating comparison image...")
        comparison_path = output_dir / "comparison.png"
        comparison_image = self.create_comparison_image(gt_volume, pred_volume, comparison_path)
        
        return {
            'gt_gifs': gt_gifs,
            'pred_gifs': pred_gifs,
            'comparison_image': comparison_image,
            'output_dir': str(output_dir)
        }
    
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
    
    def evaluate_pair(self, gt_path: str, pred_path: str, create_viz=False, output_dir="./visualization_output") -> Dict:
        """Đánh giá một cặp GT-Prediction với optional visualization"""
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
        
        result = {'metrics': metrics}
        
        # Create visualizations if requested
        if create_viz:
            viz_paths = self.create_visualizations(gt, pred, output_dir)
            result['visualizations'] = viz_paths
        
        return result

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

def create_readme(metrics: Dict[str, float], interpretation: Dict[str, str], viz_paths=None) -> str:
    """Tạo nội dung README với kết quả thực tế và đường dẫn ảnh chính xác"""
    
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
    
    # Create visualization section with actual paths
    viz_section = ""
    if viz_paths:
        gt_gifs = viz_paths['gt_gifs']
        pred_gifs = viz_paths['pred_gifs']
        
        viz_section = f"""## 🎥 So Sánh Trực Quan

### CT Axial View (Cắt Ngang)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Axial]({gt_gifs['axial']}) | ![Pred Axial]({pred_gifs['axial']}) |

### CT Sagittal View (Cắt Dọc) 

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Sagittal]({gt_gifs['sagittal']}) | ![Pred Sagittal]({pred_gifs['sagittal']}) |

### CT Coronal View (Cắt Trước-Sau)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Coronal]({gt_gifs['coronal']}) | ![Pred Coronal]({pred_gifs['coronal']}) |

### So Sánh Tổng Hợp

![Comparison]({viz_paths['comparison_image']})

"""
    else:
        viz_section = """## 🎥 So Sánh Trực Quan

*Chạy với --create-viz để tạo visualizations*

"""
    
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

{viz_section}

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

Chạy đánh giá với visualization:

```bash
python simple_evaluation_with_viz.py \\
  --gt-file "/path/to/ground_truth.npy" \\
  --pred-file "/path/to/prediction.npy" \\
  --create-readme --create-viz \\
  --output results.json
```

## 📊 Chi Tiết Kỹ Thuật

- **MSE**: {metrics['MSE']:.6f}
- **MAE**: {metrics['MAE']:.6f}
- **Orientation**: Đã sửa lỗi alignment với transpose(1,2,0)
- **Format**: Tương thích với LIDC training pipeline
{f"- **Visualizations**: {viz_paths['output_dir']}" if viz_paths else ""}

---
*Kết quả được tạo tự động từ simple_evaluation_with_viz.py*
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
    parser = argparse.ArgumentParser(description="Simple CT evaluation for X-ray2CTPA with visualization")
    parser.add_argument("--gt-file", type=str, required=True, help="Ground truth CT file")
    parser.add_argument("--pred-file", type=str, required=True, help="Prediction CT file")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--create-readme", action="store_true", help="Tạo file README_EVALUATION.md")
    parser.add_argument("--create-viz", action="store_true", help="Tạo visualizations (GIF và comparison)")
    parser.add_argument("--viz-output", type=str, default="./visualization_output", help="Thư mục output cho visualizations")
    
    args = parser.parse_args()
    
    print("🎯 SIMPLE X-RAY2CTPA EVALUATION WITH VISUALIZATION")
    print("=" * 55)
    
    # Initialize evaluator
    evaluator = SimpleEvaluatorWithViz()
    
    try:
        # Evaluate
        result = evaluator.evaluate_pair(
            args.gt_file, 
            args.pred_file, 
            create_viz=args.create_viz,
            output_dir=args.viz_output
        )
        
        metrics = result['metrics']
        viz_paths = result.get('visualizations', None)
        interpretation = interpret_results(metrics)
        
        # Print results
        print_results(metrics, interpretation)
        
        # Print visualization info
        if viz_paths:
            print(f"\n🎨 Visualizations created in: {viz_paths['output_dir']}")
            print("   Files created:")
            for view in ['axial', 'sagittal', 'coronal']:
                print(f"   - gt_{view}.gif, pred_{view}.gif")
            print(f"   - comparison.png")
        
        # Save results
        if args.output:
            results = {
                'gt_file': args.gt_file,
                'pred_file': args.pred_file,
                'metrics': metrics,
                'interpretation': interpretation,
                'visualizations': viz_paths,
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
            readme_content = create_readme(metrics, interpretation, viz_paths)
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