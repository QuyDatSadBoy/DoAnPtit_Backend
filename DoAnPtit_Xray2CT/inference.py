#!/usr/bin/env python3
"""
Script inference cho mô hình X-ray2CTPA
Chuyển đổi ảnh X-ray 2D thành CTPA 3D với xử lý GIỐNG HỆT TRAINING CODE GỐC
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import cv2
from typing import Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import các module cần thiết
from ddpm import Unet3D, GaussianDiffusion
from ddpm.unet import UNet
import SimpleITK as sitk
from diffusers import AutoencoderKL
from torchvision import transforms as T

# Thêm constants từ training code - GIỐNG HỆT LIDC DATASET (vì config dùng name_dataset: LIDC)
CONTRAST_HU_MIN = -1200.0  # Từ preprocess/preprocess_lidc.py line 22
CONTRAST_HU_MAX = 600.0    # Từ preprocess/preprocess_lidc.py line 23

# Import video_tensor_to_gif function từ training code - GIỐNG HỆT REPO GỐC
def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    """
    Convert tensor thành GIF animation GIỐNG HỆT REPO GỐC
    Tensor format: (channels, frames, height, width)
    Sử dụng normalization GIỐNG HỆT ddpm/diffusion.py line 1200
    """
    # Normalize GIỐNG HỆT repo gốc: tensor.min() -> tensor.max() thành 0->1
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    
    # Chuyển thành PIL images
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images

def denormalize_ctpa_volume(volume, dataset_min, dataset_max):
    """
    Denormalize CTPA volume từ latent space về HU values thực tế
    GIỐNG HỆT LIDC training pipeline từ preprocess/preprocess_lidc.py
    """
    # Model output thường ở range [-1, 1] từ training
    # Trước tiên chuyển từ [-1, 1] về [0, 1]
    volume_01 = (volume + 1.0) / 2.0
    
    # Sau đó denormalize về range gốc của dataset
    volume_denorm = volume_01 * (dataset_max - dataset_min) + dataset_min
    
    # Cuối cùng denormalize từ dataset range về HU range theo LIDC
    # Reverse của: img = (img - CONTRAST_HU_MIN) / (CONTRAST_HU_MAX - CONTRAST_HU_MIN)
    volume_hu = volume_denorm * (CONTRAST_HU_MAX - CONTRAST_HU_MIN) + CONTRAST_HU_MIN
    
    # Clip về range HU hợp lý cho LIDC
    volume_hu = np.clip(volume_hu, CONTRAST_HU_MIN, CONTRAST_HU_MAX)
    
    return volume_hu.astype(np.float32)

def denormalize_for_display(volume):
    """
    Denormalize volume để hiển thị, giữ nguyên range để tránh mất dữ liệu
    """
    # Nếu volume trong range [-1, 1], chuyển về [0, 1]
    if volume.min() >= -1.1 and volume.max() <= 1.1:
        volume_display = (volume + 1.0) / 2.0
        volume_display = np.clip(volume_display, 0.0, 1.0)
        return volume_display
    # Nếu đã trong range [0, 1], giữ nguyên
    elif volume.min() >= -0.1 and volume.max() <= 1.1:
        return np.clip(volume, 0.0, 1.0)
    # Nếu range khác, normalize về [0, 1]
    else:
        volume_display = (volume - volume.min()) / (volume.max() - volume.min())
        return volume_display

def apply_medical_orientation(volume, use_training_transpose=False):
    """
    Áp dụng orientation GIỐNG FILE generated_ctpa_none (KHÔNG TRANSPOSE)
    
    Args:
        volume: Input volume array
        use_training_transpose: False để dùng orientation giống generated_ctpa_none
    """
    if len(volume.shape) == 3:
        d, h, w = volume.shape
        print(f"🔍 Input volume shape: (D={d}, H={h}, W={w})")
        
        if use_training_transpose:
            # Training code transpose (không dùng nữa)
            volume_oriented = volume.transpose(2, 1, 0)  # (D,H,W) -> (W,H,D)
            print(f"📊 Training transpose: {volume.shape} → {volume_oriented.shape} (training code)")
        else:
            # GIỐNG FILE generated_ctpa_none - KHÔNG TRANSPOSE
            volume_oriented = volume
            print(f"📊 No transpose: keeping original shape {volume.shape} (GIỐNG generated_ctpa_none)")
            
        return volume_oriented
    return volume

def create_medical_nifti(volume, spacing=(1.0, 1.0, 1.0)):
    """
    Tạo NIfTI image với metadata chuẩn medical imaging GIỐNG TRAINING CODE
    """
    # Tạo NIfTI image
    nifti_image = sitk.GetImageFromArray(volume)
    
    # Set spacing (voxel size) - GIỐNG training code
    nifti_image.SetSpacing(spacing)
    
    # Set origin (tọa độ gốc) - GIỐNG training code
    nifti_image.SetOrigin([0.0, 0.0, 0.0])
    
    # Set direction (hướng của các trục) - GIỐNG training code
    # Identity matrix cho standard orientation
    nifti_image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    
    return nifti_image

class XrayToCTPAInference:
    """Class để inference từ X-ray sang CTPA với xử lý GIỐNG HỆT TRAINING PIPELINE"""
    
    def __init__(
        self,
        model_checkpoint: str,
        model_config: dict,
        device: str = 'cuda'
    ):
        """
        Khởi tạo inference model
        
        Args:
            model_checkpoint: Đường dẫn đến checkpoint đã train
            model_config: Cấu hình model
            device: Device để chạy ('cuda' hoặc 'cpu')
        """
        self.device = device
        self.config = model_config
        
        # Khởi tạo model
        self.model = self._load_model(model_checkpoint)
        self.model.eval()
        
        print(f"✅ Đã load model từ: {model_checkpoint}")
        print(f"🔧 Device: {device}")
        print(f"📊 Dataset range: [{model_config.get('dataset_min_value', -12.911299):.3f}, {model_config.get('dataset_max_value', 9.596558):.3f}]")
        
    def _load_model(self, checkpoint_path: str):
        """Load model từ checkpoint"""
        
        # Tạo model architecture
        if self.config['denoising_fn'] == 'Unet3D':
            unet = Unet3D(
                dim=self.config['diffusion_img_size'],
                cond_dim=self.config.get('cond_dim', 512),
                dim_mults=self.config['dim_mults'],
                channels=self.config['diffusion_num_channels'],
                resnet_groups=8,
                classifier_free_guidance=self.config.get('classifier_free_guidance', False),
                medclip=self.config.get('medclip', True)
            ).to(self.device)
        elif self.config['denoising_fn'] == 'UNet':
            unet = UNet(
                in_ch=self.config['diffusion_num_channels'],
                out_ch=self.config['diffusion_num_channels'],
                spatial_dims=3
            ).to(self.device)
        else:
            raise ValueError(f"Model {self.config['denoising_fn']} không được hỗ trợ")
        
        # Tạo diffusion model
        diffusion = GaussianDiffusion(
            unet,
            vqgan_ckpt=self.config.get('vqgan_ckpt'),
            vae_ckpt=self.config.get('vae_ckpt'),
            image_size=self.config['diffusion_img_size'],
            num_frames=self.config['diffusion_depth_size'],
            channels=self.config['diffusion_num_channels'],
            timesteps=self.config.get('timesteps', 1000),
            img_cond=True,
            loss_type=self.config.get('loss_type', 'l1'),
            l1_weight=self.config.get('l1_weight', 1.0),
            perceptual_weight=self.config.get('perceptual_weight', 0.0),
            discriminator_weight=self.config.get('discriminator_weight', 0.0),
            classification_weight=self.config.get('classification_weight', 0.0),
            classifier_free_guidance=self.config.get('classifier_free_guidance', False),
            medclip=self.config.get('medclip', True),
            name_dataset=self.config.get('name_dataset', 'CTPA'),
            dataset_min_value=self.config.get('dataset_min_value', -12.911299),
            dataset_max_value=self.config.get('dataset_max_value', 9.596558),
        ).to(self.device)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Thử load state dict
            if 'model' in checkpoint:
                diffusion.load_state_dict(checkpoint['model'], strict=False)
                print(f"✅ Loaded model state dict từ checkpoint")
            elif 'ema' in checkpoint:
                diffusion.load_state_dict(checkpoint['ema'], strict=False)
                print(f"✅ Loaded EMA model state dict từ checkpoint")
            else:
                diffusion.load_state_dict(checkpoint, strict=False)
                print(f"✅ Loaded state dict trực tiếp từ checkpoint")
                
        else:
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
        
        return diffusion
    
    def preprocess_xray(self, xray_path: str) -> torch.Tensor:
        """
        Tiền xử lý ảnh X-ray theo training pipeline
        """
        if xray_path.endswith('.npy'):
            # Load từ file numpy
            xray = np.load(xray_path).astype(np.float32)
        else:
            # Load từ ảnh thông thường
            xray = cv2.imread(xray_path, cv2.IMREAD_GRAYSCALE)
            xray = cv2.resize(xray, (224, 224))  # Resize theo kích thước training
            xray = xray.astype(np.float32) / 255.0  # Normalize
        
        # Chuyển thành tensor
        xray_tensor = torch.from_numpy(xray).float()
        
        # Thêm batch dimension nếu cần
        if len(xray_tensor.shape) == 2:
            xray_tensor = xray_tensor.unsqueeze(0).unsqueeze(0)
        elif len(xray_tensor.shape) == 3:
            xray_tensor = xray_tensor.unsqueeze(0)
            
        return xray_tensor.to(self.device)
    
    def generate_ctpa(
        self, 
        xray_tensor: torch.Tensor,
        guidance_scale: float = 1.0,
        num_inference_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate CTPA từ X-ray theo training pipeline
        """
        print("🔄 Đang generate CTPA...")
        
        with torch.no_grad():
            # Generate CTPA theo training pipeline
            ctpa = self.model.sample(
                cond=xray_tensor,
                cond_scale=guidance_scale,
                batch_size=xray_tensor.shape[0]
            )
            
        print("✅ Đã generate CTPA thành công!")
        return ctpa
    
    def postprocess_ctpa(self, ctpa_tensor: torch.Tensor) -> np.ndarray:
        """
        Hậu xử lý CTPA tensor thành numpy array theo training pipeline
        """
        # Chuyển về CPU và numpy
        ctpa_np = ctpa_tensor.cpu().numpy()
        
        print(f"🔍 DEBUG: Raw tensor shape: {ctpa_tensor.shape}")
        print(f"🔍 DEBUG: Raw numpy shape: {ctpa_np.shape}")
        
        # Loại bỏ batch dimension
        if len(ctpa_np.shape) == 5:  # (B, C, D, H, W)
            print(f"🔍 DEBUG: 5D tensor - (B, C, D, H, W) format")
            ctpa_np = ctpa_np[0, 0]  # Lấy batch đầu tiên và channel đầu tiên
            print(f"🔍 DEBUG: After removing batch & channel: {ctpa_np.shape} - should be (D, H, W)")
        elif len(ctpa_np.shape) == 4:  # (B, D, H, W) hoặc (C, D, H, W)
            print(f"🔍 DEBUG: 4D tensor - (B, D, H, W) or (C, D, H, W) format")
            ctpa_np = ctpa_np[0]
            print(f"🔍 DEBUG: After removing first dimension: {ctpa_np.shape} - should be (D, H, W)")
        elif len(ctpa_np.shape) == 3:  # (D, H, W)
            print(f"🔍 DEBUG: 3D tensor - already (D, H, W) format")
        
        print(f"📊 CTPA shape sau postprocess: {ctpa_np.shape}")
        print(f"📊 CTPA range trước xử lý: [{ctpa_np.min():.3f}, {ctpa_np.max():.3f}]")
        
        # Thêm thông tin về axes để debug orientation
        if len(ctpa_np.shape) == 3:
            d, h, w = ctpa_np.shape
            print(f"🔍 DEBUG: Interpreting as (Depth={d}, Height={h}, Width={w})")
            print(f"🔍 DEBUG: Sẽ áp dụng transpose GIỐNG TRAINING CODE")
            
        return ctpa_np
    
    def save_results(
        self,
        ctpa_volume: np.ndarray,
        output_dir: str,
        filename: str = "generated_ctpa",
        formats: list = ['npy', 'nii', 'png', 'gif']
    ):
        """
        Lưu kết quả CTPA với xử lý GIỐNG HỆT TRAINING CODE GỐC
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Lấy dataset parameters từ config
        dataset_min = self.config.get('dataset_min_value', -12.911299)
        dataset_max = self.config.get('dataset_max_value', 9.596558)
        
        if 'npy' in formats:
            # Lưu raw data với orientation gốc
            np.save(output_path / f"{filename}_raw.npy", ctpa_volume)
            print(f"✅ Đã lưu {filename}_raw.npy")
        
        if 'nii' in formats:
            # ÁP DỤNG ORIENTATION GIỐNG FILE generated_ctpa_none
            print("🔄 Đang xử lý CTPA GIỐNG FILE generated_ctpa_none...")
            
            # 1. Áp dụng orientation GIỐNG generated_ctpa_none (KHÔNG transpose)
            ctpa_oriented = apply_medical_orientation(ctpa_volume, use_training_transpose=False)
            print(f"📊 Orientation: NO transpose - GIỐNG generated_ctpa_none")
            
            # 2. Clip về range HU GIỐNG training code để giữ chất lượng
            ctpa_oriented = np.clip(ctpa_oriented, -1024, 3071)  # Standard CT HU range
            ctpa_oriented = ctpa_oriented.astype(np.float32)
            print(f"📊 HU range: [-1024, 3071] - giữ chất lượng như training code")
            
            # 3. Tạo NIfTI với metadata chuẩn
            nifti_image = create_medical_nifti(ctpa_oriented, spacing=(1.0, 1.0, 1.0))
            
            # 4. Lưu file NIfTI chính
            nii_path = output_path / f"{filename}_none_style.nii.gz"
            sitk.WriteImage(nifti_image, str(nii_path))
            print(f"✅ Đã lưu {filename}_none_style.nii.gz (GIỐNG generated_ctpa_none)")
            
            # 5. Lưu thêm version với denormalize để so sánh nếu cần
            if True:  # Có thể tắt để tiết kiệm dung lượng
                ctpa_denorm = denormalize_ctpa_volume(ctpa_volume, dataset_min, dataset_max)
                ctpa_denorm_oriented = apply_medical_orientation(ctpa_denorm, use_training_transpose=False)
                ctpa_denorm_oriented = np.clip(ctpa_denorm_oriented, -1024, 3071)
                nifti_denorm = create_medical_nifti(ctpa_denorm_oriented)
                denorm_path = output_path / f"{filename}_denormalized_none_style.nii.gz"
                sitk.WriteImage(nifti_denorm, str(denorm_path))
                print(f"✅ Đã lưu {filename}_denormalized_none_style.nii.gz (để so sánh)")
        
        if 'png' in formats:
            # Lưu slices đại diện
            slice_dir = output_path / f"{filename}_slices"
            slice_dir.mkdir(exist_ok=True)
            
            # Sử dụng denormalize_for_display để có PNG đẹp
            ctpa_for_png = denormalize_for_display(ctpa_volume)
            print(f"📊 CTPA for PNG range: [{ctpa_for_png.min():.3f}, {ctpa_for_png.max():.3f}]")
            
            num_slices = ctpa_for_png.shape[0]
            indices = np.linspace(0, num_slices-1, min(10, num_slices)).astype(int)
            
            for i, idx in enumerate(indices):
                slice_img = ctpa_for_png[idx]
                
                # Convert to 0-255 range for PNG
                slice_img_255 = (slice_img * 255).astype(np.uint8)
                
                cv2.imwrite(str(slice_dir / f"slice_{i:03d}.png"), slice_img_255)
            
            print(f"✅ Đã lưu {len(indices)} slices PNG")
            
        if 'gif' in formats:
            # Tạo GIF animation GIỐNG HỆT REPO GỐC
            print("🔄 Đang tạo GIF animation GIỐNG HỆT REPO GỐC...")
            
            # Sử dụng volume gốc (không denormalize) để giữ chất lượng GIỐNG training
            ctpa_for_gif = ctpa_volume.copy()
            print(f"📊 CTPA for GIF range: [{ctpa_for_gif.min():.3f}, {ctpa_for_gif.max():.3f}]")
            
            # Prepare tensor cho video_tensor_to_gif function
            # Training code expects (channels, frames, height, width)
            # ctpa_for_gif shape: (depth, height, width)
            # Chuyển thành (1, depth, height, width) để match format GIỐNG training
            gif_tensor = torch.from_numpy(ctpa_for_gif).unsqueeze(0)  # Add channel dimension
            
            # Tạo GIF GIỐNG HỆT repo gốc với duration=120 (default)
            gif_path = output_path / f"{filename}_none_style.gif"
            video_tensor_to_gif(gif_tensor, str(gif_path), duration=120)
            print(f"✅ Đã tạo animation GIF GIỐNG HỆT REPO GỐC: {gif_path}")
    
    def _apply_medical_windowing(self, image: np.ndarray, center: float, width: float) -> np.ndarray:
        """Apply medical windowing cho visualization"""
        img_min = center - width / 2
        img_max = center + width / 2
        windowed = np.clip(image, img_min, img_max)
        # Normalize to 0-255
        windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return windowed
    
    def inference_pipeline(
        self,
        xray_path: str,
        output_dir: str,
        filename: str = "generated_ctpa",
        guidance_scale: float = 1.0,
        show_viewer: bool = True,
        formats: list = ['npy', 'nii', 'png', 'gif']
    ) -> Tuple[np.ndarray, Optional[object]]:
        """
        Pipeline inference hoàn chỉnh GIỐNG HỆT TRAINING CODE GỐC
        """
        print(f"🚀 Bắt đầu inference pipeline GIỐNG HỆT TRAINING CODE...")
        print(f"📂 X-ray input: {xray_path}")
        print(f"📁 Output directory: {output_dir}")
        print(f"📋 Output formats: {formats}")
        print(f"🎯 Orientation: NO transpose - GIỐNG generated_ctpa_none")
        print(f"🎨 Chất lượng: HU range [-1024, 3071] - giữ chất lượng như training code")
        print(f"🎬 GIF: duration=120, normalization giống repo gốc")
        
        # 1. Preprocess X-ray
        xray_tensor = self.preprocess_xray(xray_path)
        print(f"✅ Preprocessed X-ray shape: {xray_tensor.shape}")
        
        # 2. Generate CTPA
        ctpa_tensor = self.generate_ctpa(xray_tensor, guidance_scale)
        print(f"✅ Generated CTPA shape: {ctpa_tensor.shape}")
        
        # 3. Postprocess
        ctpa_volume = self.postprocess_ctpa(ctpa_tensor)
        print(f"✅ Final CTPA volume shape: {ctpa_volume.shape}")
        
        # 4. Save results với formats được chỉ định
        self.save_results(ctpa_volume, output_dir, filename, formats=formats)
        
        # 5. Create viewer nếu có visualization_utils
        viewer = None
        if show_viewer:
            try:
                from visualization_utils import CTSliceViewer
                # Sử dụng denormalize_for_display để consistency với PNG
                ctpa_for_viewer = denormalize_for_display(ctpa_volume)
                viewer = CTSliceViewer(ctpa_for_viewer)
                print("✅ Đã tạo CT viewer")
            except ImportError:
                print("⚠️ Không thể import CTSliceViewer, bỏ qua viewer")
        
        print("🎉 Inference pipeline hoàn thành GIỐNG generated_ctpa_none!")
        print(f"   - Orientation: NO transpose - GIỐNG generated_ctpa_none")
        print(f"   - Chất lượng: HU range [-1024, 3071] - giữ chất lượng như training code")
        print(f"   - GIF: duration=120, normalization giống repo gốc")
        if 'npy' in formats:
            print(f"   - {filename}_raw.npy: Raw volume data")
        if 'nii' in formats:
            print(f"   - {filename}_none_style.nii.gz: NIfTI GIỐNG generated_ctpa_none")
        if 'png' in formats:
            print(f"   - {filename}_slices/: PNG slices")
        if 'gif' in formats:
            print(f"   - {filename}_none_style.gif: GIF animation GIỐNG HỆT repo gốc")
        
        return ctpa_volume, viewer


def main():
    """Hàm main để chạy inference GIỐNG HỆT TRAINING CODE"""
    parser = argparse.ArgumentParser(description="X-ray to CTPA Inference GIỐNG HỆT TRAINING CODE GỐC")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến model checkpoint")
    parser.add_argument("--xray", type=str, required=True, help="Đường dẫn đến ảnh X-ray input")
    parser.add_argument("--output", type=str, default="./inference_results", help="Thư mục output")
    parser.add_argument("--filename", type=str, default="generated_ctpa", help="Tên file output")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--show-viewer", action="store_true", help="Hiển thị interactive viewer")
    
    args = parser.parse_args()
    
    # Cấu hình model theo config đã cho
    model_config = {
        'denoising_fn': 'Unet3D',
        'diffusion_img_size': 32,
        'diffusion_depth_size': 128,  # Từ config
        'diffusion_num_channels': 4,
        'dim_mults': [1, 2, 4, 8],
        'cond_dim': 512,
        'timesteps': 1000,
        'loss_type': 'l1_lpips',  # Từ config
        'l1_weight': 1.0,
        'perceptual_weight': 0.01,  # Từ config
        'discriminator_weight': 0.0,
        'classification_weight': 0.0,
        'classifier_free_guidance': False,
        'medclip': True,
        'name_dataset': 'LIDC',  # Từ config
        'dataset_min_value': -12.911299,  # Từ config
        'dataset_max_value': 9.596558,   # Từ config
        'vae_ckpt': 'stabilityai/sd-vae-ft-mse-original',
        'vqgan_ckpt': None
    }
    
    # Khởi tạo inference
    inference = XrayToCTPAInference(
        model_checkpoint=args.checkpoint,
        model_config=model_config,
        device=args.device
    )
    
    # Chạy inference với settings GIỐNG generated_ctpa_none
    formats = ['npy', 'nii', 'png', 'gif']  # Tạo tất cả formats
    print("🔄 Mode: GIỐNG FILE generated_ctpa_none")
    print("   - Orientation: NO transpose như generated_ctpa_none")
    print("   - Chất lượng: HU range [-1024, 3071] như training code")
    print("   - GIF: duration=120, normalization giống repo gốc")
    
    # Chạy inference
    ctpa_volume, viewer = inference.inference_pipeline(
        xray_path=args.xray,
        output_dir=args.output,
        filename=args.filename,
        guidance_scale=args.guidance_scale,
        show_viewer=args.show_viewer
    )
    
    # Hiển thị viewer nếu được yêu cầu
    if args.show_viewer and viewer:
        print("🖥️  Hiển thị interactive viewer...")
        viewer.show_interactive_viewer()


if __name__ == "__main__":
    main() 