#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module CycleGAN Converter - Chuyển đổi ảnh X-ray giả sang X-ray thật
Được thiết kế để dễ dàng import vào API

Usage:
    from cyclegan_converter import CycleGANConverter
    
    converter = CycleGANConverter()
    output_path = converter.convert_single_image(input_path, output_path)
"""

import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np

# Thêm thư mục hiện tại vào path để import các module của CycleGAN
CYCLEGAN_DIR = os.path.dirname(os.path.abspath(__file__))
if CYCLEGAN_DIR not in sys.path:
    sys.path.insert(0, CYCLEGAN_DIR)

from models import create_model
from data import create_dataset


class CycleGANConverter:
    """
    Class để chuyển đổi ảnh X-ray giả (từ DiffDRR) sang X-ray thật
    sử dụng model CycleGAN đã được train
    """
    
    def __init__(self, 
                 model_name='xray_cyclegan',
                 checkpoints_dir=None,
                 gpu_id=0,
                 load_size=256,
                 crop_size=256):
        """
        Khởi tạo CycleGAN Converter
        
        Args:
            model_name: Tên model trong thư mục checkpoints
            checkpoints_dir: Đường dẫn đến thư mục checkpoints (default: ./checkpoints)
            gpu_id: ID của GPU để sử dụng (-1 cho CPU)
            load_size: Kích thước resize ảnh
            crop_size: Kích thước crop ảnh
        """
        self.model_name = model_name
        self.checkpoints_dir = checkpoints_dir or os.path.join(CYCLEGAN_DIR, 'checkpoints')
        self.gpu_id = gpu_id
        self.load_size = load_size
        self.crop_size = crop_size
        
        # Khởi tạo model
        self.model = None
        self.opt = None
        self._initialized = False
        
    def _create_options(self):
        """Tạo options cho model"""
        opt = argparse.Namespace()
        
        # Các tham số cơ bản
        opt.name = self.model_name
        opt.model = 'test'
        opt.dataset_mode = 'single'
        opt.direction = 'AtoB'
        opt.epoch = 'latest'
        opt.checkpoints_dir = self.checkpoints_dir
        opt.phase = 'test'
        opt.isTrain = False
        
        # Tham số ảnh
        opt.input_nc = 3
        opt.output_nc = 3
        opt.load_size = self.load_size
        opt.crop_size = self.crop_size
        opt.preprocess = 'resize_and_crop'
        opt.no_flip = True
        
        # Tham số model
        opt.ngf = 64
        opt.netG = 'resnet_9blocks'
        opt.norm = 'instance'
        opt.no_dropout = True
        opt.init_type = 'normal'
        opt.init_gain = 0.02
        opt.model_suffix = ''
        opt.load_iter = 0
        opt.verbose = False
        
        # GPU
        if self.gpu_id >= 0 and torch.cuda.is_available():
            opt.gpu_ids = [self.gpu_id]
        else:
            opt.gpu_ids = []
            
        return opt
    
    def initialize(self):
        """Khởi tạo model (lazy loading)"""
        if self._initialized:
            return
            
        print(f"[CycleGAN] Đang khởi tạo model '{self.model_name}'...")
        
        self.opt = self._create_options()
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        self.model.eval()
        
        self._initialized = True
        print(f"[CycleGAN] Model đã sẵn sàng!")
        
    def preprocess_image(self, image_path, target_size=None):
        """
        Tiền xử lý ảnh: resize, convert sang RGB, normalize
        
        Args:
            image_path: Đường dẫn ảnh input
            target_size: Kích thước target (default: load_size)
            
        Returns:
            torch.Tensor: Tensor ảnh đã xử lý (1, 3, H, W)
        """
        target_size = target_size or self.load_size
        
        # Đọc ảnh
        img = Image.open(image_path)
        
        # Convert RGBA sang RGB nếu cần
        if img.mode == 'RGBA':
            # Tạo background trắng
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3 là alpha channel
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize
        img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
        
        # Convert sang numpy và normalize [-1, 1]
        img_np = np.array(img).astype(np.float32)
        img_np = (img_np / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Convert sang tensor (C, H, W)
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
        
        # Thêm batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def postprocess_image(self, tensor):
        """
        Hậu xử lý tensor output thành ảnh PIL
        
        Args:
            tensor: torch.Tensor output từ model
            
        Returns:
            PIL.Image: Ảnh đã xử lý
        """
        # Chuyển về numpy
        img_np = tensor.cpu().float().numpy()
        
        # Xử lý shape
        if len(img_np.shape) == 4:
            img_np = img_np[0]  # Bỏ batch dimension
            
        # Transpose từ (C, H, W) sang (H, W, C)
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
            
        # Denormalize từ [-1, 1] sang [0, 255]
        img_np = ((img_np + 1) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(img_np)
    
    def convert_single_image(self, input_path, output_path=None):
        """
        Chuyển đổi một ảnh X-ray giả sang X-ray thật
        
        Args:
            input_path: Đường dẫn ảnh input (X-ray từ DiffDRR)
            output_path: Đường dẫn ảnh output (nếu None, tự động tạo)
            
        Returns:
            str: Đường dẫn ảnh output
        """
        # Khởi tạo model nếu chưa
        self.initialize()
        
        # Tạo output path nếu chưa có
        if output_path is None:
            dirname = os.path.dirname(input_path)
            basename = os.path.basename(input_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(dirname, f"{name}_converted{ext}")
            
        # Đảm bảo thư mục output tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"[CycleGAN] Đang chuyển đổi: {input_path}")
        
        # Tiền xử lý ảnh
        img_tensor = self.preprocess_image(input_path)
        
        # Chuyển sang GPU nếu có
        if self.opt.gpu_ids:
            img_tensor = img_tensor.cuda()
            
        # Tạo data dict theo format của CycleGAN
        data = {
            'A': img_tensor,
            'A_paths': input_path
        }
        
        # Chạy inference
        with torch.no_grad():
            self.model.set_input(data)
            self.model.test()
            visuals = self.model.get_current_visuals()
            
        # Lấy ảnh output (fake_B là ảnh đã chuyển đổi)
        if 'fake' in visuals:
            output_tensor = visuals['fake']
        elif 'fake_B' in visuals:
            output_tensor = visuals['fake_B']
        else:
            # Fallback: lấy key đầu tiên không phải 'real'
            for key, value in visuals.items():
                if 'real' not in key.lower():
                    output_tensor = value
                    break
                    
        # Hậu xử lý và lưu
        output_img = self.postprocess_image(output_tensor)
        output_img.save(output_path)
        
        print(f"[CycleGAN] Đã lưu: {output_path}")
        
        return output_path
    
    def convert_batch(self, input_paths, output_dir):
        """
        Chuyển đổi nhiều ảnh
        
        Args:
            input_paths: List đường dẫn ảnh input
            output_dir: Thư mục output
            
        Returns:
            list: Danh sách đường dẫn ảnh output
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = []
        for input_path in input_paths:
            basename = os.path.basename(input_path)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(output_dir, f"{name}_converted{ext}")
            
            self.convert_single_image(input_path, output_path)
            output_paths.append(output_path)
            
        return output_paths


def convert_drr_to_real_xray(input_path, output_path=None, gpu_id=0):
    """
    Hàm tiện ích để chuyển đổi ảnh DRR sang X-ray thật
    
    Args:
        input_path: Đường dẫn ảnh DRR từ DiffDRR
        output_path: Đường dẫn output (optional)
        gpu_id: GPU ID (-1 cho CPU)
        
    Returns:
        str: Đường dẫn ảnh output
    """
    converter = CycleGANConverter(gpu_id=gpu_id)
    return converter.convert_single_image(input_path, output_path)


# ==================== MAIN ====================
if __name__ == '__main__':
    # Test với 1 ảnh DRR
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert DRR to real X-ray using CycleGAN')
    parser.add_argument('--input', type=str, required=True, help='Input DRR image path')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    
    args = parser.parse_args()
    
    output_path = convert_drr_to_real_xray(args.input, args.output, args.gpu)
    print(f"\nKết quả: {output_path}")
