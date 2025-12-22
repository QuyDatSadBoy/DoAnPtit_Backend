#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để chuyển đổi ảnh x-ray giả sang ảnh x-ray thật sử dụng CycleGAN đã train sẵn.
Script này sẽ:
1. Sử dụng trực tiếp ảnh đã resize từ thư mục resized_images
2. Tải model CycleGAN từ thư mục checkpoints
3. Chuyển đổi tất cả ảnh
4. Lưu kết quả vào thư mục results
"""

import os
import argparse
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
from PIL import Image
import numpy as np
import glob
import shutil

def convert_xray_images(input_dir, output_dir, model_name='xray_cyclegan', model_path='./checkpoints'):
    """
    Chuyển đổi ảnh x-ray giả sang ảnh x-ray thật sử dụng CycleGAN
    
    Args:
        input_dir: Thư mục chứa ảnh x-ray giả đã resize
        output_dir: Thư mục để lưu ảnh x-ray thật
        model_name: Tên của model
        model_path: Đường dẫn đến thư mục chứa model đã train
    """
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Cấu hình các tham số
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default=input_dir, help='path to images')
    parser.add_argument('--name', type=str, default=model_name, help='name of the experiment')
    parser.add_argument('--model', type=str, default='test', help='chooses which model to use')
    parser.add_argument('--dataset_mode', type=str, default='single', help='chooses how datasets are loaded')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load?')
    parser.add_argument('--results_dir', type=str, default=output_dir, help='saves results here')
    parser.add_argument('--checkpoints_dir', type=str, default=model_path, help='models are saved here')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--eval', action='store_true', help='use eval mode during test time')
    parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    
    # Parse các tham số
    opt = parser.parse_args()
    
    # Cấu hình thêm
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    
    # Thêm thuộc tính isTrain = False vì đây là quá trình test
    opt.isTrain = False
    
    # Thêm thuộc tính model_suffix cần thiết cho test_model.py
    opt.model_suffix = ''
    
    # Thêm các thuộc tính cần thiết cho networks.define_G
    opt.ngf = 64  # số lượng gen filters trong lớp conv cuối cùng
    opt.netG = 'resnet_9blocks'  # architecture của generator
    opt.norm = 'instance'  # normalization layer
    opt.init_type = 'normal'  # initialization method
    opt.init_gain = 0.02  # scaling factor cho initialization
    
    # Cấu hình GPU/CPU
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    
    # Thêm tham số load_iter
    opt.load_iter = 0
    
    # Thêm tham số no_dropout
    opt.no_dropout = True
    
    # Thêm tham số verbose
    opt.verbose = False
    
    # Tạo dataset và model
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    # Chuyển đổi ảnh
    print(f"Bắt đầu chuyển đổi ảnh từ {input_dir} sang {output_dir}")
    
    # Tạo thư mục web để lưu kết quả
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    print('Tạo thư mục web', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    # Chuyển đổi từng ảnh
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        
        # Lưu ảnh đã chuyển đổi
        for label, img in visuals.items():
            # Lấy đường dẫn file gốc
            if isinstance(img_path, list) and len(img_path) > 0:
                path = img_path[0]
            else:
                # Tạo tên file mới nếu không có đường dẫn hợp lệ
                path = f"generated_{i}_{label}.png"
                
            # Lấy tên file từ đường dẫn
            img_name = os.path.basename(path)
            
            # Bỏ qua file có tên không hợp lệ
            if not img_name or img_name == '.':
                img_name = f"generated_{i}_{label}.png"
                
            save_path = os.path.join(output_dir, img_name)
            
            print(f"Đường dẫn gốc: {path}")
            print(f"Tên file: {img_name}")
            print(f"Đường dẫn lưu: {save_path}")
            
            # Đảm bảo tên file có phần mở rộng
            if not os.path.splitext(save_path)[1]:
                save_path = save_path + '.png'
                print(f"Đã thêm phần mở rộng: {save_path}")
            
            # Cách xử lý mới, đơn giản hơn
            img_numpy = img.cpu().float().numpy()
            print(f"Shape của tensor: {img_numpy.shape}")
            
            # Xử lý dựa vào shape thực của tensor
            if len(img_numpy.shape) == 4:  # (B, C, H, W)
                img_numpy = img_numpy[0]  # Lấy batch đầu tiên -> (C, H, W)
            
            # Xử lý dựa vào kênh màu
            if img_numpy.shape[0] == 3:  # RGB (3, H, W)
                img_numpy = np.transpose(img_numpy, (1, 2, 0))  # -> (H, W, 3)
                img_numpy = ((img_numpy + 1) / 2.0 * 255.0).astype(np.uint8)
                img_pil = Image.fromarray(img_numpy)
            else:  # Grayscale
                if len(img_numpy.shape) == 3:
                    # Trường hợp (1, H, W) hoặc (C, H, W)
                    if img_numpy.shape[0] == 1:
                        img_numpy = img_numpy[0]  # -> (H, W)
                    else:
                        # Nếu có nhiều kênh, lấy trung bình
                        img_numpy = np.mean(img_numpy, axis=0)  # -> (H, W)
                
                img_numpy = ((img_numpy + 1) / 2.0 * 255.0).astype(np.uint8)
                img_pil = Image.fromarray(img_numpy, mode='L')
            
            # Lưu ảnh
            img_pil.save(save_path)
            
            print(f'Đã chuyển đổi ảnh {i+1}: {img_name}')
    
    print(f"Hoàn thành chuyển đổi ảnh. Kết quả được lưu tại {output_dir}")
    webpage.save()

if __name__ == '__main__':
    # Đường dẫn đến thư mục chứa ảnh x-ray giả đã resize
    input_dir = './resized_images'
    
    # Đường dẫn để lưu ảnh x-ray thật
    output_dir = './result2/xray_converted'
    
    # Tên model
    model_name = 'xray_cyclegan'
    
    # Đường dẫn đến thư mục chứa model đã train
    model_path = './checkpoints'
    
    # Chuyển đổi ảnh
    convert_xray_images(input_dir, output_dir, model_name, model_path) 