import os
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import sys
import argparse

# === ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN ===
DRR_DIR = './drr_Xray'
TEMP_DIR = './resized_images'
OUTPUT_DIR = './result2/xray_converted'
MODEL_NAME = 'xray_cyclegan'
MODEL_PATH = './checkpoints'
IMG_SIZE = 128

def convert_npy_to_images(drr_dir=DRR_DIR, temp_dir=TEMP_DIR, img_size=IMG_SIZE):
    """
    Chuyển đổi các file .npy trong thư mục drr_dir thành ảnh và lưu vào temp_dir
    """
    # Tạo thư mục temp nếu chưa tồn tại
    os.makedirs(temp_dir, exist_ok=True)
    
    # Xóa tất cả các file trong thư mục temp nếu có
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Lấy danh sách các file .npy trong thư mục drr_dir
    npy_files = [f for f in os.listdir(drr_dir) if f.endswith('.npy')]
    
    print(f"Đang chuyển đổi {len(npy_files)} file .npy sang ảnh...")
    
    # Chuyển đổi từng file .npy thành ảnh
    for npy_file in tqdm(npy_files):
        # Đọc file .npy
        npy_path = os.path.join(drr_dir, npy_file)
        drr_array = np.load(npy_path)
        
        # Chuẩn hóa giá trị pixel từ 0-1
        if drr_array.min() != drr_array.max():  # Tránh chia cho 0
            normalized_array = (drr_array - drr_array.min()) / (drr_array.max() - drr_array.min())
        else:
            normalized_array = drr_array
        
        # Chuyển sang giá trị 0-255 và định dạng uint8
        img_array = (normalized_array * 255).astype(np.uint8)
        
        # Tạo ảnh từ array
        if len(img_array.shape) == 2:
            # Nếu là grayscale
            img = Image.fromarray(img_array, mode='L')
        elif len(img_array.shape) == 3 and img_array.shape[0] == 1:
            # Nếu có kích thước (1, H, W)
            img = Image.fromarray(img_array[0], mode='L')
        else:
            raise ValueError(f"Không hỗ trợ định dạng array: {img_array.shape}")
        
        # Resize ảnh về kích thước mong muốn
        img = img.resize((img_size, img_size), Image.LANCZOS)
        
        # Lưu ảnh vào thư mục temp
        img_name = os.path.splitext(npy_file)[0] + '.png'
        img_path = os.path.join(temp_dir, img_name)
        img.save(img_path)
    
    print(f"Đã chuyển đổi {len(npy_files)} file .npy thành ảnh và lưu vào {temp_dir}")
    return True

def run_cyclegan_conversion(temp_dir=TEMP_DIR, output_dir=OUTPUT_DIR, model_name=MODEL_NAME, model_path=MODEL_PATH):
    """
    Sử dụng CycleGAN để chuyển đổi ảnh từ temp_dir và lưu kết quả vào output_dir
    """
    # Lưu các tham số dòng lệnh ban đầu
    old_argv = sys.argv
    
    # Thiết lập các tham số dòng lệnh mới
    sys.argv = ['from_data_npy.py', 
                '--dataroot', temp_dir,
                '--name', model_name, 
                '--checkpoints_dir', model_path,
                '--results_dir', output_dir,
                '--model', 'test',
                '--no_dropout',
                '--dataset_mode', 'single',
                '--input_nc', '1',
                '--output_nc', '1',
                '--preprocess', 'resize_and_crop',
                '--eval'
               ]
    
    # Bây giờ import và sử dụng TestOptions
    from options.test_options import TestOptions
    from models import create_model
    from data import create_dataset
    from util.visualizer import save_images
    from util import html
    
    opt = TestOptions().parse()  # Bây giờ sẽ phân tích từ sys.argv mới
    
    # Khôi phục tham số dòng lệnh ban đầu
    sys.argv = old_argv
    
    # Tạo thư mục kết quả nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo dataset và model
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    # Tạo thư mục web để lưu kết quả
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    os.makedirs(web_dir, exist_ok=True)
    print('Tạo thư mục web', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    # Xử lý từng ảnh
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
            
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        
        # Lưu ảnh kết quả
        for label, img in visuals.items():
            img_path_item = img_path[0]
            img_name = os.path.basename(img_path_item)
            save_path = os.path.join(output_dir, img_name)
            
            # Chuyển tensor thành numpy array và lưu ảnh
            img_np = img.cpu().float().numpy()
            img_np = (np.transpose(img_np, (1, 2, 0)) + 1) / 2.0 * 255.0
            img_np = img_np.astype(np.uint8)
            
            # Xử lý ảnh grayscale và RGB
            if img_np.shape[2] == 1:
                img_pil = Image.fromarray(img_np[:,:,0], 'L')
            else:
                img_pil = Image.fromarray(img_np)
                
            img_pil.save(save_path)
            print(f'Đã chuyển đổi ảnh {i+1}: {img_name}')
    
    # Lưu trang web tổng hợp
    save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, 
                width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()
    
    return True

def main():
    print("=== BẮT ĐẦU QUÁ TRÌNH CHUYỂN ĐỔI ===")
    print(f"Thư mục ảnh DRR: {DRR_DIR}")
    print(f"Thư mục ảnh tạm: {TEMP_DIR}")
    print(f"Thư mục kết quả: {OUTPUT_DIR}")
    print(f"Tên model: {MODEL_NAME}")
    print(f"Đường dẫn model: {MODEL_PATH}")
    print("====================================")
    
    # Chuyển đổi các file .npy thành ảnh
    convert_npy_to_images()
    
    # Chạy CycleGAN để chuyển đổi ảnh
    run_cyclegan_conversion()
    
    print(f"Quá trình chuyển đổi hoàn tất. Kết quả được lưu tại {OUTPUT_DIR}")

if __name__ == "__main__":
    main()