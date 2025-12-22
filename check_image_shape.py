import os
from PIL import Image
import numpy as np

# Đường dẫn thư mục chứa ảnh
# folder_path = "/home/cuongpv/ai-medical/DiffDRR/final_results"
# folder_path = r'/home/cuongpv/ai-medical/cyclegan/pytorch-CycleGAN-and-pix2pix/datasets/synthetic2xray_resized/testA'
folder_path = r'/home/cuongpv/ai-medical/cyclegan/pytorch-CycleGAN-and-pix2pix/results/systhetic2xray/test_latest/images'

# Lấy danh sách file ảnh
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

print(f"Tổng số ảnh: {len(image_files)}")
print("-" * 50)

# Kiểm tra vài ảnh đầu tiên
for i, img_file in enumerate(image_files[:5]):
    img_path = os.path.join(folder_path, img_file)
    
    # Đọc ảnh bằng PIL
    img = Image.open(img_path)
    
    # Chuyển sang numpy array
    img_array = np.array(img)
    
    print(f"File: {img_file}")
    print(f"  Shape: {img_array.shape}")
    print(f"  Dtype: {img_array.dtype}")
    print(f"  Min: {img_array.min()}, Max: {img_array.max()}")
    print()

# Kiểm tra xem tất cả ảnh có cùng shape không
print("-" * 50)
shapes = []
for img_file in image_files:
    img_path = os.path.join(folder_path, img_file)
    img = Image.open(img_path)
    shapes.append(np.array(img).shape)

unique_shapes = set(shapes)
print(f"Các shape khác nhau trong thư mục:")
for shape in unique_shapes:
    count = shapes.count(shape)
    print(f"  Shape {shape}: {count} ảnh")
