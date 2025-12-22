import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# --- Transform configuration từ preprocess_lidcxrayy.py ---
transform_xray = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.633, 0.181),
])

def format_xray(img):
    """
    Function để format X-ray image giống như trong preprocess_lidcxrayy.py
    Output shape: (3, 224, 224) để match với dataset expectation
    """
    # Normalize
    img = transform_xray(img).float()

    #Resize
    img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)

    # Squeeze để có shape (3, 224, 224) giống như preprocess_lidcxrayy.py
    formated_img = img.squeeze().detach().cpu().numpy()
    return formated_img

# --- Cấu hình ---
# Thư mục chứa các file .png đầu vào
input_folder = '/teamspace/studios/this_studio/images_fakeB' 

# Thư mục để lưu các file .npy đầu ra
output_folder = '/teamspace/studios/this_studio/LIDC_dataset_final/XRay_preprocessed_224_224' 

# --- Bắt đầu xử lý ---

# Tạo thư mục đầu ra nếu nó chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

print(f"Bắt đầu quét thư mục: '{input_folder}'")

# Lấy danh sách tất cả các file trong thư mục đầu vào
try:
    all_files = os.listdir(input_folder)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy thư mục '{input_folder}'. Vui lòng tạo thư mục và đặt ảnh vào trong đó.")
    exit()

# Lọc ra chỉ những file có đuôi .png
png_files = [f for f in all_files if f.endswith('.png')]

if not png_files:
    print(f"Không tìm thấy file .png nào trong thư mục '{input_folder}'.")
    exit()

print(f"Tìm thấy {len(png_files)} file .png. Bắt đầu chuyển đổi...")

# Duyệt qua từng file ảnh để xử lý
for file_name in png_files:
    try:
        # Tạo đường dẫn đầy đủ đến file ảnh
        image_path = os.path.join(input_folder, file_name)
        
        # === THÊM CÁC BƯỚC TIỀN XỬ LÝ ===
        # Đọc ảnh và chuyển sang dạng mảng NumPy
        # Giả sử ảnh X-quang là ảnh thang độ xám, ta dùng .convert('L')
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1] range (từ preprocess_lidcxrayy.py)
        img_array /= np.max(img_array) if np.max(img_array) > 0 else 1.0
        
        # Chuyển đổi để có 3 channels như trong preprocess_lidcxrayy.py
        img_array = np.moveaxis(np.repeat(np.expand_dims(img_array, axis=0), 3, axis=0), 0, -1)
        
        print("Original shape:", img_array.shape)
        
        # Resize và normalize với transforms - output shape (224, 224)
        img_array = format_xray(img_array)
        print("After formatting:", img_array.shape)
        
        # Tạo tên file mới bằng cách bỏ đi 11 ký tự cuối
        # Ví dụ: 'LIDC-IDRI-0001_fakeB.png' -> 'LIDC-IDRI-0001'
        new_file_name_base = file_name[:-11] 
        
        # Thêm đuôi .npy vào tên file mới
        new_file_path = os.path.join(output_folder, f"{new_file_name_base}.npy")
        
        # Lưu mảng NumPy thành file .npy
        np.save(new_file_path, img_array)
        
        print(f"Đã chuyển đổi: {file_name} -> {new_file_name_base}.npy")
        
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_name}: {e}")

print(f"\nHoàn tất! Tất cả các file đã được chuyển đổi và lưu vào thư mục '{output_folder}'.")
print("Shape output: (3, 224, 224) - tương thích với LIDC training giống preprocess_lidcxrayy.py")