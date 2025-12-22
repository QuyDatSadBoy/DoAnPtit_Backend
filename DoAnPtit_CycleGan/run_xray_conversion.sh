#!/bin/bash

# Script để chuyển đổi ảnh x-ray giả sang ảnh x-ray thật sử dụng CycleGAN đã train sẵn

# Kiểm tra môi trường Conda
if command -v conda &> /dev/null; then
    # Kích hoạt môi trường quydat09
    echo "Kích hoạt môi trường quydat09..."
    eval "$(conda shell.bash hook)"
    conda activate quydat09
    
    # Kiểm tra xem các thư viện cần thiết đã được cài đặt chưa
    if ! python -c "import torch, torchvision, PIL, numpy" &> /dev/null; then
        echo "Cài đặt các thư viện cần thiết..."
        conda install -y pytorch torchvision torchaudio -c pytorch
        conda install -y pillow numpy
    fi
else
    echo "Conda không được tìm thấy. Vui lòng cài đặt Conda trước khi chạy script này."
    exit 1
fi

# Xóa thư mục kết quả cũ nếu tồn tại
if [ -d "./results3/xray_converted" ]; then
    echo "Xóa thư mục kết quả cũ..."
    rm -rf ./results3/xray_converted
fi

# Tạo thư mục kết quả mới
mkdir -p ./results3/xray_converted

# Kiểm tra xem thư mục xray_cyclegan đã tồn tại trong checkpoints chưa
if [ ! -d "./checkpoints/xray_cyclegan" ]; then
    echo "Tạo thư mục xray_cyclegan trong checkpoints..."
    mkdir -p ./checkpoints/xray_cyclegan
fi

# Kiểm tra xem các file model đã tồn tại trong thư mục xray_cyclegan chưa
if [ ! -f "./checkpoints/xray_cyclegan/latest_net_G.pth" ]; then
    echo "Sao chép các file model vào thư mục xray_cyclegan..."
    cp ./checkpoints/latest_net_G_A.pth ./checkpoints/xray_cyclegan/latest_net_G.pth
    cp ./checkpoints/latest_net_G_B.pth ./checkpoints/xray_cyclegan/latest_net_F.pth
    cp ./checkpoints/latest_net_D_A.pth ./checkpoints/xray_cyclegan/latest_net_D_A.pth
    cp ./checkpoints/latest_net_D_B.pth ./checkpoints/xray_cyclegan/latest_net_D_B.pth
fi

# Đếm số lượng ảnh trong thư mục resized_images
total_images=$(ls -1 ./resized_images/*.png | wc -l)
echo "Tổng số ảnh cần xử lý: $total_images"

echo "Bắt đầu chuyển đổi ảnh x-ray..."

# Chạy test.py với các tham số phù hợp
python test.py \
    --dataroot ./resized_images \
    --name xray_cyclegan \
    --model test \
    --dataset_mode single \
    --direction AtoB \
    --results_dir ./results3/xray_converted \
    --checkpoints_dir ./checkpoints \
    --phase test \
    --eval \
    --num_test 1000 \
    --aspect_ratio 1.0 \
    --display_winsize 128 \
    --gpu_ids -1 \
    --input_nc 3 \
    --output_nc 3 \
    --no_dropout \
    --load_size 128 \
    --crop_size 128 \
    --preprocess resize_and_crop

echo "Hoàn thành chuyển đổi ảnh x-ray!"

# Tạo thư mục để lưu ảnh đã chuyển đổi
mkdir -p ./results3/xray_converted/xray-real

# Di chuyển và đổi tên các ảnh fake thành x-ray-real
echo "Đang xử lý kết quả..."
for img in ./results3/xray_converted/xray_cyclegan/test_latest/images/*_fake.png; do
    if [ -f "$img" ]; then
        filename=$(basename "$img")
        # Lấy tên file gốc (không có hậu tố _fake.png)
        base_name="${filename%_fake.png}"
        # Tạo tên file mới với hậu tố x-ray-real.png
        new_filename="${base_name}-x-ray-real.png"
        cp "$img" "./results3/xray_converted/xray-real/$new_filename"
        echo "Đã sao chép $filename thành $new_filename"
    fi
done

# Xóa các ảnh real gốc
for img in ./results3/xray_converted/xray_cyclegan/test_latest/images/*_real.png; do
    if [ -f "$img" ]; then
        rm "$img"
        echo "Đã xóa $img"
    fi
done

# Đếm số lượng ảnh đã chuyển đổi
converted_images=$(ls -1 ./results3/xray_converted/xray-real/*.png | wc -l)
echo "Đã chuyển đổi thành công $converted_images ảnh"
echo "Kết quả cuối cùng được lưu tại ./results3/xray_converted/xray-real" 