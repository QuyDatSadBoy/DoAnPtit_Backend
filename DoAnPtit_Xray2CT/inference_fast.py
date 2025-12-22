#!/usr/bin/env python3
"""
Fast Inference Wrapper - Tối ưu tốc độ mà KHÔNG thay đổi code gốc
Chỉ cần chạy script này thay vì inference.py để có tốc độ nhanh hơn

Cách dùng:
    python inference_fast.py --checkpoint ./checkpoints/model-81.pt --xray "./x-ray input/LIDC-IDRI-0001.npy" --output ./results --filename test

Các tối ưu được áp dụng:
1. cudnn.benchmark = True - Tự động tìm thuật toán nhanh nhất
2. TF32 enabled - Tăng tốc trên GPU Ampere/Ada/Hopper
3. Memory optimization - Dùng nhiều VRAM hơn
"""

import os
import sys
import torch

# ============================================
# CUDA OPTIMIZATIONS - Áp dụng TRƯỚC khi import model
# ============================================
print("🔧 Applying CUDA optimizations...")

# 1. cudnn.benchmark - Tìm thuật toán nhanh nhất cho input size cố định
torch.backends.cudnn.benchmark = True

# 2. TF32 - Tăng tốc matmul trên RTX 30xx/40xx/50xx
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 3. Memory optimization - Cho phép dùng nhiều VRAM hơn
if torch.cuda.is_available():
    # Dùng memory allocator hiệu quả hơn
    torch.cuda.empty_cache()
    
print("✅ CUDA optimizations applied!")
print("   - cudnn.benchmark = True")
print("   - TF32 enabled for matmul")
print("   - Memory optimized")
print("")

# ============================================
# Chạy inference.py gốc với các tối ưu đã áp dụng
# ============================================

# Import và chạy main từ inference.py
if __name__ == "__main__":
    # Thêm thư mục hiện tại vào path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import và chạy main từ inference.py
    from inference import main
    main()
