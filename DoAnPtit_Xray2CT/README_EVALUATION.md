# Đánh Giá Model X-ray2CTPA

## Kết Quả Đánh Giá

### 📊 Chỉ Số Chất Lượng Hình Ảnh

- **SSIM**: 0.6991 (**Khá**)
- **PSNR**: 24.83 dB (**Khá**)

### 📈 Nhận Xét và Đánh Giá

#### SSIM (Structural Similarity Index)
- **Giá trị**: 0.6991/1.0
- **Ý nghĩa**: Đo độ tương đồng về cấu trúc hình ảnh
- **Đánh giá**: Đạt mức **Khá**. Model có khả năng tái tạo cấu trúc giải phẫu ở mức chấp nhận được.

#### PSNR (Peak Signal-to-Noise Ratio)  
- **Giá trị**: 24.83 dB
- **Ý nghĩa**: Đo tỷ lệ tín hiệu trên nhiễu, càng cao càng tốt
- **Đánh giá**: Đạt mức **Khá**. Chất lượng hình ảnh sinh ra tương đối sát với ground truth.

## 🎥 So Sánh Trực Quan

### CT Axial View (Cắt Ngang)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Axial](visualization_output_for_LIDC-IDRI-0001/gt_axial.gif) | ![Pred Axial](visualization_output_for_LIDC-IDRI-0001/pred_axial.gif) |

### CT Sagittal View (Cắt Dọc) 

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Sagittal](visualization_output_for_LIDC-IDRI-0001/gt_sagittal.gif) | ![Pred Sagittal](visualization_output_for_LIDC-IDRI-0001/pred_sagittal.gif) |

### CT Coronal View (Cắt Trước-Sau)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Coronal](visualization_output_for_LIDC-IDRI-0001/gt_coronal.gif) | ![Pred Coronal](visualization_output_for_LIDC-IDRI-0001/pred_coronal.gif) |

### So Sánh Tổng Hợp

![Comparison](visualization_output_for_LIDC-IDRI-0001/comparison.png)



## 🏥 Chỉ Số Y Tế

- **Lung Dice**: 0.9090 (Xuất sắc)
- **Soft Tissue Dice**: 0.8678 (Xuất sắc)
- **Chất lượng Y tế**: **Xuất sắc** (Dice TB: 0.8884)

## 🔍 Kết Luận

Model X-ray2CTPA cho kết quả **khá** với:

- ✅ **Cấu trúc giải phẫu** được tái tạo khá tốt (SSIM 0.699)
- ✅ **Chất lượng hình ảnh** ở mức khá (PSNR 24.8 dB)
- ✅ **Độ chính xác y tế** xuất sắc (Dice TB: 0.8884)

### Thang Đánh Giá

| Chỉ số | Xuất sắc | Tốt | Khá | Kém |
|--------|----------|-----|-----|-----|
| **SSIM** | ≥ 0.9 | ≥ 0.8 | ≥ 0.6 | < 0.6 |
| **PSNR** | ≥ 30 dB | ≥ 25 dB | ≥ 20 dB | < 20 dB |
| **Dice** | ≥ 0.8 | ≥ 0.6 | ≥ 0.4 | < 0.4 |

## 🚀 Cách Sử Dụng

Chạy đánh giá với visualization:

```bash
python simple_evaluation_with_viz.py \
  --gt-file "/path/to/ground_truth.npy" \
  --pred-file "/path/to/prediction.npy" \
  --create-readme --create-viz \
  --output results.json
```

## 📊 Chi Tiết Kỹ Thuật

- **MSE**: 0.013629
- **MAE**: 0.074501
- **Orientation**: Đã sửa lỗi alignment với transpose(1,2,0)
- **Format**: Tương thích với LIDC training pipeline
- **Visualizations**: visualization_output_for_LIDC-IDRI-0001

---
*Kết quả được tạo tự động từ simple_evaluation_with_viz.py*
