# X-ray2CTPA Model Evaluation

## Evaluation Results

### 📊 Image Quality Metrics

- **SSIM**: 0.7017 (**Fair**)
- **PSNR**: 24.79 dB (**Fair**)

### 📈 Analysis and Assessment

#### SSIM (Structural Similarity Index)
- **Value**: 0.7017/1.0
- **Meaning**: Measures structural similarity of images
- **Assessment**: Achieves **Fair** level. The model can accurately reconstruct anatomical structures.

#### PSNR (Peak Signal-to-Noise Ratio)  
- **Value**: 24.79 dB
- **Meaning**: Measures signal-to-noise ratio, higher is better
- **Assessment**: Achieves **Fair** level. Generated image quality is relatively close to ground truth.

## 🎥 Visual Comparison

### CT Axial View (Cross-sectional)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Axial](visualization_output_for_LIDC-IDRI-0001/gt_axial.gif) | ![Pred Axial](visualization_output_for_LIDC-IDRI-0001/pred_axial.gif) |

### CT Sagittal View (Longitudinal) 

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Sagittal](visualization_output_for_LIDC-IDRI-0001/gt_sagittal.gif) | ![Pred Sagittal](visualization_output_for_LIDC-IDRI-0001/pred_sagittal.gif) |

### CT Coronal View (Front-to-Back)

| Ground Truth | Prediction |
|:------------:|:----------:|
| ![GT Coronal](visualization_output_for_LIDC-IDRI-0001/gt_coronal.gif) | ![Pred Coronal](visualization_output_for_LIDC-IDRI-0001/pred_coronal.gif) |

### Comprehensive Comparison

![Comparison](visualization_output/comparison.png)



## 🏥 Medical Metrics

- **Lung Dice**: 0.9062 (Excellent)
- **Soft Tissue Dice**: 0.8672 (Excellent)
- **Medical Quality**: **Excellent** (Avg Dice: 0.8867)

## 🔍 Conclusion

The X-ray2CTPA model achieves **fair** results with:

- ✅ **Anatomical structure** accurately reconstructed (SSIM 0.702)
- ✅ **Image quality** at fair level (PSNR 24.8 dB)
- ✅ **Medical accuracy** excellent (Avg Dice: 0.8867)

### Evaluation Scale

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| **SSIM** | ≥ 0.9 | ≥ 0.8 | ≥ 0.6 | < 0.6 |
| **PSNR** | ≥ 30 dB | ≥ 25 dB | ≥ 20 dB | < 20 dB |
| **Dice** | ≥ 0.8 | ≥ 0.6 | ≥ 0.4 | < 0.4 |

## 🚀 Usage

Run evaluation with visualization:

```bash
python simple_evaluation_with_viz.py \
  --gt-file "/path/to/ground_truth.npy" \
  --pred-file "/path/to/prediction.npy" \
  --create-readme --create-viz \
  --output results.json
```

## 📊 Technical Details

- **MSE**: 0.013743
- **MAE**: 0.074269
- **Orientation**: Fixed alignment issue with transpose(1,2,0)
- **Format**: Compatible with LIDC training pipeline
- **Visualizations**: visualization_output

---
*Results automatically generated from simple_evaluation_with_viz.py* 