import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import SimpleITK as sitk
from pathlib import Path
import cv2
from typing import Union, Tuple, Optional
from PIL import Image
import torch
from torchvision import transforms as T

# Medical imaging constants từ training pipeline - GIỐNG HỆT LIDC DATASET
CONTRAST_HU_MIN = -1200.0  # Từ preprocess/preprocess_lidc.py line 22
CONTRAST_HU_MAX = 600.0    # Từ preprocess/preprocess_lidc.py line 23
AIR_HU_VAL = -1000.0

# Medical windowing presets chuẩn
MEDICAL_WINDOWS = {
    'lung': {'center': -600, 'width': 1500, 'name': 'Lung Window'},
    'soft_tissue': {'center': 50, 'width': 350, 'name': 'Soft Tissue'},
    'bone': {'center': 400, 'width': 1500, 'name': 'Bone Window'},
    'brain': {'center': 40, 'width': 80, 'name': 'Brain Window'},
    'liver': {'center': 60, 'width': 160, 'name': 'Liver Window'},
    'mediastinum': {'center': 50, 'width': 350, 'name': 'Mediastinum'},
    'pe_detection': {'center': 100, 'width': 700, 'name': 'PE Detection'},
    'angiography': {'center': 300, 'width': 600, 'name': 'CT Angiography'}
}

def normalize_hu_values(volume, hu_min=CONTRAST_HU_MIN, hu_max=CONTRAST_HU_MAX):
    """
    Normalize HU values theo training pipeline
    """
    volume = volume.astype(np.float32)
    volume = (volume - hu_min) / (hu_max - hu_min)
    volume = np.clip(volume, 0., 1.)
    return volume

def denormalize_hu_values(volume, hu_min=CONTRAST_HU_MIN, hu_max=CONTRAST_HU_MAX):
    """
    Denormalize về HU values thực tế
    """
    volume = volume.astype(np.float32)
    volume = volume * (hu_max - hu_min) + hu_min
    return volume

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

def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    """
    Convert tensor thành GIF animation GIỐNG HỆT REPO GỐC
    Tensor format: (channels, frames, height, width)
    
    ⚠️ QUAN TRỌNG: Function này PHẢI GIỐNG HỆT ddpm/diffusion.py line 1199-1205
    """
    # Normalize GIỐNG HỆT repo gốc: tensor.min() -> tensor.max() thành 0->1
    # CHÍNH XÁC GIỐNG ddpm/diffusion.py line 1200
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    
    # Chuyển thành PIL images - GIỐNG HỆT ddpm/diffusion.py line 1201
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    
    # Save GIF - GIỐNG HỆT ddpm/diffusion.py line 1202-1204
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images

class CTSliceViewer:
    """
    Simple CT slice viewer cho medical imaging với CTPA-specific features
    """
    
    def __init__(self, ct_volume: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Initialize CT slice viewer
        
        Args:
            ct_volume: 3D numpy array (depth, height, width) với HU values
            spacing: Voxel spacing in mm (x, y, z)
        """
        self.ct_volume = ct_volume.astype(np.float32)
        self.spacing = spacing
        self.current_slice = ct_volume.shape[0] // 2
        
        # Detect if volume has proper HU values
        self.has_hu_values = self._detect_hu_values()
        
        # Medical imaging windowing parameters - default to soft tissue
        self.window_center = 50
        self.window_width = 350
        self.current_preset = 'soft_tissue'
        
        # Volume statistics
        self.volume_min = float(ct_volume.min())
        self.volume_max = float(ct_volume.max())
        self.volume_mean = float(ct_volume.mean())
        self.volume_std = float(ct_volume.std())
        
        # Figure and axes
        self.fig = None
        self.ax = None
        self.im = None
        
        print(f"📊 CT Volume loaded:")
        print(f"   Shape: {ct_volume.shape}")
        print(f"   Range: [{self.volume_min:.2f}, {self.volume_max:.2f}]")
        print(f"   Mean±STD: {self.volume_mean:.2f}±{self.volume_std:.2f}")
        print(f"   HU values detected: {self.has_hu_values}")
        
    def _detect_hu_values(self) -> bool:
        """
        Detect if volume contains proper HU values
        """
        # Check if values are in typical HU range
        if self.ct_volume.min() < -500 and self.ct_volume.max() > 200:
            return True
        # Check if values are in normalized range
        elif self.ct_volume.min() >= -1 and self.ct_volume.max() <= 1:
            return False
        # Check if values are in [0, 1] range
        elif self.ct_volume.min() >= 0 and self.ct_volume.max() <= 1:
            return False
        else:
            # Assume HU values if range is reasonable
            return abs(self.ct_volume.max() - self.ct_volume.min()) > 100
    
    def apply_windowing(self, image: np.ndarray, center: float, width: float) -> np.ndarray:
        """
        Apply medical imaging windowing (window/level adjustment)
        """
        img_min = center - width / 2
        img_max = center + width / 2
        
        # Clip values to window range
        windowed = np.clip(image.copy(), img_min, img_max)
        
        # Normalize to 0-255 for display
        if img_max > img_min:
            windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            windowed = np.zeros_like(windowed, dtype=np.uint8)
            
        return windowed
    
    def apply_preset_windowing(self, preset_name: str):
        """
        Apply predefined medical windowing preset
        """
        if preset_name in MEDICAL_WINDOWS:
            preset = MEDICAL_WINDOWS[preset_name]
            self.window_center = preset['center']
            self.window_width = preset['width']
            self.current_preset = preset_name
            print(f"Applied {preset['name']}: W={self.window_width}, C={self.window_center}")
        else:
            print(f"Unknown preset: {preset_name}")
    
    def show_interactive_viewer(self):
        """
        Display interactive slice viewer với medical controls
        """
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, left=0.1, right=0.95)
        
        # Initial display
        windowed_slice = self.apply_windowing(
            self.ct_volume[self.current_slice], 
            self.window_center, 
            self.window_width
        )
        self.im = self.ax.imshow(windowed_slice, cmap='gray', aspect='equal')
        self.ax.set_title(f'CT Slice {self.current_slice}/{self.ct_volume.shape[0]-1} | {MEDICAL_WINDOWS[self.current_preset]["name"]}')
        
        # Slice navigation slider
        ax_slice = plt.axes([0.1, 0.15, 0.5, 0.03])
        self.slider_slice = Slider(
            ax_slice, 'Slice', 0, self.ct_volume.shape[0]-1, 
            valinit=self.current_slice, valfmt='%d'
        )
        
        # Window center slider
        ax_center = plt.axes([0.1, 0.10, 0.5, 0.03])
        self.slider_center = Slider(
            ax_center, 'Window Center', -1000, 1000, 
            valinit=self.window_center, valfmt='%d'
        )
        
        # Window width slider
        ax_width = plt.axes([0.1, 0.05, 0.5, 0.03])
        self.slider_width = Slider(
            ax_width, 'Window Width', 1, 2000, 
            valinit=self.window_width, valfmt='%d'
        )
        
        # Medical preset buttons
        button_width = 0.08
        button_height = 0.03
        start_x = 0.65
        start_y = 0.15
        
        preset_buttons = {}
        for i, (preset_key, preset_info) in enumerate(list(MEDICAL_WINDOWS.items())[:4]):  # Chỉ hiển thị 4 preset chính
            y = start_y - i * (button_height + 0.01)
            
            ax_btn = plt.axes([start_x, y, button_width, button_height])
            btn = Button(ax_btn, preset_info['name'][:8], fontsize=8)  # Rút ngắn tên
            preset_buttons[preset_key] = btn
        
        # Volume info display
        info_text = f"Volume: {self.ct_volume.shape}\nRange: [{self.volume_min:.1f}, {self.volume_max:.1f}]\nMean: {self.volume_mean:.1f}±{self.volume_std:.1f}"
        plt.figtext(0.65, 0.05, info_text, fontsize=10, verticalalignment='bottom')
        
        # Event handlers
        def update_slice(val):
            self.current_slice = int(self.slider_slice.val)
            self.update_display()
            
        def update_window_center(val):
            self.window_center = self.slider_center.val
            self.current_preset = 'custom'
            self.update_display()
            
        def update_window_width(val):
            self.window_width = self.slider_width.val
            self.current_preset = 'custom'
            self.update_display()
        
        # Create preset button handlers
        def create_preset_handler(preset_name):
            def handler(event):
                self.apply_preset_windowing(preset_name)
                self.slider_center.set_val(self.window_center)
                self.slider_width.set_val(self.window_width)
            return handler
        
        # Connect events
        self.slider_slice.on_changed(update_slice)
        self.slider_center.on_changed(update_window_center)
        self.slider_width.on_changed(update_window_width)
        
        for preset_key, btn in preset_buttons.items():
            btn.on_clicked(create_preset_handler(preset_key))
        
        plt.show()
    
    def update_display(self):
        """
        Update the displayed slice với current windowing
        """
        windowed_slice = self.apply_windowing(
            self.ct_volume[self.current_slice], 
            self.window_center, 
            self.window_width
        )
        self.im.set_array(windowed_slice)
        
        # Update title với current preset info
        preset_name = MEDICAL_WINDOWS.get(self.current_preset, {}).get('name', 'Custom')
        self.ax.set_title(
            f'CT Slice {self.current_slice}/{self.ct_volume.shape[0]-1} | '
            f'{preset_name} | W/C: {self.window_width:.0f}/{self.window_center:.0f}'
        )
        self.fig.canvas.draw()
    
    def create_animation(self, interval: int = 100, save_path: Optional[str] = None, 
                        window_preset: str = 'soft_tissue') -> animation.FuncAnimation:
        """
        Create animated GIF với medical windowing - sử dụng repo gốc settings
        """
        # Apply preset windowing
        if window_preset in MEDICAL_WINDOWS:
            preset = MEDICAL_WINDOWS[window_preset]
            center, width = preset['center'], preset['width']
            preset_name = preset['name']
        else:
            center, width = self.window_center, self.window_width
            preset_name = 'Current'
        
        if save_path:
            print(f"🔄 Saving animation với {preset_name} windowing...")
            
            # Sử dụng volume gốc (không denormalize) để giữ chất lượng giống repo gốc
            ctpa_for_gif = self.ct_volume.copy()
            print(f"📊 Volume for GIF range: [{ctpa_for_gif.min():.3f}, {ctpa_for_gif.max():.3f}]")
            
            # Convert to tensor format for repo gốc
            # Repo gốc expects (channels, frames, height, width)
            gif_tensor = torch.from_numpy(ctpa_for_gif).unsqueeze(0).float()  # Add channel dimension
            
            # Use repo gốc function với duration=120 (default)
            video_tensor_to_gif(gif_tensor, save_path, duration=120)
            print(f"✅ Animation saved to {save_path}")
        
        # Also create matplotlib animation for display
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def animate(frame):
            windowed_slice = self.apply_windowing(
                self.ct_volume[frame], center, width
            )
            ax.clear()
            ax.imshow(windowed_slice, cmap='gray', aspect='equal')
            ax.set_title(f'CT Slice {frame}/{self.ct_volume.shape[0]-1} | {preset_name}')
            ax.axis('off')
        
        anim = animation.FuncAnimation(
            fig, animate, frames=self.ct_volume.shape[0], 
            interval=interval, repeat=True
        )
            
        return anim
    
    def save_slices(self, output_dir: str, formats: list = ['npy', 'nii', 'png']):
        """
        Save CT data với medical imaging standards theo training pipeline
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        if 'npy' in formats:
            # Save raw data
            np.save(output_path / 'ct_volume_raw.npy', self.ct_volume)
            print(f"✅ Saved raw .npy: {output_path / 'ct_volume_raw.npy'}")
        
        if 'nii' in formats:
            # Save as NIfTI với proper medical metadata theo training
            ct_image = sitk.GetImageFromArray(self.ct_volume.astype(np.float32))
            ct_image.SetSpacing(self.spacing)
            ct_image.SetOrigin([0.0, 0.0, 0.0])
            ct_image.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            
            sitk.WriteImage(ct_image, str(output_path / 'ct_volume_medical.nii.gz'))
            print(f"✅ Saved medical NIfTI: {output_path / 'ct_volume_medical.nii.gz'}")
        
        if 'png' in formats:
            # Save representative slices với proper normalization
            slice_dir = output_path / 'slices'
            slice_dir.mkdir(exist_ok=True)
            
            # Use denormalize_for_display để consistency
            volume_for_png = denormalize_for_display(self.ct_volume)
            
            # Save every 5th slice
            step = max(1, self.ct_volume.shape[0] // 20)
            for i in range(0, self.ct_volume.shape[0], step):
                if self.has_hu_values:
                    # Apply medical windowing nếu có HU values
                    windowed_slice = self.apply_windowing(
                        self.ct_volume[i], 50, 350  # Soft tissue window
                    )
                else:
                    # Nếu không có HU values, chuyển về 0-255 range
                    slice_img = volume_for_png[i]
                    windowed_slice = (slice_img * 255).astype(np.uint8)
                
                cv2.imwrite(str(slice_dir / f'slice_{i:03d}.png'), windowed_slice)
            
            print(f"✅ Saved PNG slices: {slice_dir}")

def load_and_view_ct(file_path: Union[str, Path], **kwargs) -> CTSliceViewer:
    """
    Function để load và view CT data
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npy':
        ct_volume = np.load(file_path)
    elif file_path.suffix in ['.nii', '.gz']:
        ct_image = sitk.ReadImage(str(file_path))
        ct_volume = sitk.GetArrayFromImage(ct_image)
        # Get spacing from NIfTI metadata
        spacing = ct_image.GetSpacing()[::-1]  # Reverse for numpy array ordering
        kwargs['spacing'] = spacing
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    viewer = CTSliceViewer(ct_volume, **kwargs)
    return viewer

def quick_view_ct(file_path: str, window_preset: str = 'soft_tissue'):
    """
    Quick function để view CT với specific windowing preset
    """
    viewer = load_and_view_ct(file_path)
    viewer.apply_preset_windowing(window_preset)
    viewer.show_interactive_viewer()

def compare_orientations(volume, save_dir="./orientation_comparison"):
    """
    So sánh nhiều orientations khác nhau của cùng một volume
    Tạo PNG slices để so sánh trực quan
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    transpose_methods = {
        'none': lambda x: x,
        'dhw_to_whd': lambda x: x.transpose(1, 2, 0),  # (D,H,W) -> (W,H,D)
        'dhw_to_wdh': lambda x: x.transpose(1, 0, 2),  # (D,H,W) -> (W,D,H)
        'dhw_to_hdw': lambda x: x.transpose(2, 0, 1),  # (D,H,W) -> (H,D,W)
        'original_training': lambda x: x.transpose(2, 1, 0),  # (D,H,W) -> (W,H,D)
    }
    
    print(f"🔍 Comparing orientations for volume shape: {volume.shape}")
    
    # Normalize volume for display
    volume_norm = denormalize_for_display(volume)
    
    for method_name, transpose_func in transpose_methods.items():
        print(f"🔄 Processing {method_name}...")
        
        # Apply transpose
        volume_transposed = transpose_func(volume_norm)
        print(f"   Shape after transpose: {volume_transposed.shape}")
        
        # Create directory for this method
        method_dir = os.path.join(save_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        
        # Save middle slices from each axis
        if len(volume_transposed.shape) == 3:
            d, h, w = volume_transposed.shape
            
            # Axial slice (middle of first axis)
            axial_slice = volume_transposed[d//2, :, :]
            axial_img = (axial_slice * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(method_dir, f"axial_slice.png"), axial_img)
            
            # Coronal slice (middle of second axis)
            coronal_slice = volume_transposed[:, h//2, :]
            coronal_img = (coronal_slice * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(method_dir, f"coronal_slice.png"), coronal_img)
            
            # Sagittal slice (middle of third axis)
            sagittal_slice = volume_transposed[:, :, w//2]
            sagittal_img = (sagittal_slice * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(method_dir, f"sagittal_slice.png"), sagittal_img)
            
            print(f"   Saved slices: axial ({axial_slice.shape}), coronal ({coronal_slice.shape}), sagittal ({sagittal_slice.shape})")
    
    print(f"✅ Orientation comparison completed!")
    print(f"📁 Check images in: {save_dir}")
    print(f"   Compare with original DICOM slices to find correct orientation")
    
    return save_dir

if __name__ == "__main__":
    # Example usage
    print("CT Slice Viewer Utility for Medical Imaging")
    print("Available functions:")
    print("1. quick_view_ct('path/to/ct.npy', window_preset='soft_tissue')")
    print("2. load_and_view_ct('path/to/ct.npy').show_interactive_viewer()")
    print("\nAvailable windowing presets:")
    for key, info in MEDICAL_WINDOWS.items():
        print(f"   - {key}: {info['name']} (W={info['width']}, C={info['center']})") 