"""
Verification script for pure PyTorch implementation.

This script compares the outputs of:
1. Original RegNet (using lagomorph with CUDA)
2. Pure PyTorch RegNet (no CUDA compilation required)

Based on: 2025-11-02-UKB-RegNet-only-motion-analysis.ipynb
"""

import os
import sys
import numpy as np
import torch
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD')

print("=" * 80)
print("PURE PYTORCH IMPLEMENTATION VERIFICATION")
print("=" * 80)
print()

# Setup device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
print()

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# STEP 1: Load test data
# ============================================================================
print("STEP 1: Loading test data")
print("-" * 80)

ukb_data_path = Path('/vast/palmer/pi/kwan/jx332/data/2025-10-12-UKB-sample/unified_analysis_results/segmented/1002514.npy')
print(f"Loading: {ukb_data_path}")

ukb_data = np.load(ukb_data_path, allow_pickle=True).item()
print(f"Subject ID: {ukb_data.get('subject_id', 'N/A')}")

# Find middle SAS series
sas_series = []
for i, series in enumerate(ukb_data['series_list']):
    series_info = series.get('info', {})
    view = series_info.get('view', '').lower()
    if 'sas' in view:
        sas_series.append((i, series))

middle_sas_idx = len(sas_series) // 2
selected_series_idx, selected_series = sas_series[middle_sas_idx]
print(f"Selected SAS series: [{selected_series_idx}] (position {middle_sas_idx+1} of {len(sas_series)})")

# Extract temporal frames
temporal_frames = selected_series['slices']
images = np.array([frame['image'] for frame in temporal_frames])
print(f"Temporal images shape: {images.shape}")

# For quick testing, use only first 10 frames
TEST_FRAMES = 10
images = images[:TEST_FRAMES]
print(f"Using first {TEST_FRAMES} frames for testing: {images.shape}")
print()

# ============================================================================
# STEP 2: Preprocess images (detection + cropping + resizing)
# ============================================================================
print("STEP 2: Preprocessing images")
print("-" * 80)

from skimage.transform import resize
from cardiac_toolkit.detection import WholeHeartDetectorYOLOv9

# Normalize
images_normalized = images.astype(np.float32)
images_normalized = (images_normalized - images_normalized.min()) / \
                    (images_normalized.max() - images_normalized.min() + 1e-8)

# Detect whole heart on first frame
detector_model_path = '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD/cardiac_toolkit/detection/models/whole_heart_yolov9/checkpoints/best.onnx'
detector = WholeHeartDetectorYOLOv9(model_path=detector_model_path)

detection_result = detector.detect([images_normalized[0]], conf_thres=0.7, one_per_class=True)[0]
bbox_original = detection_result['whole_heart'][0]

# Expand and make square
from cardiac_toolkit.detection.utils.bbox_utils import rescale_bbox
bbox_expanded = rescale_bbox(bbox_original, scale_factor=1.5)

def make_bbox_square(bbox, img_height, img_width):
    width = bbox['x2'] - bbox['x1']
    height = bbox['y2'] - bbox['y1']
    bbox_square = bbox.copy()
    if width > height:
        diff = width - height
        bbox_square['y1'] -= diff / 2
        bbox_square['y2'] += diff / 2
    elif height > width:
        diff = height - width
        bbox_square['x1'] -= diff / 2
        bbox_square['x2'] += diff / 2
    bbox_square['x1'] = max(0, bbox_square['x1'])
    bbox_square['y1'] = max(0, bbox_square['y1'])
    bbox_square['x2'] = min(img_width, bbox_square['x2'])
    bbox_square['y2'] = min(img_height, bbox_square['y2'])
    return bbox_square

img_height, img_width = images_normalized.shape[1:3]
bbox_square = make_bbox_square(bbox_expanded, img_height, img_width)

# Crop and resize all frames
cropped_frames = []
for frame in images_normalized:
    x1, y1 = int(bbox_square['x1']), int(bbox_square['y1'])
    x2, y2 = int(bbox_square['x2']), int(bbox_square['y2'])
    cropped = frame[y1:y2, x1:x2]
    cropped_frames.append(cropped)

images_preprocessed = np.array([resize(c, (128, 128), anti_aliasing=True, preserve_range=True)
                                for c in cropped_frames], dtype=np.float32)
print(f"Preprocessed images shape: {images_preprocessed.shape}")
print()

# ============================================================================
# STEP 3: Run CarSON segmentation
# ============================================================================
print("STEP 3: Running CarSON segmentation")
print("-" * 80)

from cardiac_toolkit.segmentation.models.carson import CarSONSegmentor

carson_model_path = '/gpfs/gibbs/project/kwan/jx332/code/2025-05-DeepStrain/pretrained_models/carson_Jan2021.onnx'
segmentor = CarSONSegmentor(
    model_path=carson_model_path,
    input_format='DHW',
    batch_size=32,
    normalize_per_slice=True
)

seg_mask = segmentor.segment(images_preprocessed)
myocardium_mask = (seg_mask == 2).astype(np.uint8)
print(f"Myocardium mask shape: {myocardium_mask.shape}")
print()

# ============================================================================
# STEP 4: Run ORIGINAL RegNet (with lagomorph)
# ============================================================================
print("STEP 4: Running ORIGINAL RegNet (with lagomorph)")
print("-" * 80)

# Load configuration
regnet_config_fname = '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD/configs/ours-train-registration.json'
regnet_config = json.load(open(regnet_config_fname))

# Build original RegNet
from models import build_model
regnet_original = build_model(regnet_config['networks']['registration'], device).to(device)
print("Original RegNet loaded")

# Prepare input
regnet_input = torch.from_numpy(myocardium_mask[None, None]).to(device).float()
print(f"Input shape: {regnet_input.shape}")

# Run inference
with torch.no_grad():
    output_original = regnet_original.inference_2D_video(regnet_input)

disp_original = output_original['displacement'].detach().cpu()
print(f"Original displacement shape: {disp_original.shape}")
print(f"Original displacement range: [{disp_original.min():.4f}, {disp_original.max():.4f}]")
print()

# ============================================================================
# STEP 5: Run PURE PYTORCH RegNet
# ============================================================================
print("STEP 5: Running PURE PYTORCH RegNet (no lagomorph)")
print("-" * 80)

# Import pure PyTorch version
sys.path.insert(0, '/home/jx332/project/code/2025-08-LaMoD/debug_pure_pytorch')
from debug_pure_pytorch.regnet_pytorch import RegNet as RegNetPyTorch

# Build PyTorch RegNet with same config
regnet_pytorch = RegNetPyTorch(regnet_config['networks']['registration']).to(device)
print("Pure PyTorch RegNet created")

# Load the same weights from original model
regnet_pytorch.load_state_dict(regnet_original.state_dict())
print("Weights copied from original model")

# Run inference
with torch.no_grad():
    output_pytorch = regnet_pytorch.inference_2D_video(regnet_input)

disp_pytorch = output_pytorch['displacement'].detach().cpu()
print(f"PyTorch displacement shape: {disp_pytorch.shape}")
print(f"PyTorch displacement range: [{disp_pytorch.min():.4f}, {disp_pytorch.max():.4f}]")
print()

# ============================================================================
# STEP 6: Compare outputs
# ============================================================================
print("STEP 6: Numerical Comparison")
print("=" * 80)
print()

# Compute differences
diff = torch.abs(disp_pytorch - disp_original)
max_diff = diff.max().item()
mean_diff = diff.mean().item()
median_diff = diff.median().item()

# Compute correlation
disp_orig_flat = disp_original.flatten()
disp_pytorch_flat = disp_pytorch.flatten()
correlation = np.corrcoef(disp_orig_flat.numpy(), disp_pytorch_flat.numpy())[0, 1]

print("Displacement Field Comparison:")
print(f"  Max absolute difference:    {max_diff:.6e}")
print(f"  Mean absolute difference:   {mean_diff:.6e}")
print(f"  Median absolute difference: {median_diff:.6e}")
print(f"  Correlation coefficient:    {correlation:.10f}")
print()

# Check if differences are acceptable
TOLERANCE = 1e-4
if max_diff < TOLERANCE:
    print(f"✓ PASS: Max difference ({max_diff:.6e}) < tolerance ({TOLERANCE})")
else:
    print(f"✗ FAIL: Max difference ({max_diff:.6e}) >= tolerance ({TOLERANCE})")
print()

# Compute strain for both versions
print("STEP 7: Strain Comparison")
print("-" * 80)

from modules.data.processing.strain_analysis.pixelstrain import pixelstrain

H, W = disp_original.shape[-2:]
X, Y = np.meshgrid(np.arange(W), np.arange(H))
mask_128 = myocardium_mask[0].astype(bool)

def compute_strain(disp):
    dXt = np.moveaxis(disp[0, 0].numpy(), 0, -1)
    dYt = np.moveaxis(disp[0, 1].numpy(), 0, -1)
    return pixelstrain(X=X, Y=Y, dXt=dXt, dYt=dYt, mask=mask_128,
                      times=np.arange(disp.shape[2]))

strain_original = compute_strain(disp_original)
strain_pytorch = compute_strain(disp_pytorch)

# Compute global strain
def compute_global_strain(strain_map, mask):
    n_frames = strain_map.shape[-1]
    global_strain = []
    for t in range(n_frames):
        frame_strain = strain_map[..., t]
        valid_strain = frame_strain[mask & ~np.isnan(frame_strain)]
        if len(valid_strain) > 0:
            global_strain.append(valid_strain.mean())
        else:
            global_strain.append(np.nan)
    return np.array(global_strain)

global_cc_original = compute_global_strain(strain_original['CC'], mask_128)
global_cc_pytorch = compute_global_strain(strain_pytorch['CC'], mask_128)
global_rr_original = compute_global_strain(strain_original['RR'], mask_128)
global_rr_pytorch = compute_global_strain(strain_pytorch['RR'], mask_128)

print("\nCircumferential Strain:")
print(f"  Original peak: {np.nanmin(global_cc_original):.6f}")
print(f"  PyTorch peak:  {np.nanmin(global_cc_pytorch):.6f}")
print(f"  Difference:    {abs(np.nanmin(global_cc_original) - np.nanmin(global_cc_pytorch)):.6e}")

print("\nRadial Strain:")
print(f"  Original peak: {np.nanmax(global_rr_original):.6f}")
print(f"  PyTorch peak:  {np.nanmax(global_rr_pytorch):.6f}")
print(f"  Difference:    {abs(np.nanmax(global_rr_original) - np.nanmax(global_rr_pytorch)):.6e}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print()

if max_diff < TOLERANCE and correlation > 0.999:
    print("✓ ✓ ✓  VERIFICATION SUCCESSFUL  ✓ ✓ ✓")
    print()
    print("The pure PyTorch implementation produces nearly identical results")
    print("to the original lagomorph-based implementation.")
    print()
    print("You can now deploy models using the pure PyTorch version without")
    print("needing to compile lagomorph CUDA extensions!")
else:
    print("⚠ WARNING: Differences detected between implementations")
    print()
    print(f"Max difference: {max_diff:.6e} (tolerance: {TOLERANCE})")
    print(f"Correlation: {correlation:.10f}")
    print()
    print("The implementations may need further review.")

print()
print("Files created:")
print("  - /home/jx332/project/code/2025-08-LaMoD/debug_pure_pytorch/lagomorph_pytorch.py")
print("  - /home/jx332/project/code/2025-08-LaMoD/debug_pure_pytorch/regnet_pytorch.py")
print("=" * 80)
