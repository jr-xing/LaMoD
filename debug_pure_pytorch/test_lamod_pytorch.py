"""
Test LaMoD with pure PyTorch implementation.

Tests in two steps:
1. Run with pure PyTorch in LGE environment (no lagomorph)
2. Compare with original in LaMoD environment (skip_diffusion=True)
"""

import os
import sys
import numpy as np
import torch
import json
from pathlib import Path

sys.path.insert(0, '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD')

# Determine which environment we're in
try:
    import lagomorph as lm
    lm.FluidMetric([1.0, 0.0, 1.0])  # Test if it actually works
    LAGOMORPH_AVAILABLE = True
    ENV_NAME = "LaMoD (lagomorph available)"
except:
    LAGOMORPH_AVAILABLE = False
    ENV_NAME = "LGE (lagomorph unavailable)"

print("=" * 80)
print(f"TESTING LaMoD WITH PURE PYTORCH")
print("=" * 80)
print(f"Environment: {ENV_NAME}")
print()

# Setup device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# STEP 1: Load and preprocess data (same as notebook)
# ============================================================================
print("STEP 1: Loading and preprocessing data")
print("-" * 80)

from skimage.transform import resize
from cardiac_toolkit.detection import WholeHeartDetectorYOLOv9

# Load UKB data
ukb_data_path = Path('/vast/palmer/pi/kwan/jx332/data/2025-10-12-UKB-sample/unified_analysis_results/segmented/1002514.npy')
ukb_data = np.load(ukb_data_path, allow_pickle=True).item()

# Find middle SAS series
sas_series = []
for i, series in enumerate(ukb_data['series_list']):
    view = series.get('info', {}).get('view', '').lower()
    if 'sas' in view:
        sas_series.append((i, series))

middle_sas_idx = len(sas_series) // 2
selected_series_idx, selected_series = sas_series[middle_sas_idx]

# Extract temporal frames (use first 10 for speed)
TEST_FRAMES = 10
temporal_frames = selected_series['slices'][:TEST_FRAMES]
images = np.array([frame['image'] for frame in temporal_frames])

# Normalize
images_normalized = images.astype(np.float32)
images_normalized = (images_normalized - images_normalized.min()) / \
                    (images_normalized.max() - images_normalized.min() + 1e-8)

# Detect and crop
detector_model_path = '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD/cardiac_toolkit/detection/models/whole_heart_yolov9/checkpoints/best.onnx'
detector = WholeHeartDetectorYOLOv9(model_path=detector_model_path)
detection_result = detector.detect([images_normalized[0]], conf_thres=0.7, one_per_class=True)[0]
bbox_original = detection_result['whole_heart'][0]

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

# Crop and resize
cropped_frames = []
for frame in images_normalized:
    x1, y1 = int(bbox_square['x1']), int(bbox_square['y1'])
    x2, y2 = int(bbox_square['x2']), int(bbox_square['y2'])
    cropped = frame[y1:y2, x1:x2]
    cropped_frames.append(cropped)

images_preprocessed = np.array([resize(c, (128, 128), anti_aliasing=True, preserve_range=True)
                                for c in cropped_frames], dtype=np.float32)

print(f"Preprocessed {len(images_preprocessed)} frames to 128x128")
print()

# ============================================================================
# STEP 2: Run CarSON segmentation
# ============================================================================
print("STEP 2: Running CarSON segmentation")
print("-" * 80)

from cardiac_toolkit.segmentation.models.carson import CarSONSegmentor

carson_model_path = '/gpfs/gibbs/project/kwan/jx332/code/2025-05-DeepStrain/pretrained_models/carson_Jan2021.onnx'
segmentor = CarSONSegmentor(model_path=carson_model_path, input_format='DHW',
                            batch_size=32, normalize_per_slice=True)

seg_mask = segmentor.segment(images_preprocessed)
myocardium_mask = (seg_mask == 2).astype(np.uint8)

print(f"Myocardium mask shape: {myocardium_mask.shape}")
print()

# ============================================================================
# STEP 3: Load LaMoD model (with appropriate version based on environment)
# ============================================================================
print("STEP 3: Loading LaMoD model")
print("-" * 80)

if not LAGOMORPH_AVAILABLE:
    # Use pure PyTorch version
    print("Using pure PyTorch implementation (no lagomorph)")
    sys.path.insert(0, '/home/jx332/project/code/2025-08-LaMoD/debug_pure_pytorch')

# Load config
lamod_config_fname = '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD/configs/ours-train-LaMoD-patchify-video.json'
lamod_config = json.load(open(lamod_config_fname))

# Build model
from models.LaMoD import LaMoD
lamod_network = LaMoD(lamod_config['networks'], device).to(device)

print("LaMoD network loaded successfully!")
print()

# ============================================================================
# STEP 4: Run inference with skip_diffusion=True
# ============================================================================
print("STEP 4: Running LaMoD inference (skip_diffusion=True)")
print("-" * 80)

lamod_input = torch.from_numpy(myocardium_mask[None, None]).to(device).float()
print(f"Input shape: {lamod_input.shape}")

with torch.no_grad():
    lamod_output = lamod_network.inference(
        lamod_input,
        ori_n_frames=[lamod_input.shape[2]],
        skip_diffusion=True  # IMPORTANT: Skip diffusion to avoid randomness
    )[0]

print(f"Output keys: {list(lamod_output.keys())}")

# Extract displacement - check which one is available
# When skip_diffusion=True, reg_disp might be None, use LaMoD_disp instead
if lamod_output.get('reg_disp') is not None:
    disp_field = lamod_output['reg_disp'].detach().cpu()
    print(f"\nUsing 'reg_disp' for displacement")
elif lamod_output.get('LaMoD_disp') is not None:
    disp_field = lamod_output['LaMoD_disp'].detach().cpu()
    print(f"\nUsing 'LaMoD_disp' for displacement")
else:
    raise ValueError(f"No displacement field found in output: {lamod_output.keys()}")

print(f"Displacement field shape: {disp_field.shape}")
print(f"Displacement range: [{disp_field.min():.6f}, {disp_field.max():.6f}]")
print()

# ============================================================================
# STEP 5: Compute strain
# ============================================================================
print("STEP 5: Computing strain")
print("-" * 80)

# Upsample to 128x128
disp_field_128 = torch.nn.functional.interpolate(
    disp_field, size=(disp_field.shape[2], 128, 128),
    mode='trilinear', align_corners=False
)

from modules.data.processing.strain_analysis.pixelstrain import pixelstrain

H, W = 128, 128
X, Y = np.meshgrid(np.arange(W), np.arange(H))

dXt = np.moveaxis(disp_field_128[0, 0].numpy(), 0, -1)
dYt = np.moveaxis(disp_field_128[0, 1].numpy(), 0, -1)
mask_128 = myocardium_mask[0].astype(bool)

strain = pixelstrain(X=X, Y=Y, dXt=dXt, dYt=dYt, mask=mask_128,
                     times=np.arange(disp_field_128.shape[2]))

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

global_cc = compute_global_strain(strain['CC'], mask_128)
global_rr = compute_global_strain(strain['RR'], mask_128)

print(f"✓ Strain computation successful!")
print(f"  Circumferential strain peak: {np.nanmin(global_cc):.6f}")
print(f"  Radial strain peak: {np.nanmax(global_rr):.6f}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print()
print(f"Environment: {ENV_NAME}")
print(f"✓ Data loading and preprocessing: SUCCESS")
print(f"✓ CarSON segmentation: SUCCESS")
print(f"✓ LaMoD model loading: SUCCESS")
print(f"✓ LaMoD inference (skip_diffusion=True): SUCCESS")
print(f"✓ Strain computation: SUCCESS")
print()
print(f"Results:")
print(f"  Displacement shape: {disp_field.shape}")
print(f"  Displacement range: [{disp_field.min():.6f}, {disp_field.max():.6f}]")
print(f"  Circumferential strain: {np.nanmin(global_cc):.6f}")
print(f"  Radial strain: {np.nanmax(global_rr):.6f}")
print()

if LAGOMORPH_AVAILABLE:
    print("✓ This is the REFERENCE run (with lagomorph)")
    print("  Save these values for comparison")
else:
    print("✓ This is the PURE PYTORCH run (no lagomorph)")
    print("  Compare these values with the reference run")

print("=" * 80)
