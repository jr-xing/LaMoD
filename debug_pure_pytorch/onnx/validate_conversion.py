#!/usr/bin/env python3
"""
Validate ONNX conversion by comparing PyTorch vs ONNX outputs

This script loads UK Biobank cardiac cine data, runs inference with both
pure PyTorch and ONNX versions of RegNet, and compares the outputs to ensure
correctness of the ONNX conversion.

Author: Claude
Date: 2025-11-03
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize

# Add parent directories to path
sys.path.insert(0, '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD')
sys.path.insert(0, '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD/debug_pure_pytorch')

from regnet_pytorch import RegNet as RegNetPyTorch
from cardiac_toolkit.detection import WholeHeartDetectorYOLOv9
from cardiac_toolkit.segmentation.models.carson import CarSONSegmentor


def load_ukb_data(data_path, series_idx=11):
    """Load UK Biobank cardiac cine data."""
    print(f"Loading UK Biobank data from: {data_path}")
    ukb_data = np.load(data_path, allow_pickle=True).item()
    print(f"Subject ID: {ukb_data['subject_id']}")

    # Extract selected series
    series = ukb_data['series_list'][series_idx]
    temporal_frames = series['slices']
    n_frames = len(temporal_frames)
    print(f"Series {series_idx}: {n_frames} temporal frames")

    # Stack frames
    images = np.array([frame['image'] for frame in temporal_frames])

    # Normalize
    images_normalized = images.astype(np.float32)
    images_normalized = (images_normalized - images_normalized.min()) / \
                        (images_normalized.max() - images_normalized.min() + 1e-8)

    return images_normalized


def preprocess_images(images, detector_model_path):
    """Preprocess images: detect heart, crop, resize to 128x128."""
    print("\nPreprocessing images...")

    # Initialize detector
    detector = WholeHeartDetectorYOLOv9(model_path=detector_model_path)

    # Detect on first frame
    first_frame = images[0]
    detection_result = detector.detect([first_frame], conf_thres=0.7, one_per_class=True)[0]

    if not detection_result['whole_heart']:
        raise RuntimeError("No whole heart detected!")

    bbox = detection_result['whole_heart'][0]

    # Expand by 1.5x
    from cardiac_toolkit.detection.utils.bbox_utils import rescale_bbox
    bbox_expanded = rescale_bbox(bbox, scale_factor=1.5)

    # Make square
    width = bbox_expanded['x2'] - bbox_expanded['x1']
    height = bbox_expanded['y2'] - bbox_expanded['y1']

    bbox_square = bbox_expanded.copy()
    if width > height:
        diff = width - height
        bbox_square['y1'] -= diff / 2
        bbox_square['y2'] += diff / 2
    elif height > width:
        diff = height - width
        bbox_square['x1'] -= diff / 2
        bbox_square['x2'] += diff / 2

    # Clip to image bounds
    img_height, img_width = images.shape[1:3]
    bbox_square['x1'] = max(0, bbox_square['x1'])
    bbox_square['y1'] = max(0, bbox_square['y1'])
    bbox_square['x2'] = min(img_width, bbox_square['x2'])
    bbox_square['y2'] = min(img_height, bbox_square['y2'])

    # Crop all frames
    x1, y1 = int(bbox_square['x1']), int(bbox_square['y1'])
    x2, y2 = int(bbox_square['x2']), int(bbox_square['y2'])
    cropped_frames = [frame[y1:y2, x1:x2] for frame in images]

    # Resize to 128x128
    resized_frames = [resize(frame, (128, 128), anti_aliasing=True, preserve_range=True)
                      for frame in cropped_frames]

    images_preprocessed = np.array(resized_frames, dtype=np.float32)
    print(f"Preprocessed shape: {images_preprocessed.shape}")

    return images_preprocessed


def segment_myocardium(images, segmentor_model_path):
    """Segment myocardium using CarSON."""
    print("\nSegmenting myocardium with CarSON...")

    segmentor = CarSONSegmentor(
        model_path=segmentor_model_path,
        input_format='DHW',
        batch_size=32,
        normalize_per_slice=True
    )

    seg_mask = segmentor.segment(images)
    myocardium_mask = (seg_mask == 2).astype(np.uint8)

    print(f"Myocardium mask shape: {myocardium_mask.shape}")
    print(f"Mean pixels per frame: {myocardium_mask.sum(axis=(1, 2)).mean():.0f}")

    return myocardium_mask


def run_pytorch_inference(myocardium_mask, config_path, checkpoint_path, device):
    """Run inference with pure PyTorch RegNet model."""
    print("\n" + "="*70)
    print("Running PyTorch inference...")
    print("="*70)

    # Load model
    with open(config_path, 'r') as f:
        config = json.load(f)

    regnet = RegNetPyTorch(config['networks']['registration'])
    regnet = regnet.to(device)
    regnet.eval()

    # Prepare input
    regnet_input = torch.from_numpy(myocardium_mask[None, None]).to(device).float()
    print(f"Input shape: {regnet_input.shape}")

    # Run inference
    with torch.no_grad():
        output = regnet.inference_2D_video(regnet_input)

    displacement = output['displacement'].detach().cpu()
    print(f"Output displacement shape: {displacement.shape}")

    return displacement


def run_onnx_inference(myocardium_mask, onnx_path):
    """Run inference with ONNX RegNet model."""
    print("\n" + "="*70)
    print("Running ONNX inference...")
    print("="*70)

    import onnxruntime as ort

    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available_providers = ort.get_available_providers()
    session_providers = [p for p in providers if p in available_providers]

    print(f"Available providers: {available_providers}")
    print(f"Using providers: {session_providers}")

    session = ort.InferenceSession(onnx_path, providers=session_providers)

    # Prepare input
    onnx_input = myocardium_mask[None, None].astype(np.float32)
    print(f"Input shape: {onnx_input.shape}")

    # Run inference
    outputs = session.run(None, {'video': onnx_input})

    # outputs[0] is displacement
    displacement = outputs[0]
    print(f"Output displacement shape: {displacement.shape}")

    return displacement


def compare_outputs(pytorch_disp, onnx_disp):
    """Compare PyTorch and ONNX outputs."""
    print("\n" + "="*70)
    print("Comparing PyTorch vs ONNX outputs...")
    print("="*70)

    # Convert to numpy if needed
    if torch.is_tensor(pytorch_disp):
        pytorch_disp = pytorch_disp.numpy()

    # Check shapes
    print(f"\nShape comparison:")
    print(f"  PyTorch: {pytorch_disp.shape}")
    print(f"  ONNX:    {onnx_disp.shape}")

    if pytorch_disp.shape != onnx_disp.shape:
        print(f"⚠ WARNING: Shapes do not match!")
        return None

    # Compute metrics
    diff = pytorch_disp - onnx_disp
    mae = np.abs(diff).mean()
    max_diff = np.abs(diff).max()
    rmse = np.sqrt((diff ** 2).mean())

    # Correlation
    corr_x = np.corrcoef(pytorch_disp[0, 0].flatten(), onnx_disp[0, 0].flatten())[0, 1]
    corr_y = np.corrcoef(pytorch_disp[0, 1].flatten(), onnx_disp[0, 1].flatten())[0, 1]

    # Relative error
    pytorch_norm = np.linalg.norm(pytorch_disp)
    relative_error = np.linalg.norm(diff) / (pytorch_norm + 1e-8)

    results = {
        'mae': mae,
        'max_diff': max_diff,
        'rmse': rmse,
        'corr_x': corr_x,
        'corr_y': corr_y,
        'relative_error': relative_error,
    }

    # Print results
    print(f"\nNumerical Comparison:")
    print(f"  Mean Absolute Error (MAE):  {mae:.6f} pixels")
    print(f"  Max Absolute Difference:     {max_diff:.6f} pixels")
    print(f"  Root Mean Square Error:      {rmse:.6f} pixels")
    print(f"  Correlation (X component):   {corr_x:.6f}")
    print(f"  Correlation (Y component):   {corr_y:.6f}")
    print(f"  Relative Error:              {relative_error:.6f} ({relative_error*100:.3f}%)")

    # Pass/fail criteria
    print(f"\nValidation Results:")
    passed = True

    if mae < 0.01:
        print(f"  ✓ MAE < 0.01 pixels: PASS")
    else:
        print(f"  ✗ MAE >= 0.01 pixels: FAIL (MAE = {mae:.6f})")
        passed = False

    if max_diff < 0.1:
        print(f"  ✓ Max diff < 0.1 pixels: PASS")
    else:
        print(f"  ⚠ Max diff >= 0.1 pixels: WARNING (max = {max_diff:.6f})")

    if corr_x > 0.999 and corr_y > 0.999:
        print(f"  ✓ Correlation > 0.999: PASS")
    else:
        print(f"  ⚠ Correlation <= 0.999: WARNING (X={corr_x:.6f}, Y={corr_y:.6f})")

    if relative_error < 0.001:
        print(f"  ✓ Relative error < 0.1%: PASS")
    else:
        print(f"  ⚠ Relative error >= 0.1%: WARNING ({relative_error*100:.3f}%)")

    if passed:
        print(f"\n✓ Overall: PASS - ONNX conversion is correct!")
    else:
        print(f"\n⚠ Overall: Needs review - Check validation plots")

    results['passed'] = passed
    return results


def visualize_comparison(pytorch_disp, onnx_disp, output_dir):
    """Visualize PyTorch vs ONNX displacement comparison."""
    print(f"\nGenerating comparison visualizations...")

    # Convert to numpy
    if torch.is_tensor(pytorch_disp):
        pytorch_disp = pytorch_disp.numpy()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Select middle frame
    n_frames = pytorch_disp.shape[2]
    mid_frame = n_frames // 2

    # Plot X and Y displacement comparison
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))

    # X displacement - PyTorch
    im1 = axes[0, 0].imshow(pytorch_disp[0, 0, mid_frame], cmap='RdBu_r', vmin=-5, vmax=5)
    axes[0, 0].set_title(f'PyTorch - X Displacement (Frame {mid_frame})')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # X displacement - ONNX
    im2 = axes[0, 1].imshow(onnx_disp[0, 0, mid_frame], cmap='RdBu_r', vmin=-5, vmax=5)
    axes[0, 1].set_title(f'ONNX - X Displacement (Frame {mid_frame})')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # Y displacement - PyTorch
    im3 = axes[1, 0].imshow(pytorch_disp[0, 1, mid_frame], cmap='RdBu_r', vmin=-5, vmax=5)
    axes[1, 0].set_title(f'PyTorch - Y Displacement (Frame {mid_frame})')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # Y displacement - ONNX
    im4 = axes[1, 1].imshow(onnx_disp[0, 1, mid_frame], cmap='RdBu_r', vmin=-5, vmax=5)
    axes[1, 1].set_title(f'ONNX - Y Displacement (Frame {mid_frame})')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    # Difference - X
    diff_x = pytorch_disp[0, 0, mid_frame] - onnx_disp[0, 0, mid_frame]
    max_diff = max(np.abs(diff_x).max(), 1e-6)
    im5 = axes[2, 0].imshow(diff_x, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    axes[2, 0].set_title(f'Difference - X (max: {np.abs(diff_x).max():.6f})')
    axes[2, 0].axis('off')
    plt.colorbar(im5, ax=axes[2, 0], fraction=0.046)

    # Difference - Y
    diff_y = pytorch_disp[0, 1, mid_frame] - onnx_disp[0, 1, mid_frame]
    max_diff = max(np.abs(diff_y).max(), 1e-6)
    im6 = axes[2, 1].imshow(diff_y, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    axes[2, 1].set_title(f'Difference - Y (max: {np.abs(diff_y).max():.6f})')
    axes[2, 1].axis('off')
    plt.colorbar(im6, ax=axes[2, 1], fraction=0.046)

    plt.tight_layout()
    output_path = Path(output_dir) / 'displacement_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    # Plot temporal profiles
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean displacement over time
    pytorch_mean_x = pytorch_disp[0, 0].mean(axis=(1, 2))
    pytorch_mean_y = pytorch_disp[0, 1].mean(axis=(1, 2))
    onnx_mean_x = onnx_disp[0, 0].mean(axis=(1, 2))
    onnx_mean_y = onnx_disp[0, 1].mean(axis=(1, 2))

    axes[0, 0].plot(pytorch_mean_x, 'b-', linewidth=2, label='PyTorch')
    axes[0, 0].plot(onnx_mean_x, 'r--', linewidth=2, label='ONNX')
    axes[0, 0].set_title('Mean X Displacement Over Time')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Displacement (pixels)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(pytorch_mean_y, 'b-', linewidth=2, label='PyTorch')
    axes[0, 1].plot(onnx_mean_y, 'r--', linewidth=2, label='ONNX')
    axes[0, 1].set_title('Mean Y Displacement Over Time')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Displacement (pixels)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Difference over time
    diff_x_time = pytorch_mean_x - onnx_mean_x
    diff_y_time = pytorch_mean_y - onnx_mean_y

    axes[1, 0].plot(diff_x_time, 'k-', linewidth=2)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Difference in Mean X Displacement')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Difference (pixels)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(diff_y_time, 'k-', linewidth=2)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Difference in Mean Y Displacement')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Difference (pixels)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'temporal_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Main validation pipeline."""
    print("Pure PyTorch RegNet ONNX Conversion Validation")
    print("="*70)

    # Paths
    base_dir = Path('/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD')
    ukb_data_path = Path('/vast/palmer/pi/kwan/jx332/data/2025-10-12-UKB-sample/unified_analysis_results/segmented/1002514.npy')
    detector_model_path = base_dir / 'cardiac_toolkit/detection/models/whole_heart_yolov9/checkpoints/best.onnx'
    segmentor_model_path = Path('/gpfs/gibbs/project/kwan/jx332/code/2025-05-DeepStrain/pretrained_models/carson_Jan2021.onnx')
    config_path = base_dir / 'configs/ours-train-registration.json'
    checkpoint_path = base_dir / 'checkpoints/ours_registration.pth'
    onnx_path = base_dir / 'debug_pure_pytorch/onnx/regnet_pure_pytorch.onnx'
    output_dir = base_dir / 'debug_pure_pytorch/onnx/validation_results'

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    images = load_ukb_data(ukb_data_path, series_idx=11)

    # Preprocess
    images_preprocessed = preprocess_images(images, detector_model_path)

    # Segment
    myocardium_mask = segment_myocardium(images_preprocessed, segmentor_model_path)

    # Run PyTorch inference
    pytorch_disp = run_pytorch_inference(myocardium_mask, config_path, checkpoint_path, device)

    # Run ONNX inference
    try:
        onnx_disp = run_onnx_inference(myocardium_mask, onnx_path)
    except Exception as e:
        print(f"\n✗ ONNX inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare outputs
    results = compare_outputs(pytorch_disp, onnx_disp)

    if results is not None:
        # Visualize
        visualize_comparison(pytorch_disp, onnx_disp, output_dir)

        print(f"\n{'='*70}")
        print(f"Validation complete!")
        print(f"{'='*70}")
        print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
