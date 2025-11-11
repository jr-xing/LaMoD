#!/usr/bin/env python
"""
Test script to verify the coordinate ordering fix in interp_pytorch.

This script compares the lagomorph-free implementation with the original
to ensure displacement fields are being applied correctly.
"""

import sys
import json
import torch
import numpy as np

sys.path.insert(0, '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD')

print("="*70)
print("Testing Coordinate Ordering Fix")
print("="*70)

# Load config
config = json.load(open('configs/ours-pretrain-registration-aligned-frames.json'))
model_config = config['networks']['registration']

# Create dummy test data
print("\n1. Creating test data...")
H, W = 128, 128
src = torch.zeros(1, 1, 1, H, W)
tar = torch.zeros(1, 1, 1, H, W)

# Create a simple pattern in source (a square in the center)
src[0, 0, 0, 50:80, 50:80] = 1.0

# Create target - same square but shifted right by 10 pixels
tar[0, 0, 0, 50:80, 60:90] = 1.0

print(f"   Source: square at [50:80, 50:80]")
print(f"   Target: square at [50:80, 60:90]")
print(f"   Expected displacement: ~10 pixels rightward (positive X direction)")

# Test with lagomorph-free implementation
print("\n2. Testing lagomorph-free implementation (AFTER FIX)...")
from cardiac_toolkit_to_be_organized.motion.tracking.lamod import build_model

model_lf = build_model(model_config)
model_lf.eval()

with torch.no_grad():
    result_lf = model_lf(src, tar)

disp_lf = result_lf['displacement'].cpu().numpy()
deformed_lf = result_lf['deformed_source'].cpu().numpy()

# Compute Dice score
def compute_dice(pred, target):
    pred_bin = (pred > 0.5).astype(float)
    target_bin = target.astype(float)
    intersection = np.sum(pred_bin * target_bin)
    dice = (2.0 * intersection + 1e-5) / (np.sum(pred_bin) + np.sum(target_bin) + 1e-5)
    return dice

dice_lf = compute_dice(deformed_lf[0, 0, 0], tar[0, 0, 0].numpy())

print(f"   Displacement shape: {disp_lf.shape}")
print(f"   Mean displacement X (u[:,1]): {disp_lf[0, 1, 0].mean():.3f} pixels")
print(f"   Mean displacement Y (u[:,0]): {disp_lf[0, 0, 0].mean():.3f} pixels")
print(f"   Dice score: {dice_lf:.4f}")

# Analyze displacement in the square region
disp_x_square = disp_lf[0, 1, 0, 50:80, 50:80].mean()
disp_y_square = disp_lf[0, 0, 0, 50:80, 50:80].mean()
print(f"   Displacement in square region:")
print(f"      X: {disp_x_square:.3f} pixels (should be ~positive)")
print(f"      Y: {disp_y_square:.3f} pixels (should be ~0)")

# Visual check
print("\n3. Visual check of deformed source...")
deformed_center_y = np.where(deformed_lf[0, 0, 0, 64, :] > 0.5)[0]
if len(deformed_center_y) > 0:
    deformed_x_range = [deformed_center_y[0], deformed_center_y[-1]]
    print(f"   Deformed square X range (row 64): {deformed_x_range}")
    print(f"   Target square X range (row 64): [60, 89]")
    x_error = abs(deformed_x_range[0] - 60) + abs(deformed_x_range[1] - 89)
    print(f"   Position error: {x_error} pixels")
else:
    print("   ERROR: No deformed square found!")

print("\n" + "="*70)
print("Results Summary")
print("="*70)

if dice_lf > 0.85:
    print(f"✓ PASS: Dice score = {dice_lf:.4f} (>0.85)")
else:
    print(f"✗ FAIL: Dice score = {dice_lf:.4f} (<0.85)")

if disp_x_square > 5 and abs(disp_y_square) < 2:
    print(f"✓ PASS: Displacement direction correct (X={disp_x_square:.1f}, Y={disp_y_square:.1f})")
else:
    print(f"✗ FAIL: Displacement direction incorrect (X={disp_x_square:.1f}, Y={disp_y_square:.1f})")
    print(f"   Expected: X > 5, Y ≈ 0")

print("\n" + "="*70)

# Test with original implementation for comparison
print("\n4. Testing original implementation (with lagomorph)...")
try:
    from models import build_model as build_model_orig

    model_orig = build_model_orig(model_config)
    model_orig.eval()

    with torch.no_grad():
        result_orig = model_orig(src, tar)

    disp_orig = result_orig['displacement'].cpu().numpy()
    deformed_orig = result_orig['deformed_source'].cpu().numpy()
    dice_orig = compute_dice(deformed_orig[0, 0, 0], tar[0, 0, 0].numpy())

    print(f"   Original Dice score: {dice_orig:.4f}")
    print(f"   Original mean disp X: {disp_orig[0, 1, 0].mean():.3f}")
    print(f"   Original mean disp Y: {disp_orig[0, 0, 0].mean():.3f}")

    print("\n5. Comparison:")
    print(f"   Dice difference: {abs(dice_lf - dice_orig):.4f}")
    print(f"   Displacement X difference: {abs(disp_lf[0, 1, 0].mean() - disp_orig[0, 1, 0].mean()):.3f}")

    if abs(dice_lf - dice_orig) < 0.05:
        print("   ✓ Implementations match!")
    else:
        print("   ✗ Implementations differ significantly")

except ImportError:
    print("   (Skipping comparison - original implementation requires lagomorph)")

print("\n" + "="*70)
