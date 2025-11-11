#!/usr/bin/env python
"""
Direct comparison between lagomorph.interp and interp_pytorch.
"""

import sys
import torch
import numpy as np

sys.path.insert(0, '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD')

print("="*70)
print("Comparing lagomorph.interp vs interp_pytorch")
print("="*70)

# Create simple test image and displacement
H, W = 10, 10
img = torch.zeros(1, 1, H, W, dtype=torch.float32)
img[0, 0, 3:6, 4:7] = 1.0  # Square at rows [3,4,5], cols [4,5,6]

print("\n1. Original image (square at rows [3:6], cols [4:7]):")
print(img[0, 0].numpy())

# Create a constant displacement field: shift right by 2 pixels
u = torch.zeros(1, 2, H, W, dtype=torch.float32)
u[0, 0, :, :] = 0.0  # Y displacement
u[0, 1, :, :] = 2.0  # X displacement (shift right by 2)

print("\n2. Displacement field:")
print(f"   u[:,0] (Y displacement): {u[0, 0, 5, 5].item()}")
print(f"   u[:,1] (X displacement): {u[0, 1, 5, 5].item()}")
print(f"   Expected: square moves from cols [4:7] to cols [6:9]")

# Test with interp_pytorch
print("\n3. Testing interp_pytorch...")
from cardiac_toolkit_to_be_organized.motion.tracking.lamod.registration.lagomorph_pytorch import interp_pytorch

result_pytorch = interp_pytorch(img, u, dt=1.0)
print("   Result:")
print(result_pytorch[0, 0].numpy())

# Check if square is at correct location
square_sum_orig = result_pytorch[0, 0, 3:6, 4:7].sum().item()
square_sum_new = result_pytorch[0, 0, 3:6, 6:9].sum().item()
print(f"   Square sum at original location [3:6, 4:7]: {square_sum_orig:.2f}")
print(f"   Square sum at expected new location [3:6, 6:9]: {square_sum_new:.2f}")

if square_sum_new > 7.0:
    print("   ✓ PASS: Square moved to correct location")
else:
    print("   ✗ FAIL: Square not at expected location")

# Test with lagomorph
print("\n4. Testing lagomorph.interp (requires CUDA)...")
try:
    import lagomorph as lm

    img_cuda = img.cuda()
    u_cuda = u.cuda()

    result_lm = lm.interp(img_cuda, u_cuda, dt=1.0)
    result_lm_cpu = result_lm.cpu()

    print("   Result:")
    print(result_lm_cpu[0, 0].numpy())

    square_sum_orig_lm = result_lm_cpu[0, 0, 3:6, 4:7].sum().item()
    square_sum_new_lm = result_lm_cpu[0, 0, 3:6, 6:9].sum().item()
    print(f"   Square sum at original location [3:6, 4:7]: {square_sum_orig_lm:.2f}")
    print(f"   Square sum at new location [3:6, 6:9]: {square_sum_new_lm:.2f}")

    if square_sum_new_lm > 7.0:
        print("   ✓ PASS: Square moved to correct location")
    else:
        print("   ✗ FAIL: Square not at expected location")

    # Compare results
    print("\n5. Comparison:")
    diff = torch.abs(result_pytorch - result_lm_cpu).max().item()
    print(f"   Max absolute difference: {diff:.6f}")

    if diff < 0.01:
        print("   ✓ PASS: Implementations match!")
    else:
        print("   ✗ FAIL: Implementations differ")
        print("\n   Difference map:")
        print((result_pytorch - result_lm_cpu)[0, 0].numpy())

except Exception as e:
    print(f"   Skipped (error: {e})")

print("\n" + "="*70)
