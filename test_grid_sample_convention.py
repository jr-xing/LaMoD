#!/usr/bin/env python
"""
Test to understand PyTorch grid_sample coordinate convention.
"""

import torch
import torch.nn.functional as F
import numpy as np

print("="*70)
print("Testing PyTorch grid_sample Coordinate Convention")
print("="*70)

# Create a simple test image: a white square in specific location
H, W = 10, 10
img = torch.zeros(1, 1, H, W)
img[0, 0, 2:4, 3:5] = 1.0  # White square at rows [2,3], cols [3,4]

print("\n1. Original image (white square at rows [2,3], cols [3,4]):")
print(img[0, 0].numpy())

# Test 1: Identity grid (should return same image)
print("\n2. Test with identity grid:")
y_coords = torch.linspace(-1, 1, H)
x_coords = torch.linspace(-1, 1, W)
grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

print(f"   Grid shape: {grid.shape}")
print(f"   Grid[0, 0, 0, :] (top-left corner): {grid[0, 0, 0, :].numpy()}")  # Should be [-1, -1]
print(f"   Grid[0, -1, -1, :] (bottom-right corner): {grid[0, -1, -1, :].numpy()}")  # Should be [1, 1]

sampled = F.grid_sample(img, grid, align_corners=True)
print("\n   Sampled image (should match original):")
print(sampled[0, 0].numpy())
matches = torch.allclose(img, sampled, atol=1e-6)
print(f"   ✓ Identity test: {'PASS' if matches else 'FAIL'}")

# Test 2: Shift right by 1 pixel
print("\n3. Test shifting RIGHT by 1 pixel (cols 3:5 -> cols 4:6):")
# To shift right, we need to sample from 1 pixel to the left
# In normalized coords, 1 pixel = 2/(W-1) ≈ 0.222
shift_x = -2.0 / (W - 1)
grid_shifted = grid.clone()
grid_shifted[..., 0] += shift_x  # Modify X coordinate (first component)

sampled_right = F.grid_sample(img, grid_shifted, align_corners=True)
print(sampled_right[0, 0].numpy())
# Check if square moved from cols [3,4] to cols [4,5]
moved_right = (sampled_right[0, 0, 2:4, 4:6].sum() > 3.5)
print(f"   Square at new location [2:4, 4:6]: {'✓ PASS' if moved_right else '✗ FAIL'}")

# Test 3: Shift down by 1 pixel
print("\n4. Test shifting DOWN by 1 pixel (rows 2:4 -> rows 3:5):")
shift_y = -2.0 / (H - 1)
grid_shifted = grid.clone()
grid_shifted[..., 1] += shift_y  # Modify Y coordinate (second component)

sampled_down = F.grid_sample(img, grid_shifted, align_corners=True)
print(sampled_down[0, 0].numpy())
# Check if square moved from rows [2,3] to rows [3,4]
moved_down = (sampled_down[0, 0, 3:5, 3:5].sum() > 3.5)
print(f"   Square at new location [3:5, 3:5]: {'✓ PASS' if moved_down else '✗ FAIL'}")

print("\n" + "="*70)
print("Conclusion:")
print("="*70)
print("grid_sample expects grid[..., 0] = X coordinate (column)")
print("grid_sample expects grid[..., 1] = Y coordinate (row)")
print("To shift image RIGHT (positive X), SUBTRACT from grid[..., 0]")
print("To shift image DOWN (positive Y), SUBTRACT from grid[..., 1]")
print("\nFor deformation with displacement field u:")
print("  out(x) = I(x + u(x))  [Lagrangian view]")
print("  So grid should be: identity + u")
print("="*70)
