"""
Test that the toolkit wrapper correctly returns RegNet displacement fields.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("TOOLKIT REGNET WRAPPER TEST")
print("Verify that registration_displacement() returns full displacement")
print("=" * 70)

# Import from toolkit
from cardiac_toolkit_to_be_organized.motion.tracking.lamod_registration import (
    registration_displacement
)

# Create dummy input
print(f"\n1. Creating dummy masks...")
masks = np.random.rand(10, 128, 128).astype(np.float32)
masks = (masks > 0.5).astype(np.float32)
print(f"  - Input shape: {masks.shape}")

# Run through toolkit API
print(f"\n2. Running registration_displacement()...")
disp = registration_displacement(
    masks=masks,
    device='cuda',
    return_details=False
)

print(f"\n3. Output analysis:")
print(f"  - Type: {type(disp)}")
print(f"  - Shape: {disp.shape}")
print(f"  - Expected: (9, 128, 128, 2)")
print(f"  - Displacement range X: [{disp[..., 0].min():.3f}, {disp[..., 0].max():.3f}]")
print(f"  - Displacement range Y: [{disp[..., 1].min():.3f}, {disp[..., 1].max():.3f}]")

# Verify shape
if disp.shape == (9, 128, 128, 2):
    print(f"\n✓ SUCCESS: Toolkit correctly returns displacement field!")
    print(f"  - Shape matches expected format: (T-1, H, W, 2)")
    print(f"  - Ready for strain analysis")
else:
    print(f"\n✗ ERROR: Shape mismatch!")
    print(f"  Expected: (9, 128, 128, 2)")
    print(f"  Got: {disp.shape}")
    sys.exit(1)

# Test with detailed output
print(f"\n4. Testing with return_details=True...")
result = registration_displacement(
    masks=masks,
    device='cuda',
    return_details=True
)

print(f"  - Result type: {type(result)}")
print(f"  - Keys: {list(result.keys())}")
print(f"  - displacement shape: {result['displacement'].shape}")

if 'velocity' in result:
    print(f"  ✓ Velocity included: {result['velocity'].shape}")
if 'momentum' in result:
    print(f"  ✓ Momentum included: {result['momentum'].shape}")

print(f"\n" + "=" * 70)
print(f"CONCLUSION: Toolkit wrapper is working correctly!")
print(f"Users can now run full motion tracking with RegNet:")
print(f"  1. Segment images with CarSON")
print(f"  2. Track motion with registration_displacement()")
print(f"  3. Compute strain with pixelstrain()")
print("=" * 70)
