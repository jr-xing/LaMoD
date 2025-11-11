"""
Debug script to identify where numerical differences come from.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD')
sys.path.insert(0, '/home/jx332/project/code/2025-08-LaMoD/debug_pure_pytorch')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')

print("=" * 80)
print("DEBUGGING NUMERICAL DIFFERENCES")
print("=" * 80)
print()

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Create simple test case
print("Creating simple test case...")
batch_size = 1
H, W = 128, 128
T = 10

# Create simple test momentum field
m = torch.randn(batch_size, 2, H, W).to(device) * 0.1
print(f"Test momentum shape: {m.shape}")
print(f"Test momentum range: [{m.min():.6f}, {m.max():.6f}]")
print()

# ============================================================================
# Test 1: Compare FluidMetric.sharp()
# ============================================================================
print("TEST 1: Comparing FluidMetric.sharp()")
print("-" * 80)

alpha, gamma = 2.0, 1.0
fluid_params = [alpha, 0, gamma]

# Original lagomorph
import lagomorph as lm
metric_original = lm.FluidMetric(fluid_params)
v_original = metric_original.sharp(m)

print(f"Original velocity range: [{v_original.min():.6f}, {v_original.max():.6f}]")

# Pure PyTorch
import lagomorph_pytorch as lm_pt
from lagomorph_pytorch import FluidMetricPyTorch
metric_pytorch = FluidMetricPyTorch(fluid_params)
v_pytorch = metric_pytorch.sharp(m)

print(f"PyTorch velocity range: [{v_pytorch.min():.6f}, {v_pytorch.max():.6f}]")

# Compare
diff_v = torch.abs(v_pytorch - v_original)
print(f"\nVelocity comparison:")
print(f"  Max diff: {diff_v.max():.6e}")
print(f"  Mean diff: {diff_v.mean():.6e}")
print(f"  Correlation: {np.corrcoef(v_original.cpu().flatten(), v_pytorch.cpu().flatten())[0,1]:.10f}")
print()

# ============================================================================
# Test 2: Compare interp()
# ============================================================================
print("TEST 2: Comparing interp()")
print("-" * 80)

# Create test image and displacement
test_img = torch.randn(batch_size, 1, H, W).to(device)
test_disp = v_original * 2.0  # Use velocity as displacement

# Original lagomorph
deformed_original = lm.interp(test_img, test_disp)
print(f"Original deformed range: [{deformed_original.min():.6f}, {deformed_original.max():.6f}]")

# Pure PyTorch (already imported above)
deformed_pytorch = lm_pt.interp_pytorch(test_img, test_disp)
print(f"PyTorch deformed range: [{deformed_pytorch.min():.6f}, {deformed_pytorch.max():.6f}]")

# Compare
diff_deformed = torch.abs(deformed_pytorch - deformed_original)
print(f"\nDeformed image comparison:")
print(f"  Max diff: {diff_deformed.max():.6e}")
print(f"  Mean diff: {diff_deformed.mean():.6e}")
print(f"  Correlation: {np.corrcoef(deformed_original.cpu().flatten(), deformed_pytorch.cpu().flatten())[0,1]:.10f}")
print()

# ============================================================================
# Test 3: Compare EPDiff step by step
# ============================================================================
print("TEST 3: Comparing EPDiff.my_expmap() step by step")
print("-" * 80)

# Original lagomorph
from models.registration.EpdiffLib import Epdiff as EpdiffOriginal
epdiff_original = EpdiffOriginal(alpha=alpha, gamma=gamma)

u_seq_orig, v_seq_orig, m_seq_orig, ui_seq_orig = epdiff_original.my_expmap(
    m, num_steps=10
)

print(f"Original: {len(u_seq_orig)} steps")
print(f"  Final displacement range: [{u_seq_orig[-1].min():.6f}, {u_seq_orig[-1].max():.6f}]")

# Pure PyTorch
import EpdiffLib_pytorch
from EpdiffLib_pytorch import Epdiff as EpdiffPyTorch
epdiff_pytorch = EpdiffPyTorch(alpha=alpha, gamma=gamma)

u_seq_pytorch, v_seq_pytorch, m_seq_pytorch, ui_seq_pytorch = epdiff_pytorch.my_expmap(
    m, num_steps=10
)

print(f"PyTorch: {len(u_seq_pytorch)} steps")
print(f"  Final displacement range: [{u_seq_pytorch[-1].min():.6f}, {u_seq_pytorch[-1].max():.6f}]")

# Compare at each step
print(f"\nStep-by-step comparison:")
for i in range(min(len(u_seq_orig), len(u_seq_pytorch))):
    diff = torch.abs(u_seq_pytorch[i] - u_seq_orig[i])
    print(f"  Step {i}: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")

print()

# ============================================================================
# Test 4: Check if there's an accumulation error
# ============================================================================
print("TEST 4: Checking for accumulation error")
print("-" * 80)

# Test with fewer steps
for num_steps in [1, 2, 5, 10]:
    u_orig, _, _, _ = epdiff_original.my_expmap(m, num_steps=num_steps)
    u_pytorch, _, _, _ = epdiff_pytorch.my_expmap(m, num_steps=num_steps)

    diff = torch.abs(u_pytorch[-1] - u_orig[-1])
    print(f"  {num_steps} steps: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")

print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

if diff_v.max() < 1e-5:
    print("✓ FluidMetric is accurate (< 1e-5)")
else:
    print(f"⚠ FluidMetric has differences: {diff_v.max():.6e}")

if diff_deformed.max() < 1e-5:
    print("✓ interp() is accurate (< 1e-5)")
else:
    print(f"⚠ interp() has differences: {diff_deformed.max():.6e}")

final_diff = torch.abs(u_seq_pytorch[-1] - u_seq_orig[-1])
if final_diff.max() < 1e-3:
    print("✓ EPDiff is accurate (< 1e-3)")
else:
    print(f"⚠ EPDiff has larger differences: {final_diff.max():.6e}")
    print("  This may be due to accumulation of small errors over integration steps")

print()
print("Recommendation:")
if final_diff.max() > 0.1:
    print("  Large differences detected - need to investigate EPDiff implementation")
elif final_diff.max() > 1e-3:
    print("  Small differences likely due to numerical precision in integration")
    print("  Results should still be scientifically valid (correlation > 0.97)")
else:
    print("  Excellent agreement - implementation is accurate!")
