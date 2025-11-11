"""
Simple test script for deep learning tracking modules only.

This avoids importing visualization which has dependencies issues.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_registration():
    """Test registration displacement."""
    print("\n" + "="*70)
    print("TEST: Registration Displacement (RegNet)")
    print("="*70)

    try:
        from cardiac_toolkit_to_be_organized.motion.tracking.lamod_registration import (
            registration_displacement
        )

        # Create dummy masks
        print("Creating dummy masks (10 frames, 128x128)...")
        masks = np.random.rand(10, 128, 128).astype(np.float32)
        masks = (masks > 0.5).astype(np.float32)

        print(f"Input shape: {masks.shape}")
        print("Running RegNet inference (CPU)...")

        # Run
        disp = registration_displacement(
            masks=masks,
            device='cpu'
        )

        print(f"✓ Success!")
        print(f"  Output shape: {disp.shape}")
        print(f"  Expected: (9, 128, 128, 2)")
        assert disp.shape == (9, 128, 128, 2)
        print("✓ TEST PASSED")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lamod():
    """Test LaMoD displacement."""
    print("\n" + "="*70)
    print("TEST: LaMoD Displacement (Fast Mode)")
    print("="*70)

    try:
        from cardiac_toolkit_to_be_organized.motion.tracking.lamod import (
            lamod_displacement
        )

        # Create dummy masks
        print("Creating dummy masks (10 frames, 128x128)...")
        masks = np.random.rand(10, 128, 128).astype(np.float32)
        masks = (masks > 0.5).astype(np.float32)

        print(f"Input shape: {masks.shape}")
        print("Running LaMoD inference (CPU, skip_diffusion=True)...")

        # Run in fast mode
        disp = lamod_displacement(
            masks=masks,
            device='cpu',
            skip_diffusion=True
        )

        print(f"✓ Success!")
        print(f"  Output shape: {disp.shape}")
        print(f"  Expected: (9, 128, 128, 2)")
        assert disp.shape == (9, 128, 128, 2)
        print("✓ TEST PASSED")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("DEEP LEARNING TRACKING INTEGRATION TEST")
    print("="*70)

    results = {}

    # Test RegNet
    results['registration'] = test_registration()

    # Test LaMoD
    results['lamod'] = test_lamod()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name:20s}: {status}")

    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit(main())
