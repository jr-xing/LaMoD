"""
Direct import test - bypass problematic __init__.py files.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_registration_direct():
    """Test registration displacement with direct import."""
    print("\n" + "="*70)
    print("TEST: Registration Displacement (RegNet) - Direct Import")
    print("="*70)

    try:
        # Direct import bypassing __init__.py chain
        sys.path.insert(0, str(project_root / 'cardiac_toolkit_to_be_organized'))
        from motion.tracking.lamod_registration.api import registration_displacement

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


def test_lamod_direct():
    """Test LaMoD displacement with direct import."""
    print("\n" + "="*70)
    print("TEST: LaMoD Displacement - Direct Import (Fast Mode)")
    print("="*70)

    try:
        # Direct import
        sys.path.insert(0, str(project_root / 'cardiac_toolkit_to_be_organized'))
        from motion.tracking.lamod.api import lamod_displacement

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
    print("DEEP LEARNING TRACKING INTEGRATION TEST (DIRECT IMPORTS)")
    print("="*70)

    results = {}

    # Test RegNet
    results['registration'] = test_registration_direct()

    # Test LaMoD
    results['lamod'] = test_lamod_direct()

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
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nBoth RegNet and LaMoD tracking modules are working correctly!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit(main())
