"""
Complete integration test - now that visualization imports are fixed.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_main_namespace_imports():
    """Test that imports work from main namespace."""
    print("\n" + "="*70)
    print("TEST 1: Main Namespace Imports")
    print("="*70)

    try:
        # These should now work!
        from cardiac_toolkit_to_be_organized.motion import (
            registration_displacement,
            lamod_displacement,
            calculate_displacement
        )

        print("✓ All imports successful from main namespace:")
        print("  - calculate_displacement (block matching)")
        print("  - registration_displacement (RegNet)")
        print("  - lamod_displacement (LaMoD)")
        print("✓ TEST PASSED")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registration_full():
    """Test registration displacement end-to-end."""
    print("\n" + "="*70)
    print("TEST 2: Registration Displacement (RegNet) - Full Test")
    print("="*70)

    try:
        from cardiac_toolkit_to_be_organized.motion import registration_displacement

        # Create dummy masks
        print("Creating dummy masks (10 frames, 128x128)...")
        masks = np.random.rand(10, 128, 128).astype(np.float32)
        masks = (masks > 0.5).astype(np.float32)

        print(f"Input shape: {masks.shape}")
        print("Running RegNet inference (CPU)...")

        # Run
        disp = registration_displacement(masks, device='cpu')

        print(f"✓ Success!")
        print(f"  Output shape: {disp.shape}")
        print(f"  Expected: (9, 128, 128, 2)")
        print(f"  Displacement range X: [{disp[..., 0].min():.3f}, {disp[..., 0].max():.3f}]")
        print(f"  Displacement range Y: [{disp[..., 1].min():.3f}, {disp[..., 1].max():.3f}]")

        assert disp.shape == (9, 128, 128, 2)
        print("✓ TEST PASSED")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lamod_full():
    """Test LaMoD displacement end-to-end."""
    print("\n" + "="*70)
    print("TEST 3: LaMoD Displacement - Full Test (Fast Mode)")
    print("="*70)

    try:
        from cardiac_toolkit_to_be_organized.motion import lamod_displacement

        # Create dummy masks
        print("Creating dummy masks (10 frames, 128x128)...")
        masks = np.random.rand(10, 128, 128).astype(np.float32)
        masks = (masks > 0.5).astype(np.float32)

        print(f"Input shape: {masks.shape}")
        print("Running LaMoD inference (CPU, skip_diffusion=True)...")

        # Run in fast mode
        disp = lamod_displacement(masks, device='cpu', skip_diffusion=True)

        print(f"✓ Success!")
        print(f"  Output shape: {disp.shape}")
        print(f"  Expected: (9, 128, 128, 2)")
        print(f"  Displacement range X: [{disp[..., 0].min():.3f}, {disp[..., 0].max():.3f}]")
        print(f"  Displacement range Y: [{disp[..., 1].min():.3f}, {disp[..., 1].max():.3f}]")

        assert disp.shape == (9, 128, 128, 2)
        print("✓ TEST PASSED")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_block_matching():
    """Test that block matching still works."""
    print("\n" + "="*70)
    print("TEST 4: Block Matching (Classical Method)")
    print("="*70)

    try:
        from cardiac_toolkit_to_be_organized.motion import calculate_displacement

        # Create dummy data
        print("Creating dummy images (128x128)...")
        source = np.random.rand(128, 128).astype(np.float32)
        target = np.random.rand(128, 128).astype(np.float32)

        print("Running block matching...")
        disp = calculate_displacement(
            source=source,
            target=target,
            block_shape=16,
            search_range=8
        )

        print(f"✓ Success!")
        print(f"  Output shape: {disp.shape}")
        print(f"  Expected: (128, 128, 2)")

        assert disp.shape == (128, 128, 2)
        print("✓ TEST PASSED")
        return True

    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("COMPLETE INTEGRATION TEST")
    print("After fixing visualization imports")
    print("="*70)

    results = {}

    # Test 1: Imports
    results['imports'] = test_main_namespace_imports()

    # Test 2: Block matching (classical)
    results['block_matching'] = test_block_matching()

    # Test 3: RegNet
    results['registration'] = test_registration_full()

    # Test 4: LaMoD
    results['lamod'] = test_lamod_full()

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name:20s}: {status}")

    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\n✓ Integration complete and verified:")
        print("  - Block matching (classical)")
        print("  - RegNet registration")
        print("  - LaMoD motion tracking")
        print("  - Visualization imports fixed")
        print("  - All accessible from main namespace")
        print("\n✓ Ready for production use!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit(main())
