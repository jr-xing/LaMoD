"""
Test that the toolkit is now self-contained with all models in lamod/.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("STANDALONE TOOLKIT IMPORT TEST")
print("Verifying all models are self-contained in lamod/")
print("=" * 70)

# Test 1: Import main APIs
print("\n1. Testing main API imports...")
try:
    from cardiac_toolkit_to_be_organized.motion import (
        registration_displacement,
        lamod_displacement
    )
    print("✓ Main APIs imported successfully")
except Exception as e:
    print(f"✗ Failed to import main APIs: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Verify no imports from project root /models/ or /modules/
print("\n2. Checking that models are NOT imported from project root...")
import_errors = []

# Check if any toolkit modules are importing from project root
toolkit_path = project_root / 'cardiac_toolkit_to_be_organized'
for py_file in toolkit_path.rglob('*.py'):
    with open(py_file, 'r') as f:
        content = f.read()
        # Check for bad imports (but exclude comments)
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith('#'):
                continue
            if 'from models.' in line or 'from modules.trainer' in line:
                if 'parent.parent.parent.parent.parent' not in line:  # Allow old code in comments
                    import_errors.append(f"{py_file.relative_to(project_root)}:{line_num}: {line}")

if import_errors:
    print("✗ Found imports from project root:")
    for error in import_errors[:10]:  # Show first 10
        print(f"  {error}")
    if len(import_errors) > 10:
        print(f"  ... and {len(import_errors - 10)} more")
else:
    print("✓ No imports from project root /models/ or /modules/")

# Test 3: Verify lamod module structure (reorganized with subdirectories)
print("\n3. Checking lamod module structure...")
lamod_dir = project_root / 'cardiac_toolkit_to_be_organized/motion/tracking/lamod'
required_files = [
    # Models subdirectory
    'models/regnet.py',
    'models/video_diffusion.py',
    'models/decoder.py',
    'models/lamod_network.py',
    'models/build_model.py',
    # Registration subdirectory
    'registration/svf.py',
    'registration/epdiff_lib.py',
    # Diffusion subdirectory
    'diffusion/attention.py',
    'diffusion/diffusion_util.py',
    'diffusion/noise_smoothing.py',
    # Utils subdirectory
    'utils/patchify_utils.py',
    'utils/normalize.py',
    'utils/preprocessing.py',
    # Configs
    'configs/registration.json',
    'configs/lamod_full.json',
    # Main module files
    'api.py',
    'model.py',
    '__init__.py'
]

missing_files = []
for file in required_files:
    if not (lamod_dir / file).exists():
        missing_files.append(file)

if missing_files:
    print(f"✗ Missing files in lamod/:")
    for f in missing_files:
        print(f"  - {f}")
else:
    print("✓ All required files present in organized structure")
    print("  - models/ (RegNet, LaMoD, VideoDiffusion, Decoder)")
    print("  - registration/ (SVF, Epdiff)")
    print("  - diffusion/ (Attention, utilities)")
    print("  - utils/ (Patchify, normalize, preprocessing)")
    print("  - configs/ (JSON configurations)")

# Test 4: Test actual import of RegNet
print("\n4. Testing RegNet import from lamod...")
try:
    from cardiac_toolkit_to_be_organized.motion.tracking.lamod import RegNet
    print(f"✓ RegNet imported: {RegNet}")
except Exception as e:
    print(f"✗ Failed to import RegNet: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test build_model import
print("\n5. Testing build_model import from lamod...")
try:
    from cardiac_toolkit_to_be_organized.motion.tracking.lamod import build_model
    print(f"✓ build_model imported: {build_model}")
except Exception as e:
    print(f"✗ Failed to import build_model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("IMPORT TEST COMPLETE")
print("=" * 70)

if not import_errors and not missing_files:
    print("\n✓ SUCCESS: Toolkit is self-contained!")
    print("  All models are in lamod/, no external dependencies on /models/ or /modules/")
    sys.exit(0)
else:
    print("\n✗ ISSUES FOUND - see details above")
    sys.exit(1)
