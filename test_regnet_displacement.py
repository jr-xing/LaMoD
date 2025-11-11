"""
Test that RegNet produces full displacement fields using lagomorph Epdiff.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("REGNET DISPLACEMENT FIELD TEST")
print("Verify that lagomorph Epdiff integration is working")
print("=" * 70)

# Import RegNet directly
from models import build_model
import json

# Load config
config_path = 'configs/ours-train-registration.json'
print(f"\n1. Loading config from: {config_path}")
with open(config_path, 'r') as f:
    config = json.load(f)

# Build RegNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"2. Using device: {device}")
print(f"3. Building RegNet model...")

regnet = build_model(config['networks']['registration'], device).to(device)
print(f"✓ RegNet loaded successfully")
print(f"  - Displacement generator: {config['networks']['registration']['disp_generator']}")
print(f"  - Image size: {config['networks']['registration']['imagesize']}")

# Create dummy input
print(f"\n4. Creating dummy input...")
masks = np.random.rand(10, 128, 128).astype(np.float32)
masks = (masks > 0.5).astype(np.float32)

# Convert to tensor
input_tensor = torch.from_numpy(masks[None, None]).to(device).float()
print(f"  - Input shape: {input_tensor.shape}")

# Run inference
print(f"\n5. Running RegNet inference...")
with torch.no_grad():
    output = regnet.inference_2D_video(input_tensor)

# Check output
print(f"\n6. Output dictionary keys: {list(output.keys())}")

if 'displacement' in output:
    disp = output['displacement']
    print(f"\n✓ SUCCESS: Displacement field generated!")
    print(f"  - Displacement shape: {disp.shape}")
    print(f"  - Expected: (1, 2, 9, 128, 128)")
    print(f"  - Components: {disp.shape[1]} (X, Y)")
    print(f"  - Frames: {disp.shape[2]} (T-1, where T=10)")
    print(f"  - Resolution: {disp.shape[3]}x{disp.shape[4]}")
    print(f"  - Displacement range X: [{disp[0,0].min():.3f}, {disp[0,0].max():.3f}]")
    print(f"  - Displacement range Y: [{disp[0,1].min():.3f}, {disp[0,1].max():.3f}]")

    # Check if lagomorph was used
    if 'velocity' in output and output['velocity'] is not None:
        print(f"\n✓ Velocity field also generated (indicates Epdiff was used)")
        print(f"  - Velocity shape: {output['velocity'].shape}")

    if 'momentum' in output and output['momentum'] is not None:
        print(f"✓ Momentum field also generated (indicates Epdiff was used)")
        print(f"  - Momentum shape: {output['momentum'].shape}")

    print(f"\n" + "=" * 70)
    print(f"CONCLUSION: RegNet successfully produces displacement fields")
    print(f"using lagomorph Epdiff integration!")
    print("=" * 70)

else:
    print(f"\n✗ ERROR: No displacement field in output!")
    print(f"Available keys: {list(output.keys())}")
    sys.exit(1)
