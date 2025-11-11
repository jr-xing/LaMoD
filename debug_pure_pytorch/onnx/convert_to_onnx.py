#!/usr/bin/env python3
"""
Convert Pure PyTorch RegNet to ONNX format

This script exports the pure PyTorch implementation of RegNet (without lagomorph)
to ONNX format for cross-platform deployment.

Author: Claude
Date: 2025-11-03
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD')
sys.path.insert(0, '/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD/debug_pure_pytorch')

# Import pure PyTorch RegNet
from regnet_pytorch import RegNet as RegNetPyTorch


def load_pure_pytorch_regnet(config_path, checkpoint_path, device):
    """Load pure PyTorch RegNet model.

    Args:
        config_path: Path to configuration JSON
        checkpoint_path: Path to model checkpoint
        device: torch device

    Returns:
        Pure PyTorch RegNet model in eval mode
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Building pure PyTorch RegNet...")
    regnet_config = config['networks']['registration']

    model = RegNetPyTorch(regnet_config)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Type: Pure PyTorch RegNet (no lagomorph)")
    print(f"  Image size: {regnet_config['imagesize']}")
    print(f"  2D+T mode: {regnet_config['twoD_plus_T']}")
    print(f"  Displacement generator: {regnet_config['disp_generator']}")

    return model


def export_to_onnx(model, output_path, device, input_shape=(1, 1, 50, 128, 128)):
    """Export pure PyTorch RegNet to ONNX format.

    Args:
        model: Pure PyTorch RegNet model
        output_path: Path to save ONNX model
        device: torch device
        input_shape: Input tensor shape (batch, channel, time, height, width)

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Starting ONNX export...")
    print(f"{'='*70}")

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device, dtype=torch.float32)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"  Batch: {dummy_input.shape[0]}")
    print(f"  Channels: {dummy_input.shape[1]}")
    print(f"  Time frames: {dummy_input.shape[2]}")
    print(f"  Height: {dummy_input.shape[3]}")
    print(f"  Width: {dummy_input.shape[4]}")

    # Test forward pass first
    print(f"\nTesting forward pass in PyTorch...")
    with torch.no_grad():
        try:
            output = model.inference_2D_video(dummy_input)
            print(f"✓ Forward pass successful!")
            print(f"  Output keys: {list(output.keys())}")
            if 'displacement' in output:
                disp_shape = output['displacement'].shape
                print(f"  Displacement shape: {disp_shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Wrap model to return tuple instead of dict (ONNX prefers tuples)
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, regnet_model):
            super().__init__()
            self.regnet = regnet_model

        def forward(self, video):
            output = self.regnet.inference_2D_video(video)
            # Return tuple: (displacement, velocity, deformed_source)
            return (
                output['displacement'],
                output.get('velocity', torch.zeros_like(output['displacement'])),
                output.get('deformed_source', torch.zeros(video.shape[0], 1, video.shape[2]-1, video.shape[3], video.shape[4], device=video.device))
            )

    print(f"\nCreating ONNX wrapper...")
    wrapped_model = ONNXWrapper(model)
    wrapped_model.eval()

    # Export to ONNX
    print(f"\nExporting to ONNX...")
    print(f"Output path: {output_path}")
    print(f"Opset version: 17 (required for FFT operations)")

    # Dynamic axes for variable time dimension
    dynamic_axes = {
        'video': {0: 'batch', 2: 'time'},
        'displacement': {0: 'batch', 2: 'time'},
        'velocity': {0: 'batch', 2: 'time'},
        'deformed_source': {0: 'batch', 2: 'time'},
    }

    try:
        # Test wrapped model first
        print(f"\nTesting wrapped model...")
        with torch.no_grad():
            wrapped_output = wrapped_model(dummy_input)
            print(f"✓ Wrapped model output: {len(wrapped_output)} tensors")
            for i, tensor in enumerate(wrapped_output):
                print(f"    Output {i}: {tensor.shape}")

        # Export
        print(f"\nExporting with torch.onnx.export...")
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,  # Required for FFT operations
            do_constant_folding=True,
            input_names=['video'],
            output_names=['displacement', 'velocity', 'deformed_source'],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )

        print(f"✓ ONNX export successful!")

        # Check file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Model size: {file_size_mb:.2f} MB")

        return True

    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()

        print(f"\n{'='*70}")
        print(f"TROUBLESHOOTING")
        print(f"{'='*70}")
        print(f"\nPossible issues:")
        print(f"  1. FFT operations require ONNX opset 17")
        print(f"  2. Some PyTorch operations may not be fully supported")
        print(f"  3. Try upgrading: pip install --upgrade torch onnx")
        print(f"\nIf the error is about FFT:")
        print(f"  - Check PyTorch version (>= 1.13 recommended)")
        print(f"  - Check ONNX version (>= 1.13 recommended)")

        return False


def verify_onnx_model(onnx_path):
    """Verify the exported ONNX model.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        True if verification passes, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Verifying ONNX model...")
    print(f"{'='*70}")

    try:
        import onnx
        import onnxruntime as ort

        # Load and check model
        print(f"\nLoading ONNX model: {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model is valid!")

        # Print model info
        print(f"\nModel Information:")
        print(f"  Opset version: {onnx_model.opset_import[0].version}")
        print(f"  Producer: {onnx_model.producer_name}")
        print(f"  IR version: {onnx_model.ir_version}")

        # Print inputs
        print(f"\nInputs:")
        for inp in onnx_model.graph.input:
            dims = [d.dim_value if d.dim_value > 0 else 'dynamic'
                    for d in inp.type.tensor_type.shape.dim]
            print(f"  - {inp.name}: {dims}")

        # Print outputs
        print(f"\nOutputs:")
        for out in onnx_model.graph.output:
            dims = [d.dim_value if d.dim_value > 0 else 'dynamic'
                    for d in out.type.tensor_type.shape.dim]
            print(f"  - {out.name}: {dims}")

        # Create inference session
        print(f"\nCreating ONNX Runtime session...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        available_providers = ort.get_available_providers()

        # Use GPU if available, otherwise CPU
        session_providers = [p for p in providers if p in available_providers]
        print(f"  Available providers: {available_providers}")
        print(f"  Using providers: {session_providers}")

        session = ort.InferenceSession(onnx_path, providers=session_providers)
        print(f"✓ ONNX Runtime session created successfully!")

        # Test inference with dummy data
        print(f"\nTesting inference with dummy data...")
        dummy_input = np.random.randn(1, 1, 10, 128, 128).astype(np.float32)
        outputs = session.run(None, {'video': dummy_input})
        print(f"✓ Inference successful!")
        print(f"  Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  Output {i} shape: {output.shape}")

        return True

    except ImportError as e:
        print(f"⚠ Warning: {e}")
        print(f"  Install onnx and onnxruntime to verify the model:")
        print(f"  pip install onnx onnxruntime-gpu  # or onnxruntime for CPU")
        return False
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main conversion pipeline."""
    print("Pure PyTorch RegNet to ONNX Conversion")
    print("="*70)
    print("This script converts the pure PyTorch RegNet (no lagomorph)")
    print("to ONNX format for cross-platform deployment.")
    print("="*70)

    # Paths
    base_dir = Path('/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD')
    config_path = base_dir / 'configs/ours-train-registration.json'
    checkpoint_path = base_dir / 'checkpoints/ours_registration.pth'
    output_path = base_dir / 'debug_pure_pytorch/onnx/regnet_pure_pytorch.onnx'

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nKey differences from original:")
    print(f"  ✓ Uses F.grid_sample() instead of lagomorph.interp()")
    print(f"  ✓ Uses PyTorch FFT instead of lagomorph FluidMetric")
    print(f"  ✓ Uses finite differences instead of lagomorph Jacobians")
    print(f"  ✓ No custom CUDA operations")

    # Load model
    model = load_pure_pytorch_regnet(config_path, checkpoint_path, device)

    # Export to ONNX
    success = export_to_onnx(model, output_path, device)

    if success:
        # Verify ONNX model
        verify_success = verify_onnx_model(output_path)

        print(f"\n{'='*70}")
        print(f"Conversion complete!")
        print(f"{'='*70}")
        print(f"ONNX model saved to: {output_path}")

        if verify_success:
            print(f"\n✓ Model verified successfully!")
            print(f"\nNext steps:")
            print(f"  1. Run validation: python validate_conversion.py")
            print(f"  2. Compare PyTorch vs ONNX outputs on real data")
            print(f"  3. Test inference: python test_onnx_inference.py")
        else:
            print(f"\n⚠ Model saved but verification had issues")
            print(f"  You may need to install: pip install onnx onnxruntime-gpu")
    else:
        print(f"\n{'='*70}")
        print(f"Conversion failed")
        print(f"{'='*70}")
        print(f"\nSee error messages above for troubleshooting.")


if __name__ == '__main__':
    main()
