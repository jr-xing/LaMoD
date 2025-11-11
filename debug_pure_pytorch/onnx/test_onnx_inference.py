#!/usr/bin/env python3
"""
Test ONNX model inference

Simple script to test the exported ONNX model with dummy data.

Author: Claude
Date: 2025-11-03
"""

import numpy as np
import time
from pathlib import Path


def test_onnx_inference(onnx_path, input_shape=(1, 1, 50, 128, 128)):
    """Test ONNX model inference.

    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape

    Returns:
        outputs: Model outputs
        inference_time: Time taken for inference (seconds)
    """
    import onnxruntime as ort

    print(f"Testing ONNX inference...")
    print(f"Model: {onnx_path}")

    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available_providers = ort.get_available_providers()
    session_providers = [p for p in providers if p in available_providers]

    print(f"\nAvailable providers: {available_providers}")
    print(f"Using providers: {session_providers}")

    session = ort.InferenceSession(str(onnx_path), providers=session_providers)

    # Print model info
    print(f"\nModel Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")

    print(f"\nModel Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Create dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    print(f"\nInput shape: {dummy_input.shape}")

    # Warm-up
    print(f"\nWarming up...")
    for _ in range(3):
        _ = session.run(None, {'video': dummy_input})

    # Time inference
    print(f"Running inference...")
    n_runs = 10
    start_time = time.time()
    for _ in range(n_runs):
        outputs = session.run(None, {'video': dummy_input})
    end_time = time.time()

    inference_time = (end_time - start_time) / n_runs

    print(f"\nInference Results:")
    print(f"  Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"  Output {i} shape: {output.shape}")
        print(f"    Min: {output.min():.6f}, Max: {output.max():.6f}, Mean: {output.mean():.6f}")

    print(f"\nTiming (average over {n_runs} runs):")
    print(f"  Inference time: {inference_time*1000:.2f} ms")
    print(f"  Throughput: {1/inference_time:.2f} samples/sec")

    return outputs, inference_time


def main():
    """Main test function."""
    print("="*70)
    print("ONNX Model Inference Test")
    print("="*70)

    # Paths
    base_dir = Path('/gpfs/gibbs/project/kwan/jx332/code/2025-08-LaMoD')
    onnx_path = base_dir / 'debug_pure_pytorch/onnx/regnet_pure_pytorch.onnx'

    if not onnx_path.exists():
        print(f"\n✗ ONNX model not found: {onnx_path}")
        print(f"\nPlease run convert_to_onnx.py first:")
        print(f"  python debug_pure_pytorch/onnx/convert_to_onnx.py")
        return

    try:
        outputs, inference_time = test_onnx_inference(onnx_path)

        print(f"\n{'='*70}")
        print(f"✓ Test completed successfully!")
        print(f"{'='*70}")
        print(f"ONNX model is working correctly.")
        print(f"Average inference time: {inference_time*1000:.2f} ms")

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print(f"\nPlease install onnxruntime:")
        print(f"  pip install onnxruntime-gpu  # for GPU")
        print(f"  pip install onnxruntime       # for CPU")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
