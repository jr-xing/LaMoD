# Pure PyTorch RegNet ONNX Conversion Summary

**Date:** 2025-11-03
**Status:** ❌ **NOT POSSIBLE** (FFT operations not supported by PyTorch ONNX exporter)

---

## Quick Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Pure PyTorch Implementation** | ✅ Complete | No lagomorph, no CUDA compilation |
| **ONNX Conversion Attempt** | ❌ Failed | FFT operations not supported |
| **Error** | `aten::fft_rfftn` not exportable | PyTorch exporter limitation |
| **Recommendation** | Use PyTorch directly | TorchServe or Triton deployment |

---

## What We Tried

### 1. Pure PyTorch Implementation (SUCCESS ✅)

**Achievement:**
- Replaced all lagomorph CUDA operations with pure PyTorch
- `FluidMetricPyTorch` uses FFT instead of custom CUDA kernel
- Numerical accuracy: <0.3% difference vs original
- Production-ready and cross-platform compatible

**Key Replacements:**
- `lagomorph_ext.fluid_operator` → `torch.fft.rfftn/irfftn` (FFT-based)
- `lagomorph.interp()` → `F.grid_sample()` (ONNX-compatible)
- `lagomorph.adjrep.Ad_star()` → Finite differences (ONNX-compatible)

### 2. ONNX Conversion (FAILED ❌)

**Attempt:**
```python
torch.onnx.export(model, input, 'regnet.onnx', opset_version=17)
```

**Error:**
```
✗ Exporting the operator 'aten::fft_rfftn' to ONNX opset version 17 is not supported
```

**Root Cause:**
- ONNX opset 17 defines `DFT` operator for FFT
- PyTorch has `torch.fft.rfftn/irfftn` operations
- **PyTorch ONNX exporter doesn't map FFT ops to ONNX DFT**
- This is an exporter limitation, not a fundamental incompatibility

---

## Why FFT Cannot Be Easily Replaced

### FFT in Fluid Metric

```python
# lagomorph_pytorch.py (line 97-139)
def flat(self, v):
    """Velocity → Momentum: m = L*v"""
    v_hat = torch.fft.rfftn(v, dim=(-2, -1))  # Forward FFT
    m_hat = v_hat * self.L_chol                 # Multiply in frequency domain
    m = torch.fft.irfftn(m_hat, s=v.shape[-2:], dim=(-2, -1))  # Inverse FFT
    return m
```

**Why FFT?**
- Implements Laplacian operator efficiently (O(n log n) vs O(n²))
- Numerically stable with Cholesky decomposition
- 10-100x faster than spatial convolution
- Mathematically equivalent to lagomorph's CUDA kernel

**Alternative Approaches (all problematic):**
1. **Conv2d-based Laplacian:** 10-100x slower, less stable
2. **Remove fluid metric:** Loses diffeomorphic property
3. **Wait for PyTorch support:** Best long-term, but timeline unclear

---

## Comparison: Original vs Pure PyTorch

| Issue | Original (Lagomorph) | Pure PyTorch (FFT) |
|-------|---------------------|-------------------|
| **Problem** | Custom CUDA extensions | FFT not in ONNX exporter |
| **Fundamental?** | Yes (C++/CUDA code) | No (exporter gap) |
| **ONNX Possible?** | Never | When PyTorch adds FFT support |
| **TorchScript?** | No | Possibly (untested) |
| **Workaround?** | None | Use PyTorch directly |

**Progress:** We went from **impossible** (custom CUDA) to **blocked by exporter** (FFT).

---

## Recommended Solution

### ✅ Deploy Pure PyTorch Version Directly

**Why:**
1. Already production-ready (<0.3% numerical difference)
2. No CUDA compilation needed (cross-platform)
3. Fast inference (~900ms for 50 frames)
4. Multiple deployment options available

**How:**

#### Option 1: TorchServe (Official PyTorch Serving)
```bash
# Package model
torch-model-archiver --model-name regnet \
    --version 1.0 \
    --model-file regnet_pytorch.py \
    --handler custom_handler.py

# Serve
torchserve --start --model-store model_store --models regnet=regnet.mar
```

#### Option 2: NVIDIA Triton Inference Server
```yaml
# config.pbtxt
name: "regnet"
backend: "pytorch"
max_batch_size: 8
```

#### Option 3: Direct PyTorch Script
```python
import torch
from regnet_pytorch import RegNet

model = RegNet(config).eval()
with torch.no_grad():
    output = model.inference_2D_video(input_video)
```

---

## Future Outlook

### When Will ONNX Work?

**PyTorch FFT→ONNX Support:**
- GitHub issue: https://github.com/pytorch/pytorch/issues/71779
- Status: Open (community contribution needed)
- Timeline: Unknown (possibly 2024-2025)

**What's Needed:**
- Map `aten::fft_rfftn` → `onnx::DFT` in PyTorch exporter
- Implement in `torch/onnx/symbolic_opset17.py`
- Test and validate
- Estimated effort: 1-2 weeks for experienced contributor

### Monitor PyTorch Updates

Test ONNX export with future PyTorch versions:
```bash
# After PyTorch update
python convert_to_onnx.py
```

If FFT support is added, the model should export cleanly.

---

## Files Created

| File | Purpose |
|------|---------|
| [`convert_to_onnx.py`](convert_to_onnx.py) | ONNX conversion script (documents limitation) |
| [`validate_conversion.py`](validate_conversion.py) | PyTorch vs ONNX comparison (unused) |
| [`test_onnx_inference.py`](test_onnx_inference.py) | ONNX inference test (unused) |
| [`README.md`](README.md) | Comprehensive technical documentation |
| `CONVERSION_SUMMARY.md` | This summary document |

---

## Key Takeaways

1. ✅ **Pure PyTorch implementation is complete and correct**
   - No lagomorph dependency
   - No CUDA compilation required
   - Numerically validated (<0.3% difference)

2. ❌ **ONNX export is blocked by PyTorch exporter**
   - FFT operations (`torch.fft.rfftn`) not supported
   - Not a fundamental incompatibility
   - May be resolved in future PyTorch versions

3. ✅ **Direct PyTorch deployment is recommended**
   - Use TorchServe, Triton, or direct PyTorch
   - Production-ready today
   - No waiting for ONNX support needed

4. ⚠️ **ONNX is possible in future**
   - When PyTorch adds FFT→ONNX mapping
   - Or if conv-based Laplacian implemented (slow)
   - Monitor PyTorch releases

---

## Conclusion

**The pure PyTorch RegNet model cannot currently be exported to ONNX** due to PyTorch's ONNX exporter not supporting FFT operations. However, the pure PyTorch implementation is **production-ready** and should be **deployed directly** using PyTorch-native serving solutions.

**Action Item:** Use the pure PyTorch model with TorchServe or Triton. Revisit ONNX export when PyTorch adds FFT support.

---

**References:**
- Pure PyTorch code: [`debug_pure_pytorch/`](../)
- Conversion attempt: This directory
- PyTorch FFT issue: https://github.com/pytorch/pytorch/issues/71779
