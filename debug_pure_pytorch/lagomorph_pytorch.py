"""
Pure PyTorch implementation of lagomorph operations.
Replaces CUDA extensions with native PyTorch operations.

This module provides drop-in replacements for:
- lm.interp() - Image interpolation via displacement fields
- lm.FluidMetric - Fluid metric operator for LDDMM registration
"""

import torch
import torch.nn.functional as F
import numpy as np


def interp_pytorch(I, u, dt=1.0):
    """
    Interpolate image I using displacement field u with optional time scaling dt.

    Computes: out(x) = I(x + dt*u(x))

    This is a pure PyTorch replacement for lagomorph.interp() using F.grid_sample.

    Args:
        I: Input image tensor of shape (N, C, H, W) for 2D or (N, C, D, H, W) for 3D
        u: Displacement field of shape (N, dim, H, W) or (N, dim, D, H, W)
           where dim=2 for 2D or dim=3 for 3D
        dt: Time scaling factor (default: 1.0)

    Returns:
        Deformed image of same shape as I
    """
    N, C, *spatial_dims = I.shape
    dim = len(spatial_dims)

    if dim == 2:
        H, W = spatial_dims
        # Create identity grid in [0, H-1] x [0, W-1]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=I.dtype, device=I.device),
            torch.arange(W, dtype=I.dtype, device=I.device),
            indexing='ij'
        )

        # Add displacement (scaled by dt)
        # u is (N, 2, H, W) where u[:,0] is X displacement, u[:,1] is Y displacement
        grid_x_def = grid_x[None, :, :] + dt * u[:, 0, :, :]
        grid_y_def = grid_y[None, :, :] + dt * u[:, 1, :, :]

        # Normalize to [-1, 1] for grid_sample
        grid_x_norm = 2.0 * grid_x_def / (W - 1) - 1.0
        grid_y_norm = 2.0 * grid_y_def / (H - 1) - 1.0

        # Stack to (N, H, W, 2) - grid_sample expects (x, y) order
        grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1)

    elif dim == 3:
        D, H, W = spatial_dims
        # Create identity grid in [0, D-1] x [0, H-1] x [0, W-1]
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.arange(D, dtype=I.dtype, device=I.device),
            torch.arange(H, dtype=I.dtype, device=I.device),
            torch.arange(W, dtype=I.dtype, device=I.device),
            indexing='ij'
        )

        # Add displacement (scaled by dt)
        # u is (N, 3, D, H, W) where u[:,0] is X, u[:,1] is Y, u[:,2] is Z displacement
        grid_x_def = grid_x[None, :, :, :] + dt * u[:, 0, :, :, :]
        grid_y_def = grid_y[None, :, :, :] + dt * u[:, 1, :, :, :]
        grid_z_def = grid_z[None, :, :, :] + dt * u[:, 2, :, :, :]

        # Normalize to [-1, 1] for grid_sample
        grid_x_norm = 2.0 * grid_x_def / (W - 1) - 1.0
        grid_y_norm = 2.0 * grid_y_def / (H - 1) - 1.0
        grid_z_norm = 2.0 * grid_z_def / (D - 1) - 1.0

        # Stack to (N, D, H, W, 3) - grid_sample expects (x, y, z) order
        grid = torch.stack([grid_x_norm, grid_y_norm, grid_z_norm], dim=-1)

    else:
        raise ValueError(f"Only 2D and 3D interpolation supported, got {dim}D")

    # Use bilinear/trilinear interpolation with zero padding for out-of-bounds
    return F.grid_sample(I, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


class FluidMetricPyTorch:
    """
    Pure PyTorch implementation of FluidMetric.

    Implements the Green's function for the differential operator:
        L†L = -α∇² - β grad(div) + γ

    where:
        α (alpha): coefficient for vector Laplacian term (smoothness)
        β (beta): coefficient for gradient divergence term (incompressibility)
        γ (gamma): coefficient for L2 regularization term

    This replaces lagomorph.FluidMetric with pure PyTorch operations.
    """

    def __init__(self, params=[0.1, 0.0, 0.001]):
        """
        Initialize FluidMetric.

        Args:
            params: List of [alpha, beta, gamma] parameters
        """
        assert len(params) == 3, "params must be [alpha, beta, gamma]"
        self.params = params
        self.alpha, self.beta, self.gamma = params

        # Cache for lookup tables
        self.shape = None
        self.luts = None

    def initialize_luts(self, shape, dtype, device):
        """
        Initialize lookup tables for sine and cosine values.

        Args:
            shape: Tensor shape (N, C, H, W) or (N, C, D, H, W)
            dtype: Data type
            device: Device (cpu or cuda)
        """
        if self.shape != shape:
            self.shape = shape
            spatial_dims = shape[2:]

            # Compute complex shape (for rfft output)
            self.complexshape = list(shape)
            self.complexshape[-1] = self.complexshape[-1] // 2 + 1
            self.complexshape = tuple(self.complexshape)

            # Precompute cos/sin lookup tables for each dimension
            self.luts = {'cos': [], 'sin': []}
            for Nf, N in zip(self.complexshape[2:], spatial_dims):
                # cos lookup: 2*(1 - cos(2*pi*k/N))
                cos_lut = 2.0 * (1.0 - torch.cos(2 * np.pi * torch.arange(Nf, dtype=dtype, device=device) / N))
                self.luts['cos'].append(cos_lut)

                # sin lookup: sin(2*pi*k/N)
                sin_lut = torch.sin(2 * np.pi * torch.arange(Nf, dtype=dtype, device=device) / N)
                self.luts['sin'].append(sin_lut)

    def operator(self, mv, inverse):
        """
        Apply fluid operator (or its inverse) to momentum/velocity field.

        Args:
            mv: Momentum or velocity field (N, C, H, W) or (N, C, D, H, W)
            inverse: If True, apply inverse operator (sharp); if False, apply forward (flat)

        Returns:
            Result of operator application
        """
        # Initialize lookup tables if needed
        self.initialize_luts(shape=mv.shape, dtype=mv.dtype, device=mv.device)

        N, C, *spatial_dims = mv.shape
        dim = len(spatial_dims)

        # Forward FFT
        if dim == 2:
            fft_dims = (-2, -1)
        elif dim == 3:
            fft_dims = (-3, -2, -1)
        else:
            raise ValueError(f"Only 2D and 3D supported, got {dim}D")

        Fmv = torch.fft.rfftn(mv, dim=fft_dims, norm="forward")

        # Apply operator in Fourier domain
        Fmv = self._apply_fourier_operator(Fmv, inverse, dim)

        # Inverse FFT
        result = torch.fft.irfftn(Fmv, s=spatial_dims, dim=fft_dims, norm="forward")

        return result

    def _apply_fourier_operator(self, Fmv, inverse, dim):
        """
        Apply fluid operator in Fourier domain.

        For each Fourier coefficient, we have a symmetric matrix L that we either:
        - Multiply by (forward operator)
        - Solve against using Cholesky decomposition (inverse operator)
        """
        N, C, *freq_dims = Fmv.shape

        if dim == 2:
            return self._apply_fourier_operator_2d(Fmv, inverse, N, C, freq_dims)
        elif dim == 3:
            return self._apply_fourier_operator_3d(Fmv, inverse, N, C, freq_dims)
        else:
            raise ValueError(f"Only 2D and 3D supported")

    def _apply_fourier_operator_2d(self, Fmv, inverse, N, C, freq_dims):
        """Apply 2D fluid operator in Fourier domain."""
        assert C == 2, "2D vector field must have 2 components"

        nx, ny = freq_dims

        # Get frequency grids
        wx = self.luts['cos'][0][:, None]  # (nx, 1)
        wy = self.luts['cos'][1][None, :]  # (1, ny)
        sx = self.luts['sin'][0][:, None]
        sy = self.luts['sin'][1][None, :]

        # Compute diagonal regularization term
        # lambda = gamma + alpha * (wx + wy)
        lambda_diag = self.gamma + self.alpha * (wx + wy)

        # Build lower triangular matrix L
        # l00 = lambda - beta * wx
        # l11 = lambda - beta * wy
        # l10 = beta * sin_x * sin_y
        l00 = lambda_diag - self.beta * wx
        l11 = lambda_diag - self.beta * wy
        l10 = self.beta * sx * sy

        # Square the matrix: L†L
        L00 = l00 * l00 + l10 * l10
        L10 = l00 * l10 + l10 * l11
        L11 = l11 * l11 + l10 * l10

        # Process each batch
        result = torch.zeros_like(Fmv)

        for n in range(N):
            Fmx = Fmv[n, 0]  # (nx, ny) complex
            Fmy = Fmv[n, 1]  # (nx, ny) complex

            if inverse:
                # Solve L * x = Fmv using Cholesky decomposition
                # Cholesky factor: L = G * G^T where G is lower triangular

                # Safe sqrt to avoid numerical issues
                def safe_sqrt(x):
                    return torch.sqrt(torch.clamp(x, min=1e-8))

                # Compute Cholesky factor G
                ooG00 = 1.0 / safe_sqrt(L00)
                G10 = L10 * ooG00
                ooG11 = 1.0 / safe_sqrt(L11 - G10 * G10)

                # Solve G * y = b (forward substitution)
                y0_r = Fmx.real * ooG00
                y0_i = Fmx.imag * ooG00
                y1_r = (Fmy.real - G10 * y0_r) * ooG11
                y1_i = (Fmy.imag - G10 * y0_i) * ooG11

                # Solve G^T * x = y (backward substitution)
                x1_r = y1_r * ooG11
                x1_i = y1_i * ooG11
                x0_r = (y0_r - G10 * x1_r) * ooG00
                x0_i = (y0_i - G10 * x1_i) * ooG00

                result[n, 0] = torch.complex(x0_r, x0_i)
                result[n, 1] = torch.complex(x1_r, x1_i)
            else:
                # Forward multiply: y = L * x
                y0_r = L00 * Fmx.real + L10 * Fmy.real
                y0_i = L00 * Fmx.imag + L10 * Fmy.imag
                y1_r = L10 * Fmx.real + L11 * Fmy.real
                y1_i = L10 * Fmx.imag + L11 * Fmy.imag

                result[n, 0] = torch.complex(y0_r, y0_i)
                result[n, 1] = torch.complex(y1_r, y1_i)

        return result

    def _apply_fourier_operator_3d(self, Fmv, inverse, N, C, freq_dims):
        """Apply 3D fluid operator in Fourier domain."""
        assert C == 3, "3D vector field must have 3 components"

        nx, ny, nz = freq_dims

        # Get frequency grids
        wx = self.luts['cos'][0][:, None, None]  # (nx, 1, 1)
        wy = self.luts['cos'][1][None, :, None]  # (1, ny, 1)
        wz = self.luts['cos'][2][None, None, :]  # (1, 1, nz)
        sx = self.luts['sin'][0][:, None, None]
        sy = self.luts['sin'][1][None, :, None]
        sz = self.luts['sin'][2][None, None, :]

        # Compute diagonal regularization term
        lambda_diag = self.gamma + self.alpha * (wx + wy + wz)

        # Build lower triangular matrix L
        l00 = lambda_diag - self.beta * wx
        l11 = lambda_diag - self.beta * wy
        l22 = lambda_diag - self.beta * wz
        l10 = self.beta * sx * sy
        l20 = self.beta * sx * sz
        l21 = self.beta * sy * sz

        # Square the matrix: L†L
        L00 = l00 * l00 + l10 * l10 + l20 * l20
        L10 = l00 * l10 + l10 * l11 + l20 * l21
        L11 = l11 * l11 + l10 * l10 + l21 * l21
        L20 = l00 * l20 + l10 * l21 + l20 * l22
        L21 = l10 * l20 + l11 * l21 + l21 * l22
        L22 = l22 * l22 + l20 * l20 + l21 * l21

        # Process each batch
        result = torch.zeros_like(Fmv)

        for n in range(N):
            Fmx = Fmv[n, 0]  # (nx, ny, nz) complex
            Fmy = Fmv[n, 1]
            Fmz = Fmv[n, 2]

            if inverse:
                # Solve using 3D Cholesky decomposition
                def safe_sqrt(x):
                    return torch.sqrt(torch.clamp(x, min=1e-8))

                # Compute Cholesky factor G
                ooG00 = 1.0 / safe_sqrt(L00)
                G10 = L10 * ooG00
                G20 = L20 * ooG00
                ooG11 = 1.0 / safe_sqrt(L11 - G10 * G10)
                G21 = (L21 - G20 * G10) * ooG11
                ooG22 = 1.0 / safe_sqrt(L22 - G20 * G20 - G21 * G21)

                # Forward substitution: G * y = b
                y0_r = Fmx.real * ooG00
                y0_i = Fmx.imag * ooG00
                y1_r = (Fmy.real - G10 * y0_r) * ooG11
                y1_i = (Fmy.imag - G10 * y0_i) * ooG11
                y2_r = (Fmz.real - G20 * y0_r - G21 * y1_r) * ooG22
                y2_i = (Fmz.imag - G20 * y0_i - G21 * y1_i) * ooG22

                # Backward substitution: G^T * x = y
                x2_r = y2_r * ooG22
                x2_i = y2_i * ooG22
                x1_r = (y1_r - G21 * x2_r) * ooG11
                x1_i = (y1_i - G21 * x2_i) * ooG11
                x0_r = (y0_r - G10 * x1_r - G20 * x2_r) * ooG00
                x0_i = (y0_i - G10 * x1_i - G20 * x2_i) * ooG00

                result[n, 0] = torch.complex(x0_r, x0_i)
                result[n, 1] = torch.complex(x1_r, x1_i)
                result[n, 2] = torch.complex(x2_r, x2_i)
            else:
                # Forward multiply: y = L * x
                y0_r = L00 * Fmx.real + L10 * Fmy.real + L20 * Fmz.real
                y0_i = L00 * Fmx.imag + L10 * Fmy.imag + L20 * Fmz.imag
                y1_r = L10 * Fmx.real + L11 * Fmy.real + L21 * Fmz.real
                y1_i = L10 * Fmx.imag + L11 * Fmy.imag + L21 * Fmz.imag
                y2_r = L20 * Fmx.real + L21 * Fmy.real + L22 * Fmz.real
                y2_i = L20 * Fmx.imag + L21 * Fmy.imag + L22 * Fmz.imag

                result[n, 0] = torch.complex(y0_r, y0_i)
                result[n, 1] = torch.complex(y1_r, y1_i)
                result[n, 2] = torch.complex(y2_r, y2_i)

        return result

    def sharp(self, m):
        """
        Raise indices: convert momentum (covector field) to velocity (vector field).

        Applies the inverse operator (L†L)^(-1) which smooths the momentum.

        Args:
            m: Momentum field

        Returns:
            Velocity field
        """
        return self.operator(m, inverse=True)

    def flat(self, v):
        """
        Lower indices: convert velocity (vector field) to momentum (covector field).

        Applies the forward operator L†L.

        Args:
            v: Velocity field

        Returns:
            Momentum field
        """
        return self.operator(v, inverse=False)
