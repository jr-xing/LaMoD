# Modify Lagomorph
The lagomorph installation guide can be found on the original [GitHub repository](https://github.com/jacobhinkle/lagomorph). Though for newer PyTorch version, the `FluidMetricOperator` class need to be manually updated since the `torch.irfft` function is expired. 

To update the class, go to `metrics.py`, and replace the `FluidMetricOperator`'s code with:

```Python
class FluidMetricOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, luts, inverse, mv):
        ctx.params = params
        ctx.luts = luts
        ctx.inverse = inverse
        sh = mv.shape
        spatial_dim = len(sh)-2
        """ Fmv = torch.rfft(mv, spatial_dim, normalized=True)
        lagomorph_ext.fluid_operator(Fmv, inverse,
                luts['cos'], luts['sin'], *params)
        return torch.irfft(Fmv, spatial_dim, normalized=True,
                signal_sizes=sh[2:]) """
        # Fmv = torch.rfft(mv, spatial_dim, normalized=True)
        dim = (-2,-1) if spatial_dim==2 else (-3,-2,-1)
        Fmv = torch.fft.rfftn(mv, dim=dim, norm="forward")
        Fmv = torch.stack((Fmv.real, Fmv.imag), -1)


        lagomorph_ext.fluid_operator(Fmv, inverse,
                luts['cos'], luts['sin'], *params)
        # return torch.irfft(Fmv, spatial_dim, normalized=True,
        #         signal_sizes=sh[2:])
        return torch.fft.irfftn(torch.complex(Fmv[..., 0], Fmv[..., 1]),s=sh[2:], dim=dim, norm="forward")
    @staticmethod
    def backward(ctx, outgrad):
        sh = outgrad.shape
        spatial_dim = len(sh)-2


        """ Fmv = torch.rfft(outgrad, spatial_dim, normalized=True)
        lagomorph_ext.fluid_operator(Fmv, ctx.inverse,
                ctx.luts['cos'], ctx.luts['sin'], *ctx.params)
        return None, None, None, torch.irfft(Fmv, spatial_dim, normalized=True,
                signal_sizes=sh[2:]) """

        # Fmv = torch.rfft(outgrad, spatial_dim, normalized=True)
        dim = (-2,-1) if spatial_dim==2 else (-3,-2,-1)
        Fmv = torch.fft.rfftn(outgrad, dim=dim, norm="forward")
        Fmv = torch.stack((Fmv.real, Fmv.imag), -1)

        lagomorph_ext.fluid_operator(Fmv, ctx.inverse,
                ctx.luts['cos'], ctx.luts['sin'], *ctx.params)
        # return None, None, None, torch.irfft(Fmv, spatial_dim, normalized=True,
        #         signal_sizes=sh[2:])
        return None, None, None, torch.fft.irfftn(torch.complex(Fmv[..., 0], Fmv[..., 1]), s=sh[2:], dim=dim,norm="forward")
```