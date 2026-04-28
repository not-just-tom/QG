import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Sequence


class SpectralConv2d(eqx.Module):
    """A 2D Fourier layer that multiplies learned complex weights on low-frequency modes."""
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int
    weight: jnp.ndarray  # complex-valued weight of shape (in, out, m1, m2)

    def __init__(self, in_channels, out_channels, modes1, modes2, key=None):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key, 2)
        scale = (1.0 / (in_channels * out_channels)) ** 0.5
        real = jax.random.normal(k1, (in_channels, out_channels, modes1, modes2), dtype=jnp.float32) * scale
        imag = jax.random.normal(k2, (in_channels, out_channels, modes1, modes2), dtype=jnp.float32) * scale
        self.weight = real + 1j * imag

    def __call__(self, x):
        # x expected shape: (B, C, H, W)
        added_batch = False
        if x.ndim == 3:
            x = x[None, ...]
            added_batch = True
        B, C, H, W = x.shape

        # FFT over spatial dims
        x_ft = jnp.fft.fft2(x, axes=(-2, -1))

        # Prepare output in Fourier domain
        out_ft = jnp.zeros((B, self.out_channels, H, W), dtype=x_ft.dtype)

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W)

        # Multiply low-frequency modes with learned weights
        # x_ft[..., :m1, :m2] shape -> (B, in_ch, m1, m2)
        # weight shape -> (in_ch, out_ch, m1, m2)
        x_sub = x_ft[:, : self.in_channels, :m1, :m2]
        w_sub = self.weight[:, : self.out_channels, :m1, :m2]
        # einsum over input channel dimension
        # result shape -> (B, out_ch, m1, m2)
        out_sub = jnp.einsum("bimn,iomn->bomn", x_sub, w_sub)
        out_ft = out_ft.at[:, : self.out_channels, :m1, :m2].add(out_sub)

        # Return physical space (real part)
        x = jnp.fft.ifft2(out_ft, axes=(-2, -1)).real
        if added_batch:
            return x[0]
        return x


class FNO(eqx.Module):
    """A compact 2D FNO closure

    Behavior:
    - Lift input channels -> `width` via a 1x1 conv.
    - Repeat `depth` blocks of (SpectralConv2d + pointwise conv) with a nonlinear activation.
    - Project back to `out_channels` with a 1x1 conv.

    Notes:
    - Accepts both (C,H,W) and (B,C,H,W) inputs for the moment.
    """
    input_proj: eqx.nn.Conv2d
    spec_layers: Sequence[SpectralConv2d]
    w_layers: Sequence[eqx.nn.Conv2d]
    final: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels = 1,
        out_channels = 1,
        width = 64,
        modes1 = 16,
        modes2 = 16,
        depth = 4,
        key= jax.random.PRNGKey(0),
        **kwargs,
    ):
        keys = jax.random.split(key, 2 * depth + 2)
        k0 = keys[0]
        self.input_proj = eqx.nn.Conv2d(in_channels, width, kernel_size=1, key=k0)

        spec_layers = []
        w_layers = []
        for i in range(depth):
            ks = keys[1 + i]
            kw = keys[1 + depth + i]
            spec_layers.append(SpectralConv2d(width, width, modes1, modes2, key=ks))
            w_layers.append(eqx.nn.Conv2d(width, width, kernel_size=1, key=kw))

        self.spec_layers = spec_layers
        self.w_layers = w_layers

        kfinal = keys[-1]
        self.final = eqx.nn.Conv2d(width, out_channels, kernel_size=1, key=kfinal)

    def __call__(self, q):
        # Accept (C,H,W) or (B,C,H,W). Equinox `Conv2d` in this codebase
        # expects unbatched inputs of rank 3 (C,H,W). Some callers pass a
        # leading singleton batch dim (1,C,H,W) — handle that by squeezing
        # and re-adding it at the end so shapes are preserved.
        readd_batch = False
        x = q
        if x.ndim == 4:
            if x.shape[0] == 1:
                x = x[0]
                readd_batch = True
            else:
                raise ValueError("FNO does not support batched inputs with batch>1 in this codebase")

        if x.ndim != 3:
            raise ValueError("Input must be (C,H,W) or (1,C,H,W)")

        x = x.astype(jnp.float32)

        # Project to working width using 1x1 conv (expects C,H,W)
        x = self.input_proj(x)

        for spec, w in zip(self.spec_layers, self.w_layers):
            x_spec = spec(x)
            x_local = w(x)
            x = x_spec + x_local
            x = jax.nn.gelu(x)

        x = self.final(x)

        if readd_batch:
            return x[None, ...]
        return x
