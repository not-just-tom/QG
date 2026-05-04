import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional


# Note: learned denoiser removed — use simple spectral damping instead.


class Diffusion(eqx.Module):
    """Spectral iterative refinement generator (prototype).
    - Projects input to Fourier space.
    - Iteratively refines high-k modes via a learned per-mode denoiser.
    - Reconstructs refined field in physical space.
    Params:
      channels: number of physical channels (nz)
      cutoff: radial wavenumber threshold (modes) above which we refine
      n_steps: number of iterative refinement steps (3 recommended)
      alpha: step size multiplier for updates
    """
    channels: int
    cutoff: float
    n_steps: int
    alpha: float

    def __init__(self, channels=1, cutoff=8.0, n_steps=3, alpha=0.2, **kwargs):
        self.channels = int(channels)
        self.cutoff = float(cutoff)
        self.n_steps = int(n_steps)
        self.alpha = float(alpha)
        # No learned denoiser: iterative updates use a simple spectral damping step.

    def _make_mask(self, H, W):
        # radial wavenumber mask in index-space (uses fftfreq scaled by grid size)
        kx = jnp.fft.fftfreq(W) * W
        ky = jnp.fft.fftfreq(H) * H
        KX, KY = jnp.meshgrid(kx, ky, indexing='xy')
        kr = jnp.sqrt(KX**2 + KY**2)
        mask = kr > self.cutoff  # True for high-k modes to refine
        return mask  # shape (H, W), bool

    def __call__(self, q, verbose: Optional[bool]=False):
        # Accept (C,H,W) or (B,C,H,W)
        added_batch = False
        x = q
        if x.ndim == 3:
            x = x[None, ...]
            added_batch = True
        if x.ndim != 4:
            raise ValueError("Input must be (C,H,W) or (B,C,H,W)")
        B, C, H, W = x.shape
        assert C == self.channels

        x = x.astype(jnp.float32)

        # Fourier transform (complex)
        x_ft = jnp.fft.fft2(x, axes=(-2, -1))
        # mask shape (H, W)
        mask2d = self._make_mask(H, W)
        mask = mask2d[None, None, :, :]  # broadcast (B, C, H, W)

        # split into low/high
        high = x_ft * mask
        low = x_ft * (~mask)

        def step_fn(high, _i):
            # shape high: (B, C, H, W) complex64
            # Simple spectral damping update in place of learned denoiser:
            # update = -alpha * high  -> new_high = (1 - alpha) * high
            update = -self.alpha * high
            # clamp magnitude of update to avoid blow-ups
            mag = jnp.abs(update)
            update = jnp.where(mag > 1e2, update * (1e2 / (mag + 1e-12)), update)
            new_high = high + update
            return new_high, None

        # iterate n_steps
        high_curr = high
        for i in range(self.n_steps):
            high_curr, _ = step_fn(high_curr, i)
            if verbose:
                # compute norms
                total_energy = jnp.sum(jnp.abs(x_ft) ** 2)
                high_energy = jnp.sum(jnp.abs(high_curr) ** 2)
                low_energy = jnp.sum(jnp.abs(low) ** 2)
                print(f"step {i+1}/{self.n_steps}: high_energy={float(high_energy):.6e}, total_energy={float(total_energy):.6e}, high_frac={float(high_energy/total_energy):.6e}")

        out_ft = low + high_curr
        out = jnp.fft.ifft2(out_ft, axes=(-2, -1)).real.astype(jnp.float32)
        if added_batch:
            return out[0]
        return out
