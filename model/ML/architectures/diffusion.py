import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional


class SpectralDenoiser(eqx.Module):
    """Per-mode MLP denoiser applied to spectral coefficients.
    Operates on vectors of length 2*C (real+imag per channel) and is
    broadcasted across modes (H,W) and batch dims.
    """
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    act = staticmethod(jax.nn.gelu)

    def __init__(self, channels, hidden=128, key=jax.random.PRNGKey(0)):
        k1, k2 = jax.random.split(key, 2)
        self.linear1 = eqx.nn.Linear(2 * channels, hidden, key=k1)
        self.linear2 = eqx.nn.Linear(hidden, 2 * channels, key=k2)

    def __call__(self, x):
        # x shape: (..., 2*C)
        h = self.act(self.linear1(x))
        return self.linear2(h)


class DiffusionGenerator(eqx.Module):
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
    denoiser: SpectralDenoiser
    channels: int
    cutoff: float
    n_steps: int
    alpha: float

    def __init__(self, channels=1, cutoff=8.0, n_steps=3, alpha=0.2, hidden=128, key=jax.random.PRNGKey(0)):
        self.channels = int(channels)
        self.cutoff = float(cutoff)
        self.n_steps = int(n_steps)
        self.alpha = float(alpha)
        self.denoiser = SpectralDenoiser(self.channels, hidden=hidden, key=key)

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
            # prepare real representation (..., 2*C)
            real = jnp.real(high)
            imag = jnp.imag(high)
            # transpose to (..., H, W, C)
            t = jnp.transpose(jnp.concatenate([real, imag], axis=1), (0, 2, 3, 1))  # (B,H,W,2C)
            # apply denoiser (broadcast over leading dims)
            delta = self.denoiser(t)  # (B,H,W,2C)
            # reshape back to (B,2C,H,W) -> split
            delta_t = jnp.transpose(delta, (0, 3, 1, 2))  # (B,2C,H,W)
            delta_real, delta_imag = jnp.split(delta_t, 2, axis=1)
            delta_c = delta_real + 1j * delta_imag
            # apply mask and step-size, clamp magnitude to avoid blow-ups
            update = self.alpha * delta_c
            # optional clipping by magnitude
            mag = jnp.abs(update)
            update = jnp.where(mag > 1e2, update * (1e2 / (mag + 1e-12)), update)
            new_high = high + update * mask
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
