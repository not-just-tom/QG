"""Diagnostics helpers pulled out of solver for clarity and reuse.

Functions are JAX-friendly where appropriate and accept numpy/jax arrays.
"""
from __future__ import annotations

from typing import Tuple
import jax.numpy as jnp
from jax.numpy.fft import rfftn


def compute_ke(psi: jnp.ndarray, grid) -> float:
    """Compute total kinetic energy: 0.5 * mean(u^2 + v^2) * dx * dy

    Arguments:
        psi: streamfunction in physical space
        grid: `Grid` instance providing dx, dy
    Returns:
        KE as a float
    """
    # u = dψ/dy, v = -dψ/dx (finite-diff approx consistent with other code)
    u = jnp.gradient(psi, grid.dy, axis=-2)
    v = -jnp.gradient(psi, grid.dx, axis=-1)
    ke = 0.5 * jnp.mean(u ** 2 + v ** 2) * grid.dx * grid.dy
    return float(ke)


def compute_enstrophy(zeta: jnp.ndarray, grid) -> float:
    enst = 0.5 * jnp.mean(zeta ** 2) * grid.dx * grid.dy
    return float(enst)


def ke_spectrum(psi: jnp.ndarray, grid) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return kinetic energy spectrum E(k) and bin centers.

    Note: lightweight implementation that uses spectral relations.
    """
    # spectral streamfunction
    psih = rfftn(psi, axes=(-2, -1))
    # isotropic bins
    Kmag = grid.Kmag
    # spectral KE density: 0.5 * |ψ̂|^2 * k^2 / 2 (depends on normalization). This is a utility.
    E_spec = 0.5 * jnp.abs(psih) ** 2 * (Kmag ** 2)
    # Create radial bins
    k_flat = jnp.ravel(Kmag)
    E_flat = jnp.ravel(E_spec)
    kmax = float(jnp.max(k_flat))
    nbins = int(min(64, max(8, int(kmax))))
    bins = jnp.linspace(0.0, kmax, nbins + 1)
    kbins = 0.5 * (bins[:-1] + bins[1:])
    E_k = jnp.zeros_like(kbins)
    # naive binning (vectorized): sum into bins
    for i in range(nbins):
        mask = (k_flat >= bins[i]) & (k_flat < bins[i + 1])
        E_k = E_k.at[i].set(jnp.sum(E_flat[mask]))
    return E_k, kbins


def forcing_stats(forcing_h, grid) -> dict:
    """Return quick statistics for forcing (spectral and physical)."""
    f_spec_max = float(jnp.max(jnp.abs(forcing_h)))
    f_spec_mean = float(jnp.mean(jnp.abs(forcing_h)))
    # physical
    f_phys = jnp.fft.irfftn(forcing_h, s=(grid.ny, grid.nx))
    f_phys_max = float(jnp.max(jnp.abs(f_phys)))
    f_phys_mean = float(jnp.mean(jnp.abs(f_phys)))
    return {
        "f_spec_max": f_spec_max,
        "f_spec_mean": f_spec_mean,
        "f_phys_max": f_phys_max,
        "f_phys_mean": f_phys_mean,
    }


def check_nans(arr) -> int:
    """Return number of NaNs in array (int)."""
    return int(jnp.isnan(arr).sum())
