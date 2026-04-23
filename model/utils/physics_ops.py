"""Pure numerical physics operations for the QG model.

No plotting, saving, or diagnostic logic lives here.
All functions accept and return numpy arrays.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# PV inversion
# ---------------------------------------------------------------------------

def invert_pv_to_psi(q: np.ndarray, grid) -> np.ndarray:
    """Invert PV field q to streamfunction ψ via spectral method.

    Solves ∇²ψ = q  →  ψ̂(k) = −q̂(k) / k²

    Args:
        q:    (..., ny, nx) array of potential vorticity.
        grid: object with scalar attributes ``dx``, ``dy``.

    Returns:
        psi: same shape as *q*.
    """
    q = np.asarray(q, dtype=float)
    ny, nx = q.shape[-2], q.shape[-1]

    kx = np.fft.rfftfreq(nx, d=grid.dx) * 2.0 * np.pi   # (nx//2+1,)
    ky = np.fft.fftfreq(ny, d=grid.dy) * 2.0 * np.pi    # (ny,)
    kx2, ky2 = np.meshgrid(kx, ky, indexing="xy")       # (ny, nx//2+1)
    k2 = kx2 ** 2 + ky2 ** 2
    k2[0, 0] = 1.0  # avoid division by zero; DC component zeroed below

    q_hat = np.fft.rfftn(q, axes=(-2, -1))
    psi_hat = -q_hat / k2
    psi_hat[..., 0, 0] = 0.0  # enforce zero-mean streamfunction

    return np.fft.irfftn(psi_hat, s=(ny, nx), axes=(-2, -1))


# ---------------------------------------------------------------------------
# Velocity recovery
# ---------------------------------------------------------------------------

def velocity_from_psi(psi: np.ndarray, grid) -> tuple[np.ndarray, np.ndarray]:
    """Recover velocity fields from streamfunction.

    u = ∂ψ/∂y,   v = −∂ψ/∂x

    Args:
        psi:  (..., ny, nx).
        grid: object with scalar attributes ``dx``, ``dy``.

    Returns:
        (u, v): each the same shape as *psi*.
    """
    psi = np.asarray(psi, dtype=float)
    u = np.gradient(psi, float(grid.dy), axis=-2)
    v = -np.gradient(psi, float(grid.dx), axis=-1)
    return u, v


# ---------------------------------------------------------------------------
# Isotropic KE spectrum
# ---------------------------------------------------------------------------

def isotropic_ke_spectrum(u: np.ndarray, v: np.ndarray, grid) -> dict:
    """Compute the isotropic kinetic energy spectrum E(k).

    Args:
        u, v: (..., ny, nx) velocity arrays.
        grid: object with scalar attributes ``dx``, ``dy``.

    Returns:
        dict with:
            ``k``:  (nk,) 1-D wavenumber array (integer bin centres).
            ``E``:  (..., nk) spectral energy array (leading dims match u/v).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ny, nx = u.shape[-2], u.shape[-1]

    kx = np.fft.rfftfreq(nx, d=grid.dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=grid.dy) * 2.0 * np.pi
    kx2, ky2 = np.meshgrid(kx, ky, indexing="xy")
    kmag = np.sqrt(kx2 ** 2 + ky2 ** 2)           # (ny, nx//2+1)

    uh = np.fft.rfftn(u, axes=(-2, -1))
    vh = np.fft.rfftn(v, axes=(-2, -1))
    ke_spec = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)  # (..., ny, nx//2+1)

    kmax = int(kmag.max())
    nbins = kmax + 1
    k = np.arange(nbins, dtype=float)

    lead_shape = u.shape[:-2]
    E = np.zeros((*lead_shape, nbins))

    # Flatten spatial dims for vectorised bin selection
    kmag_flat = kmag.ravel()                          # (ny*(nx//2+1),)
    ke_flat = ke_spec.reshape(*lead_shape, -1)        # (..., ny*(nx//2+1))

    bin_idx = np.floor(kmag_flat).astype(int)
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    for i in range(nbins):
        mask = bin_idx == i
        if mask.any():
            E[..., i] = ke_flat[..., mask].sum(axis=-1)

    return {"k": k, "E": E}


# ---------------------------------------------------------------------------
# Scalar energy / enstrophy integrals
# ---------------------------------------------------------------------------

def compute_ke(psi: np.ndarray, grid) -> float:
    """Total kinetic energy: 0.5 * mean(u² + v²) * dx * dy."""
    u, v = velocity_from_psi(np.asarray(psi), grid)
    return float(0.5 * np.mean(u ** 2 + v ** 2) * grid.dx * grid.dy)


def compute_enstrophy(zeta: np.ndarray, grid) -> float:
    """Total enstrophy: 0.5 * mean(ζ²) * dx * dy."""
    return float(0.5 * np.mean(np.asarray(zeta) ** 2) * grid.dx * grid.dy)
