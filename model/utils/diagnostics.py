"""Diagnostics helpers pulled out of solver for clarity and reuse.

Functions are JAX-friendly where appropriate and accept numpy/jax arrays.
"""
from __future__ import annotations

from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import rfftn
from collections import defaultdict
from abc import ABC, abstractmethod

class Diagnostic(ABC):
    name: str

    @abstractmethod
    def requires(self) -> set[str]:
        """Fields required from solver (e.g. psi, zeta)."""

    @abstractmethod
    def compute(self, state, grid):
        """Pure JAX computation."""

    @abstractmethod
    def reduce(self, value):
        """Optional reduction (e.g. radial binning)."""

class Recorder:
    def __init__(self, cfg):
        diag_cfg = cfg.diagnostics

        self.cadence = int(getattr(diag_cfg, "cadence", 1))

        self.animate_names = set(getattr(diag_cfg, "animate", []))
        self.final_names   = set(getattr(diag_cfg, "final", []))

        self.diagnostics = []
        self._diag_map = {}
        for name in self.animate_names | self.final_names:
            d = build_diagnostic(name)
            if d is not None:
                self.diagnostics.append(d)
                self._diag_map[d.name] = d
            else:
                raise ValueError(f"Unknown diagnostic: {name}")

        # storage
        self.animate_buffers = {}              # name -> last value
        self.final_buffers = defaultdict(list) # name -> time series

    def sample(self, model):
        if model.n % self.cadence != 0:
            return

        fields = model.fields
        grid = model.grid

        for d in self.diagnostics:
            val = d.compute(fields, grid)
            val = jax.device_get(val)

            if d.name in self.animate_names:
                self.animate_buffers[d.name] = val

            if d.name in self.final_names:
                self.final_buffers[d.name].append(val)

    def write(self, path):
        import zarr, os
        # Ensure parent dir exists
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        root = zarr.open(path, mode="w")

        for name, values in self.final_buffers.items():
            arr = np.asarray(values)
            root.create_array(
                name,
                data=arr,
                chunks=True,
                overwrite=True
            )
        for name, val in self.animate_buffers.items():
            root.create_array(
                name,
                data=np.asarray(val),
                overwrite=True
            )

class VorticityDiagnostic(Diagnostic):
    name = "zeta"

    def requires(self):
        return {"zeta"}

    def compute(self, state, grid):
        return state["zeta"]
    
    def reduce(self, value):
        # I think this is fine?
        return value


class KESpectrumDiagnostic(Diagnostic):
    name = "ke_spectrum"

    def requires(self):
        return {"psi"}

    def compute(self, fields, grid):
        psi = fields["psi"]
        psih = rfftn(psi, axes=(-2, -1), norm="ortho")

        uh = -1j * grid.KY * psih
        vh =  1j * grid.KX * psih

        KE2D = 0.5 * (jnp.abs(uh)**2 + jnp.abs(vh)**2)

        return KE2D
    
    def reduce(self, KE2D, grid):
        kmag = np.asarray(grid.Kmag).astype(int)
        KE = np.asarray(KE2D)

        kmax = kmag.max()
        E_k = np.bincount(kmag.ravel(), weights=KE.ravel(), minlength=kmax+1)
        return E_k


class EnergyDiagnostic(Diagnostic):
    name = "energy"

    def requires(self):
        return {"psi"}

    def compute(self, fields, grid):
        psi = fields["psi"]
        return compute_ke(psi, grid)

    def reduce(self, value, grid=None):
        return value


class EnstrophyDiagnostic(Diagnostic):
    name = "enstrophy"

    def requires(self):
        return {"zeta"}

    def compute(self, fields, grid):
        zeta = fields["zeta"]
        return compute_enstrophy(zeta, grid)

    def reduce(self, value, grid=None):
        return value


def build_diagnostic(name: str):
    mapping = {
        "zeta": VorticityDiagnostic,
        "ke_spectrum": KESpectrumDiagnostic,
        "energy": EnergyDiagnostic,
        "enstrophy": EnstrophyDiagnostic,
    }
    cls = mapping.get(name)
    return cls() if cls is not None else None


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
