"""Diagnostics helpers pulled out of solver for clarity and reuse.

Functions are JAX-friendly where appropriate and accept numpy/jax arrays.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import rfftn
from collections import defaultdict
from abc import ABC, abstractmethod

def build_diagnostic(name: str):
    mapping = {
        "PV": VorticityDiagnostic,
        "ke_spectrum": KESpectrumDiagnostic,
        "energy": EnergyDiagnostic,
        "enstrophy": EnstrophyDiagnostic,
    }
    cls = mapping.get(name)
    return cls() if cls is not None else None

class Diagnostic(ABC):
    name: str

    @abstractmethod
    def requires(self) -> set[str]:
        """Fields required from solver (e.g. psi, zeta)."""

    @abstractmethod
    def retrieve(self, state, grid=None):
        """Pull the value from TempStates"""

class Recorder:
    def __init__(self, cfg, grid=None):
        self.grid = grid
        diag_cfg = cfg.diagnostics
        self.cadence = int(getattr(diag_cfg, "cadence", 100))

        animate = set(getattr(diag_cfg, "animate", []))
        final = set(getattr(diag_cfg, "final", []))

        animate.add('PV')
        self.animate_names = animate
        self.final_names = final 

        self.diagnostics = {}
        for name in self.animate_names | self.final_names:
            d = build_diagnostic(name)
            if d is None:
                raise ValueError(f"Unknown diagnostic: {name}")
            self.diagnostics[name] = d

        self.animate_buffers = defaultdict(list)
        self.final_buffers   = {}

    def sample(self, state):
        for name, d in self.diagnostics.items():
            # sanity check
            for field in d.requires():
                self._get_field(state, field)
            value = d.retrieve(state, grid=self.grid)
            value = jax.device_get(value)

            if name in self.animate_names:
                self.animate_buffers[name].append(value)

            if name in self.final_names:
                self.final_buffers[name] = value

    def _get_field(self, full_state, field):
        # first try TempStates
        if hasattr(full_state, field):
            return getattr(full_state, field)

        # then try inner State
        if hasattr(full_state.state, field):
            return getattr(full_state.state, field)

        raise AttributeError(f"Field '{field}' not found in TempStates or State")




class VorticityDiagnostic(Diagnostic):
    name = "PV"

    def requires(self):
        return {"q"}  # vorticity

    def retrieve(self, state, grid=None):
        return state.q

# the rest need working on vvv

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
